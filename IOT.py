import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import serial

# ================== Arduino Serial Setup ==================
# Change COM port accordingly: "COM3" for Windows, "/dev/ttyUSB0" for Linux
arduino = serial.Serial('COM3', 9600, timeout=1)

def send_iot_command(command):
    arduino.write((command + "\n").encode())
    print(f"Sent to Arduino: {command}")

# ================== Mediapipe Setup ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ================== UI State ==================
current_screen = "menu"  # menu, hand, eye
paused = False
color_pos = 0
last_action_time = 0.0
last_hand_trigger_time = 0
HAND_COOLDOWN = 2
blink_timestamps = deque(maxlen=10)
ear_buffer = deque(maxlen=8)
initial_ear = None

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ================== Helper Functions ==================
def draw_menu(img):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 100), (72, 61, 139), -1)
    cv2.putText(img, "SELECT INPUT MODE", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    cv2.rectangle(img, (30, 130), (w//2 - 40, 250), (50, 205, 50), -1)
    cv2.putText(img, "HAND MODE", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    cv2.putText(img, "(Show 2 Fingers)", (60, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    cv2.rectangle(img, (w//2 + 20, 130), (w - 30, 250), (30, 144, 255), -1)
    cv2.putText(img, "EYE MODE", (w//2 + 50, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    cv2.putText(img, "(Blink 2 Times)", (w//2 + 50, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

def draw_iot_boxes(img, color_pos):
    h, w = img.shape[:2]
    colors = [("RED", (0,0,255)), ("GREEN", (0,255,0)), ("BLUE", (255,0,0)), ("GO", (200,200,200))]
    box_positions = []
    gap = 40
    box_width = (w - 100 - (len(colors)-1)*gap) // len(colors)
    y, h_box = 250, 150

    for i, (label, color) in enumerate(colors):
        x = 50 + i*(box_width+gap)
        is_selected = x < color_pos < x + box_width

        # Draw base box
        cv2.rectangle(img, (x,y), (x+box_width,y+h_box), color, -1)
        cv2.rectangle(img, (x,y), (x+box_width,y+h_box), (0,0,0), 3)

        # If selected → apply strong highlight
        if is_selected:
            cv2.rectangle(img, (x,y), (x+box_width,y+h_box), (0,255,255), 8)  # thick border
            overlay = img.copy()
            cv2.rectangle(overlay, (x,y), (x+box_width,y+h_box), (0,255,255), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Text with shadow for visibility
        cv2.putText(img, label, (x+30,y+90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 6)  # shadow
        cv2.putText(img, label, (x+30,y+90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

        box_positions.append((label,x,y,box_width,h_box))
    return box_positions

def get_current_box(color_pos, box_positions):
    for label,x,y,w,h in box_positions:
        if x < color_pos < x+w:
            return label
    return None

def count_fingers_once(hand_results, frame):
    if not hand_results.multi_hand_landmarks:
        return 0
    total = 0
    for hand_landmarks in hand_results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        tips = [8, 12, 16, 20]
        fingers_up = 0
        for tip in tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                fingers_up += 1
        total += fingers_up
    return total

def compute_ear(landmarks, eye_indices, img_w, img_h):
    coords = [(int(landmarks[i].x*img_w), int(landmarks[i].y*img_h)) for i in eye_indices]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (A+B)/(2.0*C) if C!=0 else 0

# ================== Main Loop ==================
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

print("Starting IoT Control. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img_h,img_w = frame.shape[:2]
    current_time = time.time()

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)
    total_fingers = count_fingers_once(hand_results, frame)

    # MENU SCREEN
    if current_screen == "menu":
        draw_menu(frame)
        if total_fingers==2 and (current_time-last_hand_trigger_time)>HAND_COOLDOWN:
            current_screen="hand"; color_pos=0; paused=False; last_action_time=current_time; last_hand_trigger_time=current_time
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                left_ear = compute_ear(face_landmarks.landmark, LEFT_EYE, img_w, img_h)
                right_ear = compute_ear(face_landmarks.landmark, RIGHT_EYE, img_w, img_h)
                avg_ear = (left_ear+right_ear)/2.0
                ear_buffer.append(avg_ear)
                if initial_ear is None and len(ear_buffer)==ear_buffer.maxlen:
                    initial_ear = np.mean(ear_buffer)
                if initial_ear and avg_ear<initial_ear*0.75:
                    blink_timestamps.append(current_time)
                recent_blinks = [t for t in blink_timestamps if current_time-t<1.0]
                if len(recent_blinks)==2:
                    current_screen="eye"; color_pos=0; paused=False; last_action_time=current_time

    # HAND OR EYE SCREEN
    elif current_screen in ["hand","eye"]:
        cv2.putText(frame, f"MODE: {current_screen.upper()}", (50,65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        box_positions = draw_iot_boxes(frame, color_pos)
        current_box = get_current_box(color_pos, box_positions)

        if current_screen=="hand":
            if total_fingers==1 and (current_time-last_action_time)>1.0:
                paused=not paused; last_action_time=current_time
            if total_fingers==2 and (current_time-last_action_time)>1.0:
                if current_box:
                    if current_box=="GO": current_screen="menu"
                    else: send_iot_command(current_box)
                last_action_time=current_time

        if current_screen=="eye" and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                left_ear = compute_ear(face_landmarks.landmark, LEFT_EYE, img_w, img_h)
                right_ear = compute_ear(face_landmarks.landmark, RIGHT_EYE, img_w, img_h)
                avg_ear=(left_ear+right_ear)/2.0
                ear_buffer.append(avg_ear)
                if initial_ear is None and len(ear_buffer)==ear_buffer.maxlen:
                    initial_ear=np.mean(ear_buffer)
                if initial_ear and avg_ear<initial_ear*0.75:
                    blink_timestamps.append(current_time)
                recent_blinks=[t for t in blink_timestamps if current_time-t<1.0]
                if len(recent_blinks)==1 and (current_time-last_action_time)>1.0:
                    paused=True; last_action_time=current_time
                elif len(recent_blinks)==2 and (current_time-last_action_time)>1.0:
                    if current_box:
                        if current_box=="GO": current_screen="menu"
                        else: send_iot_command(current_box)
                    paused=False; last_action_time=current_time

        if not paused:
            color_pos += 5  # slow cursor speed
            if color_pos>img_w-100: color_pos=0

    cv2.imshow("IoT Gesture & Eye Control", frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release()
cv2.destroyAllWindows()
