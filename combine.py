import cv2
import numpy as np
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Keyboard Layout
keys_row1 = list("QWERTYUIOP")
keys_row2 = list("ASDFGHJKL")
keys_row3 = list("ZXCVBNM") + ['Space', 'Back']

typed_text = ""

# Sweep Variables
color_pos = 0
color_speed = 10
current_row = 1
paused = False
last_select_time = time.time()

def draw_keyboard(img, color_pos, current_row):
    key_positions = []
    screen_width = img.shape[1]

    # Draw Keyboard Outer Box
    cv2.rectangle(img, (50, 100), (screen_width - 50, 350), (0,0,0), 3)

    gap = 10
    key_width1 = int((screen_width - 2*50 - (len(keys_row1)-1)*gap) / len(keys_row1))
    key_width2 = int((screen_width - 2*50 - (len(keys_row2)-1)*gap) / len(keys_row2))
    key_width3 = int((screen_width - 2*50 - (len(keys_row3)-1)*gap) / len(keys_row3))

    y1 = 120
    y2 = y1 + 70
    y3 = y2 + 70

    # Draw Rows
    for i, key in enumerate(keys_row1):
        x = 50 + i * (key_width1 + gap)
        w, h = key_width1, 60
        highlight = (0,255,255) if current_row == 1 and x < color_pos < x + w else (255,255,255)
        cv2.rectangle(img, (x, y1), (x + w, y1 + h), highlight, -1)
        cv2.rectangle(img, (x, y1), (x + w, y1 + h), (0,0,0), 2)
        cv2.putText(img, key, (x + int(w/4), y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        key_positions.append((key, x, y1, w, h))

    for i, key in enumerate(keys_row2):
        x = 50 + i * (key_width2 + gap)
        w, h = key_width2, 60
        highlight = (0,255,255) if current_row == 2 and x < color_pos < x + w else (255,255,255)
        cv2.rectangle(img, (x, y2), (x + w, y2 + h), highlight, -1)
        cv2.rectangle(img, (x, y2), (x + w, y2 + h), (0,0,0), 2)
        cv2.putText(img, key, (x + int(w/4), y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        key_positions.append((key, x, y2, w, h))

    for i, key in enumerate(keys_row3):
        x = 50 + i * (key_width3 + gap)
        w, h = key_width3, 60
        highlight = (0,255,255) if current_row == 3 and x < color_pos < x + w else (255,255,255)
        cv2.rectangle(img, (x, y3), (x + w, y3 + h), highlight, -1)
        cv2.rectangle(img, (x, y3), (x + w, y3 + h), (0,0,0), 2)
        cv2.putText(img, key, (x + int(w/6), y3 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        key_positions.append((key, x, y3, w, h))

    return key_positions

def count_fingers(hand_landmarks):
    fingers_up = 0
    tips = [8, 12, 16, 20]
    for tip in tips:
        tip_pos = hand_landmarks.landmark[tip]
        mcp_pos = hand_landmarks.landmark[tip - 2]
        if tip_pos.y < mcp_pos.y:
            fingers_up += 1
    return fingers_up

def get_current_key(color_pos, key_positions, current_row):
    row_offset = 0
    if current_row == 2:
        row_offset = len(keys_row1)
    elif current_row == 3:
        row_offset = len(keys_row1) + len(keys_row2)

    row_length = len(keys_row1) if current_row == 1 else len(keys_row2) if current_row == 2 else len(keys_row3)

    for i in range(row_offset, row_offset + row_length):
        key, x, y, w, h = key_positions[i]
        if x < color_pos < x + w:
            return key
    return None

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (50, 20), (1200, 80), (220, 220, 220), -1)
    cv2.putText(frame, typed_text, (60, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

    key_positions = draw_keyboard(frame, color_pos, current_row)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_up = count_fingers(hand_landmarks)

            current_time = time.time()
            if fingers_up == 1 and current_time - last_select_time > 0.5:
                paused = not paused  # Toggle pause
                last_select_time = current_time

            elif fingers_up == 2 and current_time - last_select_time > 0.5:
                key = get_current_key(color_pos, key_positions, current_row)
                if key:
                    if key == 'Space':
                        typed_text += ' '
                    elif key == 'Back':
                        typed_text = typed_text[:-1]
                    else:
                        typed_text += key
                    last_select_time = current_time

    if not paused:
        color_pos += color_speed
        # Check if row is completed
        screen_width = frame.shape[1]
        row_width = screen_width - 100  # Considering margins
        if color_pos > row_width:
            color_pos = 0
            current_row += 1
            if current_row > 3:
                current_row = 1

    cv2.imshow("Virtual Keyboard - Single Sweep", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
