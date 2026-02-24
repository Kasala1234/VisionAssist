import cv2
import numpy as np
import speech_recognition as sr
import time

typed_text = ""
highlight_time = 0
HIGHLIGHT_DURATION = 1.5

# Simple keyboard layout just for display, no key clicking here
keys_row1 = ['GO'] + list("QWERTYUIOP")
keys_row2 = list("ASDFGHJKL")
keys_row3 = list("ZXCVBNM") + ['Space', 'Back']

def draw_keyboard(img):
    screen_width = img.shape[1]
    gap = 10
    key_widths = [
        int((screen_width - 100 - (len(keys_row1) - 1) * gap) / len(keys_row1)),
        int((screen_width - 100 - (len(keys_row2) - 1) * gap) / len(keys_row2)),
        int((screen_width - 100 - (len(keys_row3) - 1) * gap) / len(keys_row3))
    ]
    y_positions = [120, 190, 260]
    key_rows = [keys_row1, keys_row2, keys_row3]

    for row_idx, (keys_row, key_width, y) in enumerate(zip(key_rows, key_widths, y_positions), start=1):
        for i, key in enumerate(keys_row):
            x = 50 + i * (key_width + gap)
            w, h = key_width, 60
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            font_scale = 0.9
            cv2.putText(img, key, (x + int(w / 4), y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

recognizer = sr.Recognizer()
mic = sr.Microphone()

print("Voice Dictation Keyboard started. Speak naturally.")
print("Say 'go' to exit.")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with mic as source:
    recognizer.adjust_for_ambient_noise(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]

    # Display typed text
    cv2.rectangle(frame, (50, 20), (1200, 80), (220, 220, 220), -1)
    cv2.putText(frame, typed_text, (60, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Draw keyboard
    draw_keyboard(frame)

    cv2.imshow("Voice Dictation Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    try:
        with mic as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=4, phrase_time_limit=4)
        command = recognizer.recognize_google(audio)
        command_lower = command.lower().strip()
        print(f"Recognized: {command}")

        if command_lower == "go":
            print("Exit command received. Exiting...")
            break
        else:
            # Append recognized sentence with a space
            typed_text += command + " "

    except sr.WaitTimeoutError:
        # Just keep listening
        pass
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Recognition error; {e}")

cap.release()
cv2.destroyAllWindows()
