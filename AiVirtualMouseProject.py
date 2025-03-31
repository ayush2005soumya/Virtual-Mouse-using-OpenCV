import cv2
import mediapipe as mp
import pyautogui
import math

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

# Dragging flag
dragging = False

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmark.landmark

            # Coordinates for fingers
            index_tip = landmarks[8]     # Index Finger Tip
            thumb_tip = landmarks[4]     # Thumb Tip
            middle_tip = landmarks[12]   # Middle Finger Tip

            # Convert to screen coordinates
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)
            screen_x = int(index_tip.x * screen_width)
            screen_y = int(index_tip.y * screen_height)

            # Move mouse with index finger
            pyautogui.moveTo(screen_x, screen_y)

            # Distance calculations
            thumb_dist = math.hypot(index_x - int(thumb_tip.x * w), index_y - int(thumb_tip.y * h))
            middle_dist = math.hypot(index_x - int(middle_tip.x * w), index_y - int(middle_tip.y * h))

            # Pinch click and drag
            if thumb_dist < 40:
                if not dragging:
                    dragging = True
                    pyautogui.mouseDown()
                cv2.putText(frame, 'Dragging', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if dragging:
                    dragging = False
                    pyautogui.mouseUp()

            # Scroll gesture: when index and middle finger are apart
            if middle_dist < 50:
                cv2.putText(frame, 'Scrolling', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if index_y < middle_tip.y * h:
                    pyautogui.scroll(20)  # Scroll up
                else:
                    pyautogui.scroll(-20)  # Scroll down

    # Display
    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()