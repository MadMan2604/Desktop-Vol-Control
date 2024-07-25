import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start capturing from the webcam
cap = cv2.VideoCapture(0)

# To track the previous positions
prev_x = None
flick_threshold = 80  # Threshold for detecting a wrist flick

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    h, w, c = frame.shape
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hands
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates of the wrist (landmark 0)
            wrist_x = int(hand_landmarks.landmark[0].x * w)
            wrist_y = int(hand_landmarks.landmark[0].y * h)
            
            # Initialize prev_x if not set
            if prev_x is None:
                prev_x = wrist_x
            
            # Calculate the movement delta
            dx = wrist_x - prev_x
            
            # Check if the movement is significant enough to consider it a flick
            if abs(dx) > flick_threshold:
                if dx > 0:
                    pyautogui.hotkey('winleft', 'ctrl', 'left')  # Flick right: switch to the next desktop
                else:
                    pyautogui.hotkey('winleft', 'ctrl', 'right')  # Flick left: switch to the previous desktop
                
                # Reset prev_x after a flick to prevent repeated triggering
                prev_x = None
            else:
                # Update prev_x only if no significant flick is detected
                prev_x = wrist_x
    
    # Display the frame
    cv2.imshow('Hand Tracking', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
