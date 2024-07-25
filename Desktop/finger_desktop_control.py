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

# Threshold distance to detect if fingers are touching
touch_threshold = 0.05  # Adjust based on your camera setup

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
            
            # Get coordinates of thumb tip (landmark 4), index tip (landmark 8), and middle tip (landmark 12)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            
            # Calculate the distances between thumb and index/middle fingertips
            dist_thumb_index = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
            )
            dist_thumb_middle = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y])
            )
            
            # Check if the thumb is touching the index finger
            if dist_thumb_index < touch_threshold:
                print("left")
                pyautogui.hotkey('winleft', 'ctrl', 'right')  # Switch to the next desktop
                # Optional: Add a small sleep to prevent multiple triggers
                pyautogui.sleep(1)
            
            # Check if the thumb is touching the middle finger
            elif dist_thumb_middle < touch_threshold:
                print("right")
                pyautogui.hotkey('winleft', 'ctrl', 'left')  # Switch to the previous desktop
                # Optional: Add a small sleep to prevent multiple triggers
                pyautogui.sleep(1)
    
    # Display the frame
    cv2.imshow('Hand Tracking', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
