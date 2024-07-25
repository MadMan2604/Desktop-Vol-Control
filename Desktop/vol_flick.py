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

# To track the previous positions of the index finger
prev_index_x = None
flick_threshold = 50  # Threshold for detecting a flick, in pixels

# Define thumb and index finger landmarks
THUMB_TIP = 4
INDEX_TIP = 8

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
            
            # Get coordinates of thumb tip and index fingertip
            thumb_tip_x = int(hand_landmarks.landmark[THUMB_TIP].x * w)
            thumb_tip_y = int(hand_landmarks.landmark[THUMB_TIP].y * h)
            index_tip_x = int(hand_landmarks.landmark[INDEX_TIP].x * w)
            index_tip_y = int(hand_landmarks.landmark[INDEX_TIP].y * h)
            
            # Initialize prev_index_x if not set
            if prev_index_x is None:
                prev_index_x = index_tip_x
            
            # Calculate the movement delta
            dx = index_tip_x - prev_index_x
            
            # Check if the movement is significant enough to consider it a flick
            if abs(dx) > flick_threshold:
                if dx > 0:
                    pyautogui.hotkey('winleft', 'ctrl', 'left')  # Flick right: switch to the next desktop
                else:
                    pyautogui.hotkey('winleft', 'ctrl', 'right')  # Flick left: switch to the previous desktop
                
                # Reset prev_index_x after a flick to prevent repeated triggering
                prev_index_x = None
            else:
                # Update prev_index_x only if no significant flick is detected
                prev_index_x = index_tip_x
            
            # Calculate the distance between thumb tip and index fingertip
            distance = np.sqrt((index_tip_x - thumb_tip_x) ** 2 + (index_tip_y - thumb_tip_y) ** 2)
            
            # Threshold for detecting a "pinch" gesture
            pinch_threshold = 50
            
            if distance < pinch_threshold:
                print("Volume Control Activated")
                # You can add code here to control the volume
                # For example:
                # pyautogui.press('volumeup') or pyautogui.press('volumedown')
    
    # Display the frame
    cv2.imshow('Hand Tracking', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
