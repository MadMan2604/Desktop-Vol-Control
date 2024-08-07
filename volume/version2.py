import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Defined Global Variables
x1 = y1 = x2 = y2 = 0
webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get volume control interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)  # Flip horizontally for mirror effect
    frame_width, frame_height, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:
                    cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255), thickness=3)
                    x1 = x
                    y1 = y
                if id == 4:
                    cv2.circle(img=image, center=(x, y), radius=5, color=(0, 0, 255), thickness=3)
                    x2 = x
                    y2 = y

            # Use distance between two points to calculate volume control
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 4
            cv2.line(image, (x1, y1), (x2, y2), (125, 125, 125), 5)

            current_volume = volume.GetMasterVolumeLevelScalar()
            if dist > 50:
                # Increase volume
                new_volume = min(1.0, current_volume + 0.05)
            else:
                # Decrease volume
                new_volume = max(0.0, current_volume - 0.05)

            volume.SetMasterVolumeLevelScalar(new_volume, None)

    cv2.imshow("Hand volume control using python", image)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
