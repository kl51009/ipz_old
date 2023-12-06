import mediapipe as mp
import numpy as np
import cv2
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

vec = lambda x: np.array([x.x, x.y, x.z])

def draw_circle(img, v, r, color):
    h, w, _ = image.shape
    x, y = v[0] * w, v[1] * h
    print(v[0], v[1])
    print(x, y)
    return cv2.circle(image, (int(x), int(y)), r, color, 2)

TIPS = [
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
]

MCPS = [
    mp_hands.HandLandmark.INDEX_FINGER_MCP,
    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    mp_hands.HandLandmark.PINKY_MCP,
    mp_hands.HandLandmark.RING_FINGER_MCP,
]

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        show_fingers = True
        hand_state = {}

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extracting landmarks
                landmarks = hand_landmarks.landmark

                # Wrist landmark
                wrist = vec(landmarks[mp_hands.HandLandmark.WRIST])

                # Finger distances
                d_tips = list()
                d_mcps = list()

                for mcp_i, tip_i in zip(MCPS, TIPS):
                    mcp = vec(landmarks[mcp_i])
                    tip = vec(landmarks[tip_i])
                    d_tips.append(np.linalg.norm(tip - wrist))
                    d_mcps.append(np.linalg.norm(mcp - wrist))

                    if show_fingers:
                        draw_circle(image, mcp, 12, (0, 0, 200))
                        draw_circle(image, tip, 12, (0, 200, 0))
                        draw_circle(image, wrist, 12, (200, 0, 0))

                d_tips = np.array(d_tips)
                d_mcps = np.array(d_mcps)

                # Determine if the hand is open or closed based on the distance
                if d_tips.sum() > d_mcps.sum():
                    hand_state[i] = False
                else:
                    hand_state[i] = True

        for i, state in hand_state.items():
            if state:
                cv2.putText(image, f'{i} Closed', (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, f'{i} Open', (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
