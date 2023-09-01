import cv2
import math
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, hands_image_processing, draw=True):
        hands_image_processingRGB = cv2.cvtColor(
            hands_image_processing, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(hands_image_processingRGB)
        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        hands_image_processing, handLM, self.mpHands.HAND_CONNECTIONS)

        return hands_image_processing

    def findPosition(self, hands_image_processing, handNo=0):
        landmarks_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                height, width, _ = hands_image_processing.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmarks_list.append([id, cx, cy])
        return landmarks_list

    def findDistance(self, point1, point2):
        distance = math.sqrt((point2[2]-point1[2])
                             ** 2 + (point2[1]-point1[1])**2)
        return distance

    def pixels_to_cm(self, pixels, focal_length_cm, object_width_cm):
        CM = (object_width_cm * focal_length_cm) / pixels
        return CM

    def getFingers(self, hands_image_processing, flip=False, handNo=0):
        fingers = [1, 1, 1, 1, 1]

        landmarks_list = self.findPosition(
            hands_image_processing, handNo=handNo)

        if landmarks_list:

            WRIST = landmarks_list[0]
            thumb_base = landmarks_list[2]
            thumb_ip = landmarks_list[3]
            thumb_tip = landmarks_list[4]

            hand, orientation_hand = self.identifyHand(
                hands_image_processing, flip=flip)

            is_thumb_closed_front = (thumb_tip[1] < thumb_ip[1]) if not flip else (
                thumb_tip[1] > thumb_ip[1])

            is_thumb_closed_back = not is_thumb_closed_front

            if hand == "Right":
                if (orientation_hand == "Front" and is_thumb_closed_front) or (
                        orientation_hand == "Back" and is_thumb_closed_back):
                    fingers[0] = 0

            else:  # Left Hand
                if (orientation_hand == "Front" and is_thumb_closed_back) or (
                        orientation_hand == "Back" and is_thumb_closed_front):
                    fingers[0] = 0

            for i in range(1, 5):
                finger_base = landmarks_list[4*i + 2]
                finger_tip = landmarks_list[4*i + 4]

                if self.findDistance(WRIST, finger_base) > self.findDistance(finger_tip, WRIST):
                    fingers[i] = 0
        else:
            raise Exception("NO Hand Found")

        return fingers

    def identifyHand(self, hands_image_processing, flip=False, handNo=0):
        landmarks_list = self.findPosition(
            hands_image_processing, handNo=handNo)

        if landmarks_list:
            wrist_x = landmarks_list[0][1]
            thumb_tip_x = landmarks_list[3][1]
            index_finger_x = landmarks_list[5][1]
            pinky_mcp_x = landmarks_list[17][1]
            palm_center_y = (landmarks_list[0][2] + landmarks_list[9][2]) / 2

        hand = "Right" if ((not flip and thumb_tip_x > pinky_mcp_x and wrist_x < index_finger_x) or
                           (flip and thumb_tip_x < pinky_mcp_x and wrist_x > index_finger_x)) else "Left"

        orientation = "Front" if ((not flip and wrist_x < palm_center_y) or (
            flip and wrist_x > palm_center_y)) else "Back"

        if hand == "Right":
            if orientation == "Back":
                hand = "Left"
        elif hand == "Left":
            if orientation == "Front":
                hand = "Right"
                orientation = "Back"
            else:
                orientation = "Front"

        return hand, orientation
