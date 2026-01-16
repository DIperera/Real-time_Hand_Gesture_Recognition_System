"""
Gesture Detector Module
Simple hand gesture detection using OpenCV
Compatible with all Python versions - no MediaPipe needed!
"""

import cv2
import numpy as np


class GestureDetector:
    """Detects hand gestures using OpenCV color and contour detection"""

    def __init__(self):
        """Initialize hand detector"""
        print("✅ Hand detector initialized (OpenCV-based)")
        print("💡 TIP: Use good lighting and plain background for best results")

        # Multiple HSV ranges for better skin detection
        self.lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        self.lower_skin2 = np.array([0, 40, 80], dtype=np.uint8)
        self.upper_skin2 = np.array([25, 255, 255], dtype=np.uint8)

    def detect_gesture(self, frame):
        """
        Detect hand gesture using color and contour detection

        Args:
            frame: Input image frame (BGR)

        Returns:
            tuple: (annotated_frame, gesture_name)
        """
        gesture_name = None

        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for skin color (using multiple ranges for better detection)
        mask1 = cv2.inRange(hsv, self.lower_skin1, self.upper_skin1)
        mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to clean up
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest contour (assumed to be hand)
            hand_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(hand_contour)

            # Only process if contour is large enough
            if area > 5000:
                # Draw contour
                cv2.drawContours(frame, [hand_contour], 0, (0, 255, 0), 2)

                # Get convex hull
                hull = cv2.convexHull(hand_contour, returnPoints=False)

                if len(hull) > 3 and len(hand_contour) > 3:
                    try:
                        defects = cv2.convexityDefects(hand_contour, hull)

                        if defects is not None:
                            # Count fingers
                            finger_count = self.count_fingers_from_defects(
                                defects, hand_contour)

                            # Get bounding rectangle
                            x, y, w, h = cv2.boundingRect(hand_contour)
                            cv2.rectangle(
                                frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Classify gesture
                            gesture_name = self.classify_by_finger_count(
                                finger_count)

                            # Display finger count
                            cv2.putText(frame, f"Fingers: {finger_count}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    except:
                        pass  # Ignore errors in defect calculation

        return frame, gesture_name

    def count_fingers_from_defects(self, defects, contour):
        """Count extended fingers using improved convexity defects analysis"""
        finger_count = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Calculate triangle sides
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # Calculate semi-perimeter
            s_tri = (a + b + c) / 2

            # Calculate area using Heron's formula
            if s_tri * (s_tri - a) * (s_tri - b) * (s_tri - c) < 0:
                continue

            area = np.sqrt(s_tri * (s_tri - a) * (s_tri - b) * (s_tri - c))

            # Avoid division by zero
            if b * c == 0:
                continue

            # Calculate angle using cosine rule
            cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_angle = min(1.0, max(-1.0, cos_angle))  # Clamp to valid range
            angle = np.arccos(cos_angle)

            # More strict angle check and depth check for better accuracy
            # Depth: distance from the farthest point to the convex hull
            depth = d / 256.0

            # Count as finger if:
            # 1. Angle is less than 90 degrees (sharp angle)
            # 2. The defect is deep enough (not noise)
            # 3. Triangle has reasonable area
            if angle <= np.pi / 2 and depth > 20 and area > 100:
                finger_count += 1

        # Finger count = defects + 1 (because defects are gaps between fingers)
        # But cap at 5 fingers max
        result = min(finger_count + 1, 5)

        # If we detect 6+ fingers, it's likely noise - default to open palm (5)
        if finger_count > 5:
            result = 5

        return result

    def classify_by_finger_count(self, finger_count):
        """
        Classify gesture based on finger count

        Args:
            finger_count: Number of extended fingers

        Returns:
            str: Gesture name
        """
        gestures = {
            0: "fist",
            1: "pointing_up",
            2: "peace",
            3: "three",
            4: "four",
            5: "five"
        }

        return gestures.get(finger_count, "Unknown")
