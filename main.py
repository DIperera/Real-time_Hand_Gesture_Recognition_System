"""
Hand Gesture Recognition Application
Real-time hand gesture detection using webcam with emoji name output
"""

import cv2
from utils.gesture_detector import GestureDetector
from config import GESTURE_NAMES, WINDOW_NAME
import time


def main():
    """Main function to run the hand gesture recognition application"""

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize gesture detector
    detector = GestureDetector()

    # Variables for FPS calculation
    prev_time = 0

    print("🚀 Hand Gesture Recognition Started!")
    print("Press 'q' to quit the application")
    print("-" * 50)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from webcam")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect hand gestures
        frame, gesture_name = detector.detect_gesture(frame)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Display gesture name prominently
        if gesture_name and gesture_name != "Unknown":
            # Large gesture name display
            cv2.putText(frame, f"Gesture: {gesture_name}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

            # Display emoji representation
            emoji_text = GESTURE_NAMES.get(gesture_name, "")
            cv2.putText(frame, emoji_text,
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display instructions
        cv2.putText(frame, "Press 'q' to quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Show the frame
        cv2.imshow(WINDOW_NAME, frame)

        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Exiting application...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Application closed successfully")


if __name__ == "__main__":
    main()
