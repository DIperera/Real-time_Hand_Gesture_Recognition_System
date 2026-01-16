# 🖐️ Hand Gesture Recognition System

This is a real-time hand gesture recognition application that uses your webcam to detect and classify hand gestures instantly.

## Technologies Used:

- **OpenCV (cv2)** - Core computer vision library for image processing and hand detection
- **NumPy** - Numerical computing for array operations
- **Python 3.8+** - Programming language

## Key Features:

- Real-time detection with live webcam feed
- Recognizes 6 different gestures based on finger count (0-5 fingers)
- Displays gesture names with emoji representations (✊ Fist, ☝️ Pointing Up, ✌️ Peace, etc.)
- FPS counter for performance monitoring
- Skin color-based detection using HSV color space and contour analysis
- No ML training required - works immediately out of the box
- Lightweight - only requires OpenCV and NumPy

## How It Works:

The system uses computer vision techniques including:

- HSV color space conversion for skin tone detection
- Morphological operations to clean noise
- Contour detection to find the hand shape
- Convexity defects analysis to count extended fingers
- Gesture classification based on finger count

## Project Structure:

- `main.py` - Application entry point with webcam loop
- `gesture_detector.py` - Core gesture detection logic
- `config.py` - Gesture mappings and settings
- `hand_landmarker.task` - Pre-trained model file
