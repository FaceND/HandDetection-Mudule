# Hand Detection Module

The Hand Detection Module is a Python library that utilizes the power of computer vision 
and machine learning to accurately detect and analyze human hands in images and videos. 
Whether you're building gesture recognition systems, virtual reality applications, or 
interactive user interfaces, this module provides an easy-to-use interface for identifying 
hand landmarks and gestures.

## Table of Contents

* [Introduction](#introduction)
* [Usage](#usage)
* [Methods](#methods)
* [Examples](#examples)
* [Key Features](#key-features)
* [License](#license)

## Introduction

The `handDetector` class in this module provides methods to perform hand detection and finger counting using the Mediapipe library. It can identify the orientation of the hand (front or back) and count the number of fingers being held up.

## Usage

To use the `handDetector` module, you need to have Python and the required libraries installed. You can install the required libraries using the following command:

```bash
pip install opencv-python mediapipe
```
## Methods
The `handDetector` class provides the following methods:
- `findHands(image, draw=True)`: Finds and marks hands in the given image.
- `findPosition(image, handNo=0)`: Finds the landmarks' positions of a specific hand.
- `findDistance(point1, point2)`: Calculates the distance between two points.
- `pixels_to_cm(pixels, focal_length_cm, object_width_cm)`: Converts pixels to centimeters.
- `getFingers(image, flip=False, handNo=0)`: Detects and counts the fingers in the image.
- `identifyHand(image, flip=False, handNo=0)`: Identifies the orientation of the hand (front or back).

## Examples
```python
import cv2
from handDetectionMudule import handDetector

# Create an instance of the handDetector class
detector = handDetector()

# Load an image or start a video capture
hands_image_processing = cv2.imread("hand_image.jpg")

# Find hands in the image
hands_with_landmarks = detector.findHands(hands_image_processing, draw=True)

# Get the hand and orientation' status
hand, orientation = detector.identifyHand(hands_with_landmarks)

# Get the fingers' status
fingers = detector.getFingers(hands_with_landmarks)

print(orientation, hand, "Hand")
print("Fingers:", fingers)
```

## Key Features
- Accurate hand detection and landmark localization using the MediaPipe library.
- Differentiates between left and right hands, considering their orientation (front or back).
- Detects open and closed fingers, making it suitable for gesture recognition.
- Calculates distances between hand landmarks for more advanced applications.
- Seamless integration with various projects such as robotics, gaming, and interactive multimedia.

The module comes with an intuitive API that allows you to easily extract hand landmarks, determine finger gestures, and identify the orientation of the hand. Whether you're a developer, researcher, or hobbyist, this module is a valuable tool for implementing hand-related functionalities in your projects.

Get started today by integrating the Hand Detection Module into your Python applications. Enhance your projects with accurate hand detection and unlock a new level of interactivity and control.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
