# Object_detection-using-yolo

Object detection using yolo on custom dataset and coco data

# Prerequisites
Before running the script, make sure you have the following dependencies installed:

Ultralytics (pip install -U git+https://github.com/ultralytics/yolov5.git)
OpenCV (pip install opencv-python)
cvzone (pip install cvzone)



# Description
The script performs the following steps:

Captures video from the webcam.
Uses the YOLO model from Ultralytics to perform real-time object detection.
Draws bounding boxes, confidence scores, and class names on the detected objects.
Displays the video feed with object detection results.

# Classes
The script is configured to detect a variety of classes, including but not limited to:

person
bicycle
car
motorbike
...
Please refer to the classNames list in the script for the complete list of classes.


# Keyboard Shortcut
Press 'q' to quit the application.

# Credits
Ultralytics YOLO for the YOLO model.
OpenCV for computer vision functionalities.
cvzone for drawing utilities.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
