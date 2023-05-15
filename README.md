# moving-object-detection-with-open-cv
Moving object detection is an important task in computer vision, and it can be achieved using various techniques. One of the most popular methods is using OpenCV, an open-source computer vision library, to detect motion in a video stream


Here is a quick summary of how to use OpenCV to do moving object detection:

Capture the video stream: To capture the video stream from a camera or video file, use OpenCV's VideoCapture function.


Grayscale frame conversion: To streamline the processing, convert the colour frames to grayscale.

Determine the history: To determine the video stream's backdrop, use the BackgroundSubtractor class in OpenCV.

To distinguish the foreground from the background, apply a thresholding method.

Apply morphological operations: To reduce noise and fine-tune the foreground mask, use morphological procedures like erosion and dilation.

Find the moving things: To find the moving objects in the foreground, use OpenCV's contour detection technique.

Draw bounding boxes: Draw bounding boxes around the detected objects to highlight their location in the video stream.
