'''import cv2 # importing the library 

img = cv2.imread("download.png")   ## read hte image 

cv2.imwrite("sampledownload.png",img)    ## save a image with different name and extension 

cv2.imshow("project_computer_vision",img)
cv2.waitKey(0)

cv2.destroyAllWindows()'''


'''import cv2 # import opencv library
vs = cv2.VideoCapture(0)  # initalize camera
while True:  # infinte loop
    _,img =vs.read()  ## read the frame from the camera
    cv2.imshow("VideoStream", img) # show a frame
    # below line , frame will show continously untill you press a button on keyboard
    key = cv2.waitKey(1) & 0xFF
     # record my key press - Hex
    if key == ord("a"):
        break # infinte loop will be broken
vs.release() # relase the caemra
cv2.destroyAllWindows() #all opended windows will be closed .'''

'''import cv2
import time  # import time lib
import imutils   ## import imutils

file_img = cv2.imread("C:/computer vision and deep learning/img.jpg")
cam = cv2.VideoCapture(2)  ## initialize camera
time.sleep(1) ## giving 1 sec delay

firstFrame=None
area = 800  # threshold for how much change can be noticed in moving object 

while True:
    _, cam_img =cam.read()  # reading frame  from the camera 
    text = "Normal"     ## no moving object detection .
    
    # pre_precossing
    img = imutils.resize(cam_img, width=500) # resize the frame to 500 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert colour image to gray scale image
    gaussianImg =cv2.GaussianBlur(grayImg, (21,21),0) ## smootheening 

    # save the first frame into  the firstframe variable
     # from the 2nd iteration it won't go inside  this if condition
    if firstFrame is None:
            firstFrame = gaussianImg
            continue
    
    imgDiff = cv2.absdiff(firstFrame,gaussianImg) # difference b/w first bg frame with the current frame
    threshImg = cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1] #detected region will be converted into Binary
    threshImg = cv2.dilate(threshImg,None,iterations=2)
    
    cnts = cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
                 continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x , y),(x + w, y + h),(0,255,0),2)
        text = "Moving Object Detected done"
    print(text)
    cv2.putText(img,text, (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255),2)
    cv2.imshow("cameraFeed",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()'''

import cv2
import time
import imutils

# Read the image from the file
file_img = cv2.imread("C:/computer vision and deep learning/img.jpg")

# Initialize the camera
cam = cv2.VideoCapture(2)
time.sleep(1)  # Giving 1 sec delay

firstFrame = None
area = 800  # Threshold for how much change can be noticed in moving object 

while True:
    _, cam_img = cam.read()  # Reading frame from the camera

    text = "Normal"  # No moving object detection.
    
    # Pre-processing
    img = imutils.resize(cam_img, width=500)  # Resize the frame to 500 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert color image to grayscale image
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)  # Smoothing

    # Save the first frame into the firstFrame variable
    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg)  # Difference between the first bg frame and the current frame
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]  # Detected region will be converted into Binary
    threshImg = cv2.dilate(threshImg, None, iterations=2)
    
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object Detected done"

    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

    
    
    
        



