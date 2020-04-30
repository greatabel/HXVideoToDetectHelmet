# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://stackoverflow.com/questions/33853802/mouse-click-events-on-a-live-stream-with-opencv-and-python
import cv2
import numpy as np


rect = (0,0,0,0)
startPoint = False
endPoint = False

cap = cv2.VideoCapture("rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=3&subtype=0")
#cap = cv2.VideoCapture("test2.dav")
#cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
# cap.set(3, 1920)
# cap.set(4, 1080)
# cap.set(cv2.CAP_PROP_FPS,5)
#cap = cv2.VideoCapture(0)

def on_mouse(event,x,y,flags,params):

    global rect,startPoint,endPoint

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if startPoint == True and endPoint == True:
            startPoint = False
            endPoint = False
            rect = (0, 0, 0, 0)

        if startPoint == False:
            rect = (x, y, 0, 0)
            startPoint = True
        elif endPoint == False:
            rect = (rect[0], rect[1], x, y)
            endPoint = True

waitTime = 50

#Reading the first frame
(grabbed, frame) = cap.read()

while(cap.isOpened()):

    (grabbed, frame) = cap.read()

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)    

    #drawing rectangle
    if startPoint == True and endPoint == True:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
        print(' 正方形位置:', (rect[0], rect[1]), (rect[2], rect[3]))
    cv2.imshow('frame',frame)

    key = cv2.waitKey(waitTime) 

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()