# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://stackoverflow.com/questions/33853802/mouse-click-events-on-a-live-stream-with-opencv-and-python
import cv2
import numpy as np
import json


video_urls = [
    "rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=3&subtype=0"
]
saved_config_filename = 'i8url_rect_dict.json'
url_rect_dict = {}

rect = (0,0,0,0)
startPoint = False
endPoint = False


print('设置好当前摄像头的检测区域后，按 escape 键，程序将保存设置，进入下一个摄像头设置！')

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


for video_url in video_urls:
    video_flag = True
    rect = (0, 0, 0, 0)

    cap = cv2.VideoCapture(video_url)
    #cap = cv2.VideoCapture("test2.dav")
    #cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
    # cap.set(3, 1920)
    # cap.set(4, 1080)
    # cap.set(cv2.CAP_PROP_FPS,5)
    #cap = cv2.VideoCapture(0)
    waitTime = 50

    #Reading the first frame
    (grabbed, frame) = cap.read()

    while(cap.isOpened() and video_flag): 

        (grabbed, frame) = cap.read()

        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', on_mouse)    

        #drawing rectangle
        if startPoint == True and endPoint == True:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
            print(' 正方形位置:', (rect[0], rect[1]), (rect[2], rect[3]))
        cv2.imshow('frame',frame)

        key = cv2.waitKey(waitTime) 
        # escape 键
        if key == 27:
            # break
            print('setting ', video_url, ' square:', (rect[0], rect[1]), (rect[2], rect[3]))
            url_rect_dict[video_url] = (rect[0], rect[1]), (rect[2], rect[3])
            video_flag = False

cap.release()
cv2.destroyAllWindows()

# This saves your dict
with open(saved_config_filename, 'w') as f:
    # passing an indent parameter makes the json pretty-printed
    json.dump(url_rect_dict, f, indent=2) 


# url_rect_dict = {}
# # This loads your dict
# with open(saved_config_filename, 'r') as f:
#     url_rect_dict = json.load(f)

print(url_rect_dict) 