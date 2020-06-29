# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://stackoverflow.com/questions/33853802/mouse-click-events-on-a-live-stream-with-opencv-and-python

from gluoncv import model_zoo, data, utils
#from matplotlib import pyplot as plt
import mxnet as mx

import cv2
import numpy as np
import json
import urllib


def deal_specialchar_in_url(istr):
    s = istr.find('//')
    e = istr.find('@')
    # print(s, e)
    subs = istr[s+2: e]
    name, ps = subs.split(':')
    encoded_name_ps = urllib.parse.quote_plus(name)+ ":" + urllib.parse.quote_plus(ps)
    result = 'rtsp://' + encoded_name_ps + istr[e:]
    # print(result)
    return result



# video_urls = [
#     "rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=1&subtype=0",
#     "rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=3&subtype=0"
    
 
# ]


video_urls = [
    "rtsp://admin:yxgl$666@192.168.200.210:554/Streaming/Channels/1",
    "rtsp://admin:yxgl$666@192.168.200.211:554/Streaming/Channels/1",
    "rtsp://admin:yxgl$666@192.168.200.232:554/Streaming/Channels/1",

    "rtsp://admin:yxgl$666@192.168.200.233:554/Streaming/Channels/1",

]

# video_urls = [
#     "rtsp://admin:yxgl$666@192.168.200.182:554/Streaming/Channels/1",
#     "rtsp://admin:yxgl$666@192.168.200.183:554/Streaming/Channels/1",
#     "rtsp://admin:yxgl$666@192.168.200.190:554/Streaming/Channels/1",

#     "rtsp://admin:yxgl$666@192.168.200.204:554/Streaming/Channels/1",
#     "rtsp://admin:12345@192.168.200.76:554/Streaming/Channels/1",
#     "rtsp://admin:12345@192.168.200.91:554/Streaming/Channels/1",   
#     "rtsp://admin:12345@192.168.200.95:554/Streaming/Channels/1",
#     "rtsp://admin:12345@192.168.200.97:554/Streaming/Channels/1"
# ]
saved_config_filename = 'i8url_rect_dict.json'
url_rect_dict = {}

rect = (0,0,0,0)
startPoint = False
endPoint = False

print('选择区域时，请保持左上 --> 右下 的方式绘制对角线 ')
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
    addr = deal_specialchar_in_url(video_url)
    cap = cv2.VideoCapture(addr)
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
        # print('frame', type(frame))

        new_frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

        x, orig_img = data.transforms.presets.yolo.transform_test(new_frame,  short=512, max_size=2000)
        frameI = orig_img
        # print('frameI', type(frameI))
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        

        # cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', on_mouse)    

        #drawing rectangle
        if startPoint == True and endPoint == True:
            cv2.rectangle(frameI, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
            # cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
            print(' 正方形位置:', (rect[0], rect[1]), (rect[2], rect[3]))
        # cv2.imshow('frame',frame)
        cv2.imshow('frame', frameI[...,::-1])

        key = cv2.waitKey(waitTime) 
        # escape 键
        if key == 27:
            # break
            print('setting ', video_url, ' square:', (rect[0], rect[1]), (rect[2], rect[3]))
            results = [rect[0], rect[1], rect[2], rect[3]]
            print(type(results), 'results', results, results[0])
            # 如果点击者先右后左的话 保证左侧的点在前，右侧的点在后
            if results[0] > results[2]:                
                print('左右位置互换')
                results =  [results[2], results[3], results[0], results[1]]
                print(results)
            # 如果由下向上的话, 保存成另外2个对角点，方便后续与目标方框比较包含关系
            if results[1] > results[3]:
                print('对角线互换')
                results = [results[0], results[3], results[2], results[1]]
                print(results)

            url_rect_dict[addr] = results
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
print('完成设置不同摄像头检测区域：')
print(url_rect_dict) 