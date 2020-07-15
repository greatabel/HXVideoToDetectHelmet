# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://stackoverflow.com/questions/33853802/mouse-click-events-on-a-live-stream-with-opencv-and-python

from gluoncv import model_zoo, data, utils
#from matplotlib import pyplot as plt
import mxnet as mx

import cv2
import numpy as np
import json
import urllib
from i11process_frame import deal_specialchar_in_url


# 海康
# 12345
# yxgl$666
# yxgl!123456
# yxgl123456

# 库房
# input_str= 'rtsp://admin:yxgl$666@192.168.200.182:554'
# 电气室
# input_str= 'rtsp://admin:yxgl$666@192.168.200.183:554'
# 电气室
# input_str= 'rtsp://admin:yxgl$666@192.168.200.184:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.185:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.186:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.187:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.188:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.189:554'
# 下料口
# input_str= 'rtsp://admin:yxgl$666@192.168.200.190:554'

# camera is error 
# input_str= 'rtsp://admin:yxgl$666@192.168.200.191:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.192:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.193:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.194:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.197:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.198:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.202:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.205:554'

# camear is error(error while decoding MB 58 3, bytestream -34)
# input_str= 'rtsp://admin:yxgl$666@192.168.200.203:554'

# cameara is not online
# input_str= 'rtsp://admin:yxgl$666@192.168.200.200:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.201:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.206:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.207:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.208:554'

# 下料口
# input_str= 'rtsp://admin:yxgl$666@192.168.200.195:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.196:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.199:554'
# 水浠下楼梯
# input_str= 'rtsp://admin:yxgl$666@192.168.200.204:554'


# input_str= 'rtsp://admin:12345@192.168.200.51:554'
# input_str= 'rtsp://admin:12345@192.168.200.52:554'
# input_str= 'rtsp://admin:12345@192.168.200.53:554'
# input_str= 'rtsp://admin:12345@192.168.200.54:554'
# input_str= 'rtsp://admin:12345@192.168.200.55:554'
# input_str= 'rtsp://admin:12345@192.168.200.56:554'

# wait
# input_str= 'rtsp://admin:yxgl$666@192.168.200.57:554'

# input_str= 'rtsp://admin:12345@192.168.200.58:554'
# input_str= 'rtsp://admin:12345@192.168.200.59:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.60:554'
# input_str= 'rtsp://admin:12345@192.168.200.61:554'

# not online:
# input_str= 'rtsp://admin:12345@192.168.200.62:554'


# input_str= 'rtsp://admin:12345@192.168.200.63:554'
# input_str= 'rtsp://admin:12345@192.168.200.64:554'
# 大厅：
# input_str= 'rtsp://admin:12345@192.168.200.65:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.66:554'
# input_str= 'rtsp://admin:12345@192.168.200.67:554'

# not online:
# input_str= 'rtsp://admin:12345@192.168.200.68:554'

# input_str= 'rtsp://admin:12345@192.168.200.70:554'
# input_str= 'rtsp://admin:12345@192.168.200.71:554'
# 传送带
# input_str= 'rtsp://admin:12345@192.168.200.72:554'
# 监控 煤区
# input_str= 'rtsp://admin:12345@192.168.200.73:554'
# input_str= 'rtsp://admin:12345@192.168.200.74:554'
# input_str= 'rtsp://admin:12345@192.168.200.75:554'
# 山脚下
# input_str= 'rtsp://admin:12345@192.168.200.76:554'
# 水处理
# input_str= 'rtsp://admin:12345@192.168.200.77:554'
# input_str= 'rtsp://admin:12345@192.168.200.78:554'
# input_str= 'rtsp://admin:12345@192.168.200.79:554'
# input_str= 'rtsp://admin:12345@192.168.200.80:554'
# 楼梯
# input_str= 'rtsp://admin:12345@192.168.200.81:554'
# input_str= 'rtsp://admin:12345@192.168.200.82:554'
# input_str= 'rtsp://admin:12345@192.168.200.83:554'

# not online
# input_str= 'rtsp://admin:12345@192.168.200.84:554'
# 监控室
# input_str= 'rtsp://admin:12345@192.168.200.85:554'
# input_str= 'rtsp://admin:yxgl123456@192.168.200.86:554'
# 楼道
# input_str= 'rtsp://admin:12345@192.168.200.87:554'
# 机器旁边
# input_str= 'rtsp://admin:12345@192.168.200.88:554'

#  not online
# input_str= 'rtsp://admin:12345@192.168.200.89:554'

# input_str= 'rtsp://admin:12345@192.168.200.90:554'
# 沙漏旁边
# input_str= 'rtsp://admin:12345@192.168.200.91:554'
# input_str= 'rtsp://admin:12345@192.168.200.92:554'
# 某办公室
# input_str= 'rtsp://admin:12345@192.168.200.93:554'

#not online:
# input_str= 'rtsp://admin:12345@192.168.200.94:554'
# 泥浆
# input_str= 'rtsp://admin:12345@192.168.200.95:554'
# input_str= 'rtsp://admin:12345@192.168.200.96:554'
# 水洗机器旁
# input_str= 'rtsp://admin:12345@192.168.200.97:554'
# input_str= 'rtsp://admin:12345@192.168.200.98:554'

# wait
# input_str= 'rtsp://admin:yxgl!123456@192.168.200.99:554'
# 地磅
# input_str= 'rtsp://admin:yxgl$666@192.168.200.101:554'
# input_str= 'rtsp://admin:12345@192.168.200.102:554'

# not line
# input_str= 'rtsp://admin:12345@192.168.200.103:554'

# input_str= 'rtsp://admin:12345@192.168.200.105:554'
# 江边
# input_str= 'rtsp://admin:12345@192.168.200.106:554'
# 进厂
# input_str= 'rtsp://admin:12345@192.168.200.107:554'
# input_str= 'rtsp://admin:12345@192.168.200.109:554'

# 旋转的水上
# input_str= 'rtsp://admin:yxgl$666@192.168.200.110:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.111:554'
# 电机室
# input_str= 'rtsp://admin:yxgl$666@192.168.200.112:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.113:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.114:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.210:554/Streaming/Channels/1'
# input_str= 'rtsp://admin:huaxin12345@10.248.10.43:554:554/Streaming/Channels/1'

# 后期新加摄像头
# input_str= 'rtsp://admin:yxgl$666@192.168.200.209:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.210:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.211:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.212:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.213:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.214:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.215:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.216:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.217:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.218:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.219:554'
#  ...
# input_str= 'rtsp://admin:yxgl$666@192.168.200.233:554'

video_urls = [
    # start 本地测试
    ["admin", "admin123", "10.248.10.100:554", 1, 'dahua'],  # ipv4
    ["admin", "admin123", "10.248.10.100:554", 3, 'dahua'],

    ["admin", "huaxin12345", "10.248.10.43:554", 1, "hik"],
    # end   本地测试

    # ['admin', 'yxgl$666','192.168.200.182:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.183:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.184:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.185:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.186:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.188:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.189:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.190:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.195:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.196:554',1, 'hik'], # 10
    # ['admin', 'yxgl$666','192.168.200.199:554',1, 'hik'],
    # ['admin', 'yxgl$666','192.168.200.204:554',1, 'hik'],    
    


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

saved_config_filename = 'i11url_rect_dict.json'
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


for rtsp_obj in video_urls:
    # 大华的情况 ：
    if rtsp_obj[4] == 'dahua':
        video_url = "rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" \
            % (rtsp_obj[0], rtsp_obj[1], rtsp_obj[2], rtsp_obj[3])
    # 网络摄像头是海康:
    elif rtsp_obj[4] == 'hik':
        video_url = "rtsp://%s:%s@%s/Streaming/Channels/%d" % (rtsp_obj[0], rtsp_obj[1], rtsp_obj[2], rtsp_obj[3])


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
