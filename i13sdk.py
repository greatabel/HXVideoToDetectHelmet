#test.py
#coding=utf-8
import HKIPcamera
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


from i13rabbitmq import sender

# print(cv2.__version__)
# 海康摄像头
def HKI_base64(ip, name, pw, queueid):
    HKIPcamera.init(ip, name, pw)
    print('here0', ip, name, pw)
    # HKIPcamera.getfram()
    while 1:
        print(ip, name, pw, '#'*5)
        # frame = HKIPcamera.getframe()
        frame = HKIPcamera.getframe()
        # time.sleep(3)
        print('1.1', type(frame))
        img = np.array(frame)
        # print('img=', img)
        # cv2.imshow('Camera', img)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sender('localhost', img, queueid)
        cv2.namedWindow('Camera', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('Camera', img)
        cv2.waitKey(1)
        # print('here')
        time.sleep(3)
    HKIPcamera.release()

if __name__ == '__main__':
    # rtsp://admin:test1024@10.248.10.111:554/h264/ch33/main/av_stream
    # HKI_base64('10.248.10.111','admin','test1024')
    # source /etc/profile

    # camera ip:
    HKI_base64('10.248.10.43','admin','huaxin12345')
