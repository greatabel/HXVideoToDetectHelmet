#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import datetime
import numpy as np
import traceback
import time
import urllib

class process:
    def __init__(self):
        # 所需参数
        self.detect_area = [0, 1, 0, 1]  # 截取的图像区域，范围为0~1，长宽
        self.interval = 1/15  # 采样间隔（单位：秒）
        self.resize_height = 600  # 调整大小后的图像高
        self.__threshold = 20  # 图像二值化的阀值
        self.__kernel = np.ones((3, 3), np.uint8)  # 图像膨胀相关参数
        # 其他参数（无需调整）
        self.resize_width = -1
        # 写文件初始化
        self.__fps = 25
        self.__video_name = 'output_video.mp4'
        self.__size = [0, 0]
        self.__count = 0

    def set_pic_size(self, width, height):
        self.__size = [width, height]
        self.__video_write = cv2.VideoWriter(self.__video_name, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
        print("width=%d,high=%d"%(width, height))

    def set_resize_length(self, length_input):
        self.resize_width = length_input
        print("resize_width=%d,resize_high=%d"%(self.resize_width,self.resize_height))

    def deal_specialchar_in_url(self, istr):
        s = istr.find('//')
        e = istr.rfind('@')
        subs = istr[s+2: e]
        name, ps = subs.split(':')
        encoded_name_ps = urllib.parse.quote_plus(name)+ ":" + urllib.parse.quote_plus(ps)
        result = 'rtsp://' + encoded_name_ps + istr[e:]
        return result

    def write_frame(self, frame_input):
        file_name = "image/" + str(self.__count) + ".jpg"
        self.__count += 1
        cv2.imwrite(file_name, frame_input)

    def write_video(self, frame_input):
        self.__video_write.write(frame_input)

    def detect_function(self, detect_input, detect_input_last):
        # 缩小图像
        start_tm_1 = time.time()
        frame_resize = cv2.resize(detect_input,(self.resize_width, self.resize_height),interpolation=cv2.INTER_CUBIC)
        # 将彩色图转换为灰度图
        frame_gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        # 将图像与上一帧的图像作差，并作二值化处理
        ret, diff = cv2.threshold(cv2.absdiff(detect_input_last, frame_gray), self.__threshold, 255, cv2.THRESH_BINARY)
        # 直方图图像增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(diff)
        diff_clahe = cv2.GaussianBlur(cl1, (7, 7), 0)
        # 对图像进行sobel边缘检测
        x_func = cv2.Sobel(diff_clahe, cv2.CV_16S, 1, 0)
        y_func = cv2.Sobel(diff_clahe, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x_func)
        absY = cv2.convertScaleAbs(y_func)
        diff_sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        # 对边缘检测后的图像作二值化处理
        ret_threshold, diff_threshold = cv2.threshold(diff_sobel, self.__threshold, 255, cv2.THRESH_BINARY)
        # 对图像进行形态学闭运算，并进行中值滤波
        diff_median = cv2.medianBlur(cv2.morphologyEx(diff_threshold, cv2.MORPH_OPEN, self.__kernel), 5)
        # 寻找封闭区域并填充
        #opencv3
        # img, contours, hierarchy = cv2.findContours(diff_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #opencv4
        contours, hierarchy = cv2.findContours(diff_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_lower = np.array([0,140,140])
        red_upper = np.array([15,255,255])
        # 遍历区域列表，筛选面积较大区域，判定为人像区域，提取对应区域外接矩形框的信息
        for c in contours:
            area = cv2.contourArea(c)
            if area > 10000:
                x, y, w, h = cv2.boundingRect(c)
                h_w_ratio = h/w
                if 1.3<=h_w_ratio<=3:
                    crop_hsv = cv2.cvtColor(frame_resize[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
                    frame_p = cv2.GaussianBlur(frame_resize[y:y+h, x:x+w],(5,5),0)
                    mask = cv2.inRange(crop_hsv, red_lower, red_upper)
                    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
                    mask = cv2.GaussianBlur(mask, (3, 3), 0)
                    res = cv2.bitwise_and(frame_p,frame_p,mask=mask)
                    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                    num = 0
                    for c in cnts:
                        area = cv2.contourArea(c)
                        x1, y1, w1, h1 = cv2.boundingRect(c)
                        print('area=', area)
                        if area > 100:
                            num = num+1
                            cv2.rectangle(res, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                    if num>=1:
                        cv2.rectangle(frame_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame_resize, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #cv2.imshow("crop_hsv",crop_hsv)
                    #cv2.imshow("mask",mask)
                    #cv2.imshow("res",res)
                    #cv2.waitKey(0)

        cv2.imshow("image", frame_resize)
        cv2.waitKey(1)
        #self.write_frame(frame_resize)
        # self.write_video(detect_input)

if __name__ == '__main__':
    # 初始化姿态检测主函数
    process = process()

    # 获取视频

    rtsp_url = "rtsp://admin:yxgl$666@192.168.200.215:554/Streaming/Channels/1"
    url = process.deal_specialchar_in_url(rtsp_url)
    videoCapture = cv2.VideoCapture(url)

    # videoCapture = cv2.VideoCapture('/media/liujin/disk/project/data/safty_belt/2.mp4')

    # 获得码率及尺寸
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    # 设置视频尺寸
    process.set_pic_size(size[0], size[1])

    print("fps=",fps)
    start_time = datetime.datetime.now()
    # 读帧
    success, frame = videoCapture.read()
    # 获取图像高宽
    y_origin, x_origin = frame.shape[0:2]

    # 选择需要作检测的区域
    print("x_origin=%d,y_origin=%d"%(x_origin,y_origin))
    frame_detect = frame[int(process.detect_area[0] * y_origin): int(process.detect_area[1] * y_origin), int(
        process.detect_area[2] * x_origin): int(process.detect_area[3] * x_origin)]
    y, x = frame_detect.shape[0:2]
    # 记录缩小后图像的长度
    print("x=%d,y=%d,resize_height=%d"%(x,y,process.resize_height))
    process.set_resize_length(int(x / (y / process.resize_height)))
    # 将彩色图转换为灰度图并存储到frame_gray_last中
    frame_gray_last = cv2.cvtColor(cv2.resize(frame_detect, (process.resize_width, process.resize_height),
                                              interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    count = 0
    while True:
        try:
            count += 1
            if count > 10:  # 跳过前几帧
                # 安全衣识别核心函数部分
                start_time = time.time()
                process.detect_function(frame_detect, frame_gray_last)
                end_time = time.time() 
                print("frame is:", 1.0/(end_time-start_time) )
            success, frame = videoCapture.read()  # 获取下一帧
            if frame is None:
                break
            # 截取下一帧的检测区域
            frame_gray_last = cv2.cvtColor(cv2.resize(frame_detect, (process.resize_width, process.resize_height),
                                                      interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
            frame_detect = frame[int(process.detect_area[0] * y_origin): int(process.detect_area[1] * y_origin), int(
        process.detect_area[2] * x_origin): int(process.detect_area[3] * x_origin)]
        except Exception as e:
            traceback.print_exc()
            cv2.destroyAllWindows()
            videoCapture.release()
            exit(0)
    end_time = datetime.datetime.now()
    print((end_time - start_time))
    cv2.destroyAllWindows()
    videoCapture.release()


