#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import datetime
import numpy as np


class fall_detection:
    def __init__(self):
        # 所需参数
        self.detect_area = [0.1, 0.9, 0.1, 0.9]  # 截取的图像区域，范围为0~1，长宽
        self.interval = 0.1  # 采样间隔（单位：秒）
        self.resize_height = 600  # 调整大小后的图像高
        self.__threshold = 5  # 图像二值化的阀值
        self.__kernel = np.ones((5, 5), np.uint8)  # 图像膨胀相关参数
        # 其他参数（无需调整）
        self.resize_length = -1
        # 形态学处理后标志区域的长宽、中心位置、角度信息
        self.__rectangle = []

    def set_resize_length(self, length_input):
        self.resize_length = length_input

    def detect_function(self, detect_input, detect_input_last):
        # 缩小图像
        frame_resize = cv2.resize(detect_input, (self.resize_height, self.resize_length), interpolation=cv2.INTER_CUBIC)
        # 将彩色图转换为灰度图
        frame_gray = cv2.cvtColor(frame_resize, cv2.COLOR_RGB2GRAY)
        # 将图像与上一帧的图像作差，并作二值化处理
        ret, diff = cv2.threshold(cv2.absdiff(detect_input_last, frame_gray), self.__threshold, 255, cv2.THRESH_BINARY)
        # 直方图图像增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        diff_clahe = cv2.GaussianBlur(clahe.apply(diff), (7, 7), 0)
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
        # 寻找最大区域并填充
        contours, hierarchy = cv2.findContours(diff_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 记录矩形区域的长宽中心位置
        if len(self.__rectangle) > 5:
            self.__rectangle.pop(0)
        self.__rectangle.append([rect[1], rect[0], rect[2]])
        print(self.__rectangle)
        # 调用警告判决函数
        self.rectangle_detect()
        # 显示模块
        diff_color = cv2.merge([diff_median, diff_median, diff_median])
        cv2.drawContours(diff_color, [box], 0, [255, 255, 0], 2)
        ###
        cv2.imshow("diff_out", diff_color)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        ###

    # 对矩形的位置进行判断，输出警告标志位
    def rectangle_detect(self):
        warn_rate = 0
        for count_rectangle in range(0, len(self.__rectangle)):
            if 45 > self.__rectangle[2] > -45 and self.__rectangle[0][0] / self.__rectangle[0][1] < 0.6:
                continue
            else:
                warn_rate += 1
        if warn_rate > 3:
            print("fall warn!")
            exit(0)


if __name__ == '__main__':
    # 初始化姿态检测主函数
    detect = fall_detection()

    # 获取视频
    videoCapture = cv2.VideoCapture('video/test.mp4')
    # 获得码率及尺寸
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    start_time = datetime.datetime.now()  # 记录开始时间

    # 读帧
    success, frame = videoCapture.read()
    # 对第一张图像进行处理
    # 获取图像长宽
    x_origin, y_origin = frame.shape[0:2]

    # 选择需要作检测的区域
    frame_detect = frame[int(detect.detect_area[2] * x_origin): int(detect.detect_area[3] * x_origin), int(
        detect.detect_area[0] * y_origin): int(detect.detect_area[1] * y_origin)]
    x, y = frame_detect.shape[0:2]
    # 记录缩小后图像的长度
    detect.set_resize_length(int(x / (y / detect.resize_height)))
    # 将彩色图转换为灰度图并存储到frame_gray_last中
    frame_gray_last = cv2.cvtColor(cv2.resize(frame_detect, (detect.resize_height, detect.resize_length),
                                              interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2GRAY)

    count = 0
    while count < 1800:
        try:
            count += 1
            if count % int(fps * detect.interval) == 0 and count > 1 and count > 70:  # 跳过第一次循环
                # 烟雾检测主函数
                detect.detect_function(frame_detect, frame_gray_last)
            success, frame = videoCapture.read()  # 获取下一帧
            if frame is None:
                break
            # 截取下一帧的检测区域
            frame_gray_last = cv2.cvtColor(cv2.resize(frame_detect, (detect.resize_height, detect.resize_length),
                                                      interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2GRAY)
            frame_detect = frame[int(detect.detect_area[2] * x_origin): int(detect.detect_area[3] * x_origin), int(
                detect.detect_area[0] * y_origin): int(detect.detect_area[1] * y_origin)]
        except Exception as e:
            exit(0)
    end_time = datetime.datetime.now()
    print((end_time - start_time))
    videoCapture.release()
