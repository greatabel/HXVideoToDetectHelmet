#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import datetime
import numpy as np
import traceback
import time
import urllib

class color_detection:
    def __init__(self):
        # 所需参数
        self.detect_area = [0, 1, 0, 1]  # 截取的图像区域，范围为0~1，长宽
        self.interval = 1/15  # 采样间隔（单位：秒）
        self.resize_height = 600  # 调整大小后的图像高
        self.__threshold = 20  # 图像二值化的阀值
        self.__kernel = np.ones((3, 3), np.uint8)  # 图像膨胀相关参数
        # 其他参数（无需调整）
        self.resize_length = -1
        # 形态学处理后标志区域的长宽、中心位置、角度信息

    def set_resize_length(self, length_input):
        self.resize_length = length_input
        print("width=%d,high=%d"%(self.resize_height,self.resize_length))

    def detect_function(self, detect_input, detect_input_last):
        # 缩小图像
        start_tm_1 = time.time()
        frame_resize = cv2.resize(detect_input, (self.resize_height, self.resize_length), interpolation=cv2.INTER_CUBIC)
        # 将彩色图转换为灰度图
        frame_gray = cv2.cvtColor(frame_resize, cv2.COLOR_RGB2GRAY)
        # 将图像与上一帧的图像作差，并作二值化处理
        ret, diff = cv2.threshold(cv2.absdiff(detect_input_last, frame_gray), self.__threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow("img",frame_resize)
        #cv2.waitKey(0)
        #cv2.imshow("diff",diff)
        #cv2.waitKey(0)
        # 直方图图像增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(diff)
        #cv2.imshow("createCLAHE",cl1)
        #cv2.waitKey(0)
        diff_clahe = cv2.GaussianBlur(cl1, (7, 7), 0)
        #cv2.imshow("GaussianBlur",diff_clahe)
        #cv2.waitKey(0)
        # 对图像进行sobel边缘检测
        x_func = cv2.Sobel(diff_clahe, cv2.CV_16S, 1, 0)
        y_func = cv2.Sobel(diff_clahe, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x_func)
        absY = cv2.convertScaleAbs(y_func)
        diff_sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        #cv2.imshow("diff_sobel",diff_sobel)
        #cv2.waitKey(0)
        # 对边缘检测后的图像作二值化处理
        ret_threshold, diff_threshold = cv2.threshold(diff_sobel, self.__threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow("diff_threshold",diff_threshold)
        #cv2.waitKey(0)
        # 对图像进行形态学闭运算，并进行中值滤波
        diff_median = cv2.medianBlur(cv2.morphologyEx(diff_threshold, cv2.MORPH_OPEN, self.__kernel), 5)
        #cv2.imshow("diff_median",diff_median)
        #cv2.waitKey(0)
        # 寻找封闭区域并填充
        #opencv3
        #img, contours, hierarchy = cv2.findContours(diff_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #opencv4
        contours, hierarchy = cv2.findContours(diff_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 计算矩形框的面积
        start_tm_9 = time.time()
        detect_list = []
        count_list = 0
        # 遍历区域列表，筛选面积较大区域，判定为人像区域，提取对应区域外接矩形框的信息
        for count_cnt in contours:
            if cv2.contourArea(count_cnt) > 5000:
                x, y, w, h = cv2.boundingRect(count_cnt)
                h_w_ratio = h/w
                if 1.5<=h_w_ratio<=3:
                    detect_list.append(count_list)
            count_list += 1
        cnt = []
        for i in range(len(detect_list)):
            cnt.append(contours[detect_list[i]])
        rect = []
        for i in range(len(detect_list)):
            rect.append(cv2.minAreaRect(cnt[i]))
        box = []
        for i in range(len(detect_list)):
            box.append(np.int0(cv2.boxPoints(rect[i])))

        detect_flag = []  # 记录当前box下是否检测到安全衣
        for detect_i in range(len(detect_list)):
            crop = [1000, -1, 1000, -1]
            for count_box in range(0, len(box[detect_i])):
                if crop[0] > box[detect_i][count_box][0]:
                    crop[0] = box[detect_i][count_box][0]
                if crop[1] < box[detect_i][count_box][0]:
                    crop[1] = box[detect_i][count_box][0]
                if crop[2] > box[detect_i][count_box][1]:
                    crop[2] = box[detect_i][count_box][1]
                if crop[3] < box[detect_i][count_box][1]:
                    crop[3] = box[detect_i][count_box][1]
            for count_crop in range(len(crop)):
                if crop[count_crop] < 0:
                    crop[count_crop] = 0
            start_tm_10 = time.time()
            print("###time_cost_9:", start_tm_10-start_tm_9)
            crop_hsv = cv2.cvtColor(frame_resize[crop[2]:crop[3], crop[0]:crop[1]], cv2.COLOR_BGR2HSV)
            start_tm_11 = time.time()
            #cv2.imshow("crop_hsv",crop_hsv)
            #cv2.waitKey(0)
            count_orange = 0
            count_total = 0

            # 对当前矩形框中的像素作值判断，筛选与安全衣颜色相近的像素
            for i in range(0, crop_hsv.shape[0]):
                for j in range(0, crop_hsv.shape[1]):
                    count_total += 1
                    if 0 < crop_hsv[i][j][0] < 15 and 150 < crop_hsv[i][j][1] < 255 and 150 < crop_hsv[i][j][2] < 255:
                        crop_hsv[i][j][2] = 255
                        count_orange += 1
                    else:
                        crop_hsv[i][j][2] = 0
                        continue
            #cv2.imshow("crop_hsv_2",crop_hsv)
            #cv2.waitKey(0)
            # 判断区域内与安全衣颜色相近的像素个数是否大于某一定值，并作出该区域是否有安全衣的判断
            if count_orange > 250:
                cv2.drawContours(frame_resize, [box[detect_i]], 0, [0, 255, 0], 2)
                print("count_orange=", count_orange)
                detect_flag.append(True)
            else:
                cv2.drawContours(frame_resize, [box[detect_i]], 0, [0, 0, 255], 2)
                detect_flag.append(False)

            start_tm_12 = time.time()
            print("###time_cost_11:", start_tm_12-start_tm_11)
        cv2.imshow("image", frame_resize)
        cv2.waitKey(1)

        return box, detect_flag


# video_writer主要功能：将判断是否有安全衣后的视频帧存储为图片格式，便于后续处理
class video_writer:
    def __init__(self, width, height):
        # self.__fps = 30
        # self.__video_write = cv2.VideoWriter("video/result.avi",
        #                                     cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size, True)
        self.__size = [width, height]
        self.__count = 0

    def write_frame(self, frame_input, box_input, flag_input):
        frame_resize = cv2.resize(frame_input, (self.__size[1], self.__size[0]), interpolation=cv2.INTER_CUBIC)
        for detect_i in range(len(flag_input)):
            if flag_input[detect_i]:
                cv2.drawContours(frame_resize, [box_input[detect_i]], 0, [0, 255, 0], 2)
            else:
                cv2.drawContours(frame_resize, [box_input[detect_i]], 0, [0, 0, 255], 2)
        file_name = "image/" + str(self.__count) + ".jpg"
        self.__count += 1
        cv2.imwrite(file_name, frame_resize)

def deal_specialchar_in_url(istr):
    s = istr.find('//')
    e = istr.rfind('@')
    # print(s, e)
    subs = istr[s+2: e]
    name, ps = subs.split(':')
    encoded_name_ps = urllib.parse.quote_plus(name)+ ":" + urllib.parse.quote_plus(ps)
    result = 'rtsp://' + encoded_name_ps + istr[e:]
    print("#####result=",result)
    return result

if __name__ == '__main__':
    rtsp_url = "rtsp://admin:yxgl$666@192.168.200.211:554:554/Streaming/Channels/1"
    # 初始化姿态检测主函数
    detect = color_detection()
    # 获取视频
    #videoCapture = cv2.VideoCapture('/media/liujin/disk/project/data/safty_belt/1.mp4')
    url = deal_specialchar_in_url(rtsp_url)
    print('url', url)
    videoCapture = cv2.VideoCapture(url)
    # 获得码率及尺寸
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps)
    start_time = datetime.datetime.now()  # 记录开始时间

    # 读帧
    success, frame = videoCapture.read()
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
    writer = video_writer(detect.resize_length, detect.resize_height)  # 初始化video_writer

    count = 0
    detect_box = []
    flag = []
    while True:
        try:
            count += 1
            if count > 10:  # 跳过前几帧
                # 安全衣识别核心函数部分
                start_time = time.time()
                detect_box, flag = detect.detect_function(frame_detect, frame_gray_last)
                end_time = time.time() 
                print("frame is:", 1.0/(end_time-start_time) )
            # 记录本帧的安全衣识别情况，保存为图片格式
            #writer.write_frame(frame, detect_box, flag)
            success, frame = videoCapture.read()  # 获取下一帧
            if frame is None:
                break
            # 截取下一帧的检测区域
            frame_gray_last = cv2.cvtColor(cv2.resize(frame_detect, (detect.resize_height, detect.resize_length),
                                                      interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2GRAY)
            frame_detect = frame[int(detect.detect_area[2] * x_origin): int(detect.detect_area[3] * x_origin), int(
                detect.detect_area[0] * y_origin): int(detect.detect_area[1] * y_origin)]
        except Exception as e:
            print(count)
            traceback.print_exc()
            cv2.destroyAllWindows()
            videoCapture.release()
            exit(0)
    end_time = datetime.datetime.now()
    print((end_time - start_time))
    cv2.destroyAllWindows()
    videoCapture.release()


