#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import datetime
import numpy as np

if __name__ == '__main__':
    # 获取视频
    videoCapture = cv2.VideoCapture('video/rain.mp4')

    # 获得码率及尺寸
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    start_time = datetime.datetime.now()  # 记录开始时间

    # 需要的参数
    detect_area = [0.1, 0.9, 0.1, 0.9]  # 截取的图像区域，范围为0~1，长宽
    interval = 0.1  # 采样间隔（单位：秒）
    resize_height = 600  # 调整大小后的图像高
    threshold = 5  # 图像二值化的阀值
    area = 0.2  # 前后取样点最大容忍差异面积
    height = 0.4  # 二次对比时截取图像的高度
    area_warn = 0.2  # 二次对比时最大容忍差异面积
    # 图像膨胀相关参数
    kernel = np.ones((5, 5), np.uint8)
    is_dilate = True

    # 读帧
    success, frame = videoCapture.read()
    # 对第一张图像进行处理
    # 获取图像长宽
    x_origin, y_origin = frame.shape[0:2]
    print('x_origin, y_origin=', x_origin, y_origin)
    # 选择需要作检测的区域
    frame_detect = frame[int(detect_area[2] * x_origin): int(detect_area[3] * x_origin), int(
        detect_area[0] * y_origin):int(detect_area[1] * y_origin)]
    x, y = frame_detect.shape[0:2]
    print('x, y =', x, y)
    # 计算缩小后图像的面积和宽度
    resize_length = int(x / (y / resize_height))
    resize_area = resize_height * resize_length
    # 将彩色图转换为灰度图并存储到frame_gray_last中
    frame_gray_last = cv2.cvtColor(cv2.resize(frame_detect, (resize_height, resize_length),
                                              interpolation=cv2.INTER_CUBIC),
                                   cv2.COLOR_RGB2GRAY)
    # 出现问题帧存储于此
    warn_frame = []
    # 警戒等级：每次检测到出现问题的帧时为2，接下来检测时如果没有问题则依次减一，当此信号为0时清空warn_frame
    warn_rank = 0

    count = 0
    while count < 3000:
        count += 1
        if count % int(fps * interval) == 0 and count > 1 and count > 70:  # 跳过第一次循环
            try:
                # 缩小图像
                frame_resize = cv2.resize(frame_detect, (resize_height, resize_length), interpolation=cv2.INTER_CUBIC)
                # 将彩色图转换为灰度图
                frame_gray = cv2.cvtColor(frame_resize, cv2.COLOR_RGB2GRAY)
                # 将图像与原图作差，并作二值化处理
                ret, diff = cv2.threshold(cv2.absdiff(frame_gray_last, frame_gray), threshold, 255, cv2.THRESH_BINARY)
                ###
                print('count=', count)
                cv2.imshow("diff", diff)
                cv2.imshow("orig", frame_detect)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                ###
                # 统计出现差异的像素个数
                count_diff = 0
                for i in range(resize_length):
                    for j in range(resize_height):
                        if diff[i, j] == 255:
                            count_diff += 1
                if float(count_diff) / float(resize_area) > area:  # 判断两次采样差异是否超过设置的最大容忍差异面积
                    warn_rank = 2  # 设置警戒等级
                    if len(warn_frame) >= 1:  # 判断存储问题帧的列表是否有元素
                        # 进入二次对比过程
                        # 对两幅差异图像进行作差二值化处理
                        ret, diff_warn = cv2.threshold(cv2.absdiff(warn_frame[0], diff),
                                                       threshold, 255, cv2.THRESH_BINARY)
                        # 将图像截取上半部分，由于烟雾是向上走的，所以预期这里会出现较大差异
                        cropped_diff = diff_warn[0: int(height * resize_height), :]
                        # 对出现差异的像素点进行计数
                        count_warn = 0
                        for i in range(int(height * resize_height)):
                            for j in range(resize_length):
                                if diff[i, j] == 255:
                                    count_warn += 1
                        if float(count_diff) / (height * resize_length * resize_height) > area_warn:
                            # 输出报警信号
                            print("detected with area %.2f%%" % (
                                    100 * float(count_diff) / (height * resize_length * resize_height)))
                            end_time = datetime.datetime.now()
                            print((end_time - start_time))
                            videoCapture.release()
                            exit(1)
                    else:
                        # 如果存储问题帧的列表为空，则添加列表
                        warn_frame.append(diff)
                elif warn_rank > 0:
                    # 降低警戒等级
                    warn_rank -= 1
                    if warn_rank == 0:
                        # 警戒度为0，清空warn_frame
                        warn_frame.pop()
                else:
                    # 警戒度为0时，更新前次采样帧
                    frame_gray_last = frame_gray
            except Exception as e:
                print(e)
                break
        success, frame = videoCapture.read()  # 获取下一帧
        # 截取下一帧的检测区域
        frame_detect = frame[int(detect_area[2] * x_origin): int(detect_area[3] * x_origin), int(
            detect_area[0] * y_origin):int(detect_area[1] * y_origin)]
    end_time = datetime.datetime.now()
    print((end_time - start_time))
    videoCapture.release()
