import cv2

import time
import multiprocessing as mp
import os

import logging
import logging.handlers
from random import choice, random

import csv
import ast
import urllib

import numpy
# from i13sdk import HKI_base64
from i13rabbitmq import sender
import i13rabbitmq_config
#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks


rtsp_file_path = 'i13rtsp_list.csv'
queue_rtsp_dict = {}
img_name = ''




def deal_specialchar_in_url(istr):
    # encode处理 rtsp地址中特殊字符，比如加号 ，否则连不上
    s = istr.find('//')
    e = istr.rfind('@')
    # print(s, e)
    subs = istr[s+2: e]
    name, ps = subs.split(':')
    encoded_name_ps = urllib.parse.quote_plus(name)+ ":" + urllib.parse.quote_plus(ps)
    result = 'rtsp://' + encoded_name_ps + istr[e:]
    # print(result)
    return result


def image_put(q, queueid):

    name = queue_rtsp_dict.get(queueid, None)[1]
    pwd = queue_rtsp_dict.get(queueid, None)[2]
    ip = queue_rtsp_dict.get(queueid, None)[3]
    channel = queue_rtsp_dict.get(queueid, None)[4]
    camera_corp = queue_rtsp_dict.get(queueid, None)[5]
    default_enter_rule = queue_rtsp_dict.get(queueid, None)[9]
    print(default_enter_rule, '#'*20)
    # 大华的情况 ：
    if camera_corp == 'dahua':
        full_vedio_url = "rtsp://%s:%s@%s/cam/realmonitor?channel=%s&subtype=0" % (name, pwd, ip, channel)
    # 网络摄像头是海康:
    elif camera_corp == 'hik':
        full_vedio_url = "rtsp://%s:%s@%s/Streaming/Channels/%s" % (name, pwd, ip, channel)

    if camera_corp in ('dahua', 'hik'):

        try:
            ull_vedio_url = deal_specialchar_in_url(full_vedio_url)
            cap = cv2.VideoCapture(full_vedio_url)
            # 通过timeF控制多少帧数真正读取1帧到队列中
            timeF = 15
            count = 1 
            if 'LifeJacket' in default_enter_rule:
                timeF = 1

            while True:
                if cap.isOpened():
                    status, frame = cap.read()
                    if status:
                        if count % timeF == 0:
                            # print('pick=', count)
                            print(type(frame), numpy.size(frame), 'queueid=', queueid)
                            # time.sleep(0.5)
                            # print('sleep 0.5')

                            
                            if 'LifeJacket' in default_enter_rule:
                                sender(i13rabbitmq_config.Where_This_Server_ReadFrom, frame, queueid, 'LifeJacket')
                            if 'SafetyBelt' in default_enter_rule:
                                sender(i13rabbitmq_config.Where_This_Server_ReadFrom, frame, queueid, 'SafetyBelt')
                            
                            sender(i13rabbitmq_config.Where_This_Server_ReadFrom, frame, queueid, 'hello')
                        count += 1
        except cv2.error as e:
            print('#'*10, 'cv2 error:', e)

        # full_vedio_url = deal_specialchar_in_url(full_vedio_url)
        # cap = cv2.VideoCapture(full_vedio_url)
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
        

        # count = 0

        # while(True):
        #     # Capture frame-by-frame
        #     ret, frame = cap.read()
        #     # print(frame, type(frame),'#', ret)
        #     if frame is not None:
        #         from i13rabbitmq import sender
        #         print(type(frame), numpy.size(frame), 'queueid=', queueid)
        #         sender('localhost', frame, queueid)
        #         # # Our operations on the frame come here
        #         # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #         # Display the resulting frame
        #         # cv2.imshow('frame',gray)
                
        #         # cv2.namedWindow('not-sdk', flags=cv2.WINDOW_NORMAL)
        #         # cv2.imshow('not-sdk',frame)

        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
    # elif camera_corp in ('hik_sdk'):
    #     HKI_base64(ip[:-4], name, pwd, queueid)



def run_multi_camera(camera_ip_l):
    print(camera_ip_l)
    global queue_rtsp_dict

    # mp.set_start_method(method='spawn')  # init0
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]
    processes = []



    
    for queue, camera_ip in zip(queues, camera_ip_l):
        # rect = ast.literal_eval(camera_ip[7])
        # print(camera_ip, camera_ip[0], '##', rect, type(rect))

        # reader from camera ,according to placeid range setting in i13rabbitmq_confg
        if int(camera_ip[0]) >= i13rabbitmq_config.AI_SERVER_NUMBER_placeid_start and \
           int(camera_ip[0]) <= i13rabbitmq_config.AI_SERVER_NUMBER_placeid_end:
            print(camera_ip[0], ' in append')
            processes.append(mp.Process(target=image_put, 
                args=(queue, camera_ip[0])))

            queue_rtsp_dict[camera_ip[0]] = camera_ip
        else:
            print(camera_ip[0], '_'*20, ' not allow in this server')
        # processes.append(mp.Process(target=image_get, args=(queue, camera_ip[2])))

    [process.start() for process in processes]
    [process.join() for process in processes]


def load_rtsp_list():
    with open(rtsp_file_path, newline='') as f:
        reader = csv.reader(f)
        rtsp_list = list(reader)
    # print(data, '#'*10, data[0],'\n', data[0][0])
    return rtsp_list

if __name__ == '__main__':
    camera_ip_l = load_rtsp_list()
    # python3 i11temp_yangxin.py --gpu=True --network=yolo3_mobilenet0.25_voc
    run_multi_camera(camera_ip_l) 
