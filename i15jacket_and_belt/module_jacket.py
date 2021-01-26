# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 2020
@author: liujin
@function: recognize lifejacket
@version: V1.0
@modify: 2020/10/30
"""
import cv2
import datetime
import numpy as np
import traceback
import time
import urllib
import os,sys
import importlib
import pika
import base64
import json
import logging
import logging.handlers
import multiprocessing as mp
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from  common_message_decison_maker import *


if sys.version > '3':
    PY3 = True
    from scenes.config_file import config
    importlib.reload(sys)
else:
    PY3 = False
    sys.path.append("./scenes")
    from config_file import config


class SceneManager(object):
    def __init__(self, bw_image_dir):
        """
        :param bw_image_dir: black/white images indicating targeted areas by white polygon
        """
        assert os.path.exists(bw_image_dir)
        assert os.path.isdir(bw_image_dir)
        self.bw_image_dir = bw_image_dir
        self.Scenes = {}

    def create_scene(self, sceneId, values):
        """
        :param sceneId: integer ID of the camera scene
        :return: the created new scene
        """
        new_scene = Scene(sceneId=sceneId, bw_image_dir=self.bw_image_dir, values=values)
        self.Scenes[sceneId] = new_scene
        return new_scene

    def get_scene(self, sceneId):
        """
        :param sceneId: integer ID of the camera scene
        :return: the existing scene
        """
        print(sceneId, ' get_scene')
        if sceneId in self.Scenes.keys():
            return self.Scenes[sceneId]
        else:
            return self.Scenes['default']

    def check_this_frame_has_zone_crosser(self, scene, feet_position, buffer):
        """
        As the func name indicates
        :param scene:
        :param feet_position:
        :return:
        """
        pos = feet_position.astype(int)
        if scene.point_warn_zone_test((pos[0], pos[1]), buffer=buffer):  # warn zone
            return True
        return False

class Scene(object):
    def __init__(self, sceneId, bw_image_dir, values):
        """
        :param sceneId: ID of this scene, IP Address concatenating integers from 1~32
        :param bw_image_dir: directory to store the black white image of safe & warn zones
        """
        self.warn_polygons = None
        self.sceneId = sceneId
        self.resized_height = 600
        self.resized_width = -1
        self.size = (0, 0)
        self.resized = (0, 0)
        self.red_lower = values[0]
        self.red_upper = values[1]
        self.blue_lower = values[2]
        self.blue_upper = values[3]
        warn_white_path = os.path.join(bw_image_dir, "%s.jpg" % sceneId)
        print("warn_white_path=", warn_white_path)
        if not os.path.exists(warn_white_path):
            print("[Warn] %s Dose not exist, use default.jpg instead."%(warn_white_path))
            warn_white_path = os.path.join(bw_image_dir, "default.jpg")
        self.warn_polygons = self.get_polygon_by_black_white_img(warn_white_path)
        # KNN背景分割器
        self.bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    def get_polygon_by_black_white_img(self, bw_image_path):
        """
        :param bw_image_path: path to the black/white image indicating zones by white area
        :return: a list of polygons containing white region of the bw_image
        """
        bw_image = cv2.imread(bw_image_path)
        self.size = (bw_image.shape[1], bw_image.shape[0])
        self.resized_width = int(self.size[0] / (self.size[1] / self.resized_height))
        self.resized = (self.resized_width, self.resized_height)
        bw_image = cv2.resize(bw_image, self.resized)
        bw_image = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(bw_image, thresh=128, maxval=256, type=0)
        contours, hierarchy = cv2.findContours(thresh, mode=1, method=1)
        zone_polygons = [cv2.approxPolyDP(curve=contour, epsilon=5, closed=True) for contour in contours]
        return zone_polygons

    def point_warn_zone_test(self, point, buffer=0):
        """
        :param point: Point in (x, y) format , test if a point is in safe zone of this scene
        :return: True if point in the polygon
        """
        for zone in self.warn_polygons:
            if cv2.pointPolygonTest(contour=zone, pt=point, measureDist=True) + buffer > 0:
                return True
        return False


class LifeJacketDetector:
    def __init__(self):
        #场景初始化
        self.sm = SceneManager(bw_image_dir="./scenes")
        for scenes in config['scene_jacket']:
            sceneId = list(scenes.keys())[0]
            values = scenes[sceneId]
            self.sm.create_scene(sceneId, values)

    def write_frame(self, frame_input):
        daytime = datetime.datetime.now().strftime('%Y-%m-%d')
        save_path = 'image_jacket/'+daytime
        is_exist = os.path.exists(save_path)
        if not is_exist:
            os.umask(0)
            os.makedirs(save_path)
        hourtime = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        file_name = os.path.join(save_path,"%s-%s.jpg" % (daytime,hourtime))
        cv2.imwrite(file_name, frame_input)

    def prediction(self, frame, sceneId):
        #获取场景参数
        scene = self.sm.get_scene(sceneId)
        red_lower = np.array([156,70,70])
        red_upper = np.array([180,255,255])
        red_lower_2 = scene.red_lower
        red_upper_2 = scene.red_upper
        blue_lower = scene.blue_lower
        blue_upper = scene.blue_upper
        img_org = frame.copy()
        #获得码率及尺寸
        size = (frame.shape[1], frame.shape[0])
        # cv2.imwrite('./screenshots/test.jpg' , frame)
        frame_resize = cv2.resize(frame, (scene.resized_width, scene.resized_height))
        if scene.size != size:
            print("ERROR:[%s]:frame_size=(%d,%d) black_white_size=(%d,%d)"%(sceneId, size[0], size[1], scene.size[0],scene.size[1]))
            return False, frame_resize

        # 获取前景mask
        fg_mask = scene.bs.apply(frame_resize)
        # 形态学处理
        kernel1 = np.ones((5,5),np.uint8)
        kernel2 = np.ones((9,9),np.uint8)
        th = cv2.threshold(fg_mask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        dilated = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel1)
        dilated =  cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel2)
        dilated = cv2.medianBlur(dilated, 5)
        #opencv3
        # img, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #opencv4
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_warning = False

        for c in contours:
            area = cv2.contourArea(c)
            if 10000 <= area <= 40000:
                x, y, w, h = cv2.boundingRect(c)
                h_w_ratio = h/w
                point = (int(x+w/2), int(y+h))
                b_in_zone = scene.point_warn_zone_test(point)
                if b_in_zone == False:
                    continue
                if 1<=h_w_ratio<=3:
                    crop_hsv = cv2.cvtColor(frame_resize[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
                    frame_p = cv2.GaussianBlur(frame_resize[y:y+h, x:x+w],(5,5),0)
                    mask = cv2.inRange(crop_hsv, red_lower, red_upper)
                    mask_2 = cv2.inRange(crop_hsv, red_lower_2, red_upper_2)
                    mask = cv2.bitwise_or(mask,mask_2)
                    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
                    mask = cv2.medianBlur(mask, 5)

                    mask_b = cv2.inRange(crop_hsv, blue_lower, blue_upper)
                    mask_b = cv2.erode(mask_b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
                    mask_b = cv2.dilate(mask_b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),iterations=2)
                    mask_b = cv2.medianBlur(mask_b, 5)

                    res = cv2.bitwise_and(frame_p,frame_p,mask=mask)
                    res_blue = cv2.bitwise_and(frame_p, frame_p, mask=mask_b)
                    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                    cnts_blue = cv2.findContours(mask_b.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                    num_red = 0
                    num_blue = 0
                    max_r_h = 0
                    count_red = 0
                    count_blue = 0
                    for c in cnts:
                        area = cv2.contourArea(c)
                        x1, y1, w1, h1 = cv2.boundingRect(c)
                        h_w_ratio = h1/w1
                        head_ratio = y1/h
                        if area > 500:
                            count_red = count_red+1
                        if (area > 500 and head_ratio > 0.1 and h_w_ratio > 0.3 and h_w_ratio < 3) \
                               or (head_ratio <= 0.1 and area > 2000):
                            if (y1+h1) >= max_r_h:
                                max_r_h = y1+h1
                            num_red = num_red+1
                            cv2.rectangle(res, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

                    for c in cnts_blue:
                        area = cv2.contourArea(c)
                        x1, y1, w1, h1 = cv2.boundingRect(c)
                        h_w_ratio = h1/w1
                        if area > 1250:
                            count_blue = count_blue+1
                        if area > 1250 and (y1+h1) <= max_r_h and h_w_ratio < 1:
                            num_blue = num_blue+1
                            cv2.rectangle(res_blue, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 0), 2)

                    if num_red>=1 and num_blue==0:
                        cv2.rectangle(frame_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        if count_red == 0 and count_blue == 0:
                            is_warning = False
                        else:
                            is_warning = True
                            cv2.rectangle(frame_resize, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #cv2.imshow("crop_hsv",crop_hsv)
                    #cv2.imshow("mask",mask)
                    #cv2.imshow("res",res)
                    #cv2.imshow("res_blue",res_blue)
  
        cv2.polylines(frame_resize, scene.warn_polygons, True, (0, 255, 255), 2)
        if is_warning:
            cv2.putText(frame_resize, "WARNING", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 0, 255), 2)
            self.write_frame(img_org)
        else:
            cv2.putText(frame_resize, "NORMAL", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        imshow_name = "image"+str(sceneId)
        cv2.imshow(imshow_name, frame_resize)
        if cv2.waitKey(1) & 0xFF == (ord('q') or ord('Q')):
            camera.release()
            raise Exception("exit")
        return is_warning, frame_resize





class LifeJacketWraper(object):
    def __init__(self, gpu_id=0):
        """
        Wrapper of LifeJacketWraper, declaring queues and register recall
        :param gpu_id:
        """
        self.alertType = 2
        self.eventsByScene = {}

        for scenes in config['scene_jacket']:
            sceneId = list(scenes.keys())[0]
            self.eventsByScene[sceneId] = {'eventId': None, 'eventPics': []}
            
        # Connections
        username = config['mq_username']
        pwd = config['mq_pswd']
        ip = config['mq_server_host']
        port = config['mq_server_port']
        user_pwd = pika.PlainCredentials(username, pwd)
        self.con = pika.BlockingConnection(pika.ConnectionParameters(host=ip, port=port, credentials=user_pwd))
        self.ch = self.con.channel()

        # Queue name declaration
        self.qn_in = config['lifeJacket_q_in']
        # self.qn_out = config['frame_q_out']

        # self.ex_in = config['frame_q_in']
        # self.ex_out = config['frame_q_out']

        # self.rtk_in = config['frame_q_in']
        # self.rtk_out = config['frame_q_out']

        # Exchange declaration
        # self.ch.exchange_declare(exchange=self.ex_in, exchange_type='direct', durable=True)
        # self.ch.exchange_declare(exchange=self.ex_out, exchange_type='direct', durable=True)

        # # Queue declaration
        # self.q_in = self.ch.queue_declare(queue=self.qn_in, durable=True)
        # self.q_out = self.ch.queue_declare(queue=self.qn_out, durable=True)

        # # Bind queues to corresponding exchanges
        # self.ch.queue_bind(exchange=self.ex_in, queue=self.qn_in, routing_key=self.rtk_in)
        # self.ch.queue_bind(exchange=self.ex_out, queue=self.qn_out, routing_key=self.rtk_out)

        # Init Detector
        self.detector = LifeJacketDetector()


    def getJsonObj(self, body):
        # get json string from binary body
        data_string = bytes.decode(body)
        # load to json obj
        obj_json = json.loads(data_string)
        return obj_json

    def getOpencvImg(self, obj_json):
        # get image bytes string
        img = base64.b64decode(obj_json['img'].encode())
        # get image array
        img_opencv = cv2.imdecode(np.fromstring(img, np.uint8), 1)
        h, w, c = img_opencv.shape
        return img_opencv, h, w, c

    def getTimeStr(self, obj_json):
        picId = obj_json['picId']
        time_stamp = int(picId.split('|')[1]) // 1000
        loc_time = time.localtime(time_stamp)
        time_str = time.strftime("%Y-%m-%d_%H:%M:%S", loc_time)
        return time_str

    def serialize(self, detectionResults):

        detectionResults = pickle.dumps(detectionResults)
        detectionResults = base64.b64encode(detectionResults)
        detectionResults = bytes.decode(detectionResults, encoding='utf-8')
        return detectionResults

    def running(self, log_queue):
        h = logging.handlers.QueueHandler(log_queue)  # Just the one handler needed
        root = logging.getLogger()
        root.addHandler(h)
        # send all messages, for demo; no other level or filter logic applied.
        root.setLevel(logging.DEBUG)

        def callback(ch, method, properties, body):
            print('start callback')
                #     'placeid': queueid,
                # 'time': now,
                # 'img': picData_string
            start_time = time.time()
            obj_json = self.getJsonObj(body=body)
            sceneId = str(obj_json["placeid"])
            picId =sceneId
            # Getting json string from mq and get the image array
            timeID =  str(obj_json["time"])
            print("%s %s"%(timeID, obj_json['placeid']))

            img_opencv, h, w, c = self.getOpencvImg(obj_json)
            # Prediction
            is_alarm, myframe = self.detector.prediction(img_opencv, sceneId)
            print('is_alarm=', is_alarm, type(myframe))
            if  is_alarm:
                self.eventsByScene[sceneId]['eventId'] = picId + "|" + str(self.alertType)
                self.eventsByScene[sceneId]['eventPics'].append({'picId': picId, 'alertObjects': {}})
                response_dict = {
                     'protocol': '1.0.0',
                     'alertType': self.alertType,
                     'eventId': self.eventsByScene[sceneId]['eventId'],
                     'eventPics': self.eventsByScene[sceneId]['eventPics'],
                     'Time01StampID': timeID,
                }
                # dumps json obj
                response_dict = json.dumps(response_dict, sort_keys=True, indent=2)
                print("response_dict=", response_dict)
                # alert need to implement

                # add temp send 
                print('----------------', ' save image')


                queueid = int(sceneId)
                logger = logging.getLogger(str(queueid))
                now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
                img_name= str(sceneId) + '_'+ now + '.jpg'
                cv2.imwrite('./screenshots/' + img_name, myframe)

                warning_signal = 'no-jacket-in-area'
                print('----------------', logging.CRITICAL, warning_signal + '#' + img_name + '#' +queue_rtsp_dict.get(queueid, None)[5]
                            + '#' + queue_rtsp_dict.get(queueid, None)[6])
                logger.log(logging.CRITICAL, warning_signal + '#' + img_name + '#' +queue_rtsp_dict.get(queueid, None)[5]
                            + '#' + queue_rtsp_dict.get(queueid, None)[6])
                print('@after logger.log()')


                # self.ch.basic_publish(exchange=self.ex_out, routing_key=self.qn_out, body=response_dict) 
            else:
                self.eventsByScene[sceneId] = {'eventId': None, 'eventPics': []} 

            ch.basic_ack(delivery_tag=method.delivery_tag)
            cost_time = time.time()-start_time
            print('callback:%f ms'%(cost_time*1000))

        # Register the consume function
        self.ch.basic_consume(queue=self.qn_in,on_message_callback=callback,auto_ack=False,exclusive=False,
                      consumer_tag=None,
                      arguments=None)
        print('[*] Waiting for logs. To exit press CTRL+C')
        # Starting consuming
        self.ch.start_consuming()

queue_rtsp_dict = {}

if __name__ == '__main__':
    camera_ip_l = load_rtsp_list()
    for line in camera_ip_l:
        print('line:', line, type(line))
        queue_rtsp_dict[int(line[0])] = line[1:]
    log_queue = mp.Queue(-1)
    listener = mp.Process(target=listener_process,
                                       args=(log_queue, listener_configurer_jacket))
    listener.start()

    lifeJacketWrapper = LifeJacketWraper()
    lifeJacketWrapper.running(log_queue)



