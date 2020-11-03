#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from log import Logger

if sys.version > '3':
    PY3 = True
    from scenes.config_file import config

    importlib.reload(sys)
else:
    PY3 = False
    sys.path.append("./scenes")
    from config_file import config

logger = Logger(logname='./log/module_jacket.log', loglevel=1, logger="module_jacket").getlog()

class SceneManager(object):
    def __init__(self, bw_image_dir):
        """
        :param bw_image_dir: black/white images indicating targeted areas by white polygon
        """
        assert os.path.exists(bw_image_dir)
        assert os.path.isdir(bw_image_dir)
        self.bw_image_dir = bw_image_dir
        self.Scenes = {}

    def create_scene(self, sceneId):
        """
        :param sceneId: integer ID of the camera scene
        :return: the created new scene
        """
        new_scene = Scene(sceneId=sceneId, bw_image_dir=self.bw_image_dir)
        self.Scenes[sceneId] = new_scene
        return new_scene

    def get_scene(self, sceneId):
        """
        :param sceneId: integer ID of the camera scene
        :return: the existing scene
        """
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
    def __init__(self, sceneId, bw_image_dir):
        """
        :param sceneId: ID of this scene, IP Address concatenating integers from 1~32
        :param bw_image_dir: directory to store the black white image of safe & warn zones
        """
        self.warn_polygons = None
        self.sceneId = sceneId
        self.frame_num = 0
        self.history = 30
        self.resized_height = 600 #调整大小后的图像高
        self.resized_width = -1
        self.size = (0, 0)
        self.resized = (0, 0)
        warn_white_path = os.path.join(bw_image_dir, "%s.jpg" % sceneId)
        print("warn_white_path=", warn_white_path)
        if not os.path.exists(warn_white_path):
            print("[Warn] %s Dose not exist, use default.jpg instead."%(warn_white_path))
            warn_white_path = os.path.join(bw_image_dir, "default.jpg")
        self.warn_polygons = self.get_polygon_by_black_white_img(warn_white_path)
        # KNN背景分割器
        self.bs = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        self.bs.setHistory(self.history)

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
        img, contours, hierarchy = cv2.findContours(thresh, mode=1, method=1)
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
        for sceneId in config["scene_jacket"]:
            self.sm.create_scene(sceneId=sceneId)

    def write_frame(self, frame_input):
        save_path = 'image/'
        is_exist = os.path.exists(save_path)
        if not is_exist:
            os.umask(0)
            os.makedirs(save_path)
        daytime = datetime.datetime.now().strftime('%Y-%m-%d')
        hourtime = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        file_name = os.path.join(save_path,"%s-%s.jpg" % (daytime,hourtime))
        cv2.imwrite(file_name, frame_input)

    def prediction(self, frame, sceneId):
        #颜色分布区域
        red_lower = np.array([156,70,70])
        red_upper = np.array([180,255,255])
        red_lower_2 = np.array([0,140,140])
        red_upper_2 = np.array([15,255,255])
        blue_lower = np.array([100,120,30])
        blue_upper = np.array([124,255,255])
        #获取场景参数
        scene = self.sm.get_scene(sceneId)
        #获得码率及尺寸
        size = (frame.shape[1], frame.shape[0])
        logger.info("sceneId=%s size=(%d,%d)",sceneId, size[0], size[1])
        if scene.size != size:
            logger.error("[%s]:size=(%d,%d) scene.size=(%d,%d)",sceneId, size[0], size[1], scene.size[0],scene.size[1])
            return False

        now = datetime.datetime.now()
        hour = now.hour
        start = 7
        end = 19

        if int(hour) not in list(range(start, end)):
            logger.info("Hour:%d not in the range(%d %d)", hour,start,end)
            time.sleep(60)
            return False

        frame_resize = cv2.resize(frame, (scene.resized_width, scene.resized_height))
        # 获取前景mask
        fg_mask = scene.bs.apply(frame_resize)
        if scene.frame_num < scene.history:
            scene.frame_num += 1
            return False

        #形态学处理后，找外接矩形框
        th = cv2.threshold(fg_mask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        dilated = cv2.medianBlur(dilated, 5)
        #opencv3
        img, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #opencv4
        #contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_warning = False

        for c in contours:
            area = cv2.contourArea(c)
            if area > 10000:
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
                    max_b_h = 0
                    for c in cnts:
                        area = cv2.contourArea(c)
                        x1, y1, w1, h1 = cv2.boundingRect(c)
                        if area > 500:
                            if (y1+h1) >= max_r_h:
                                max_r_h = y1+h1
                            num_red = num_red+1
                            cv2.rectangle(res, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

                    for c in cnts_blue:
                        area = cv2.contourArea(c)
                        x1, y1, w1, h1 = cv2.boundingRect(c)
                        w_h_ratio = w1/h1
                        if area > 1250 and (y1+h1) <= max_r_h:
                            num_blue = num_blue+1
                            cv2.rectangle(res_blue, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 0), 2)

                    if num_red>=1 and num_blue==0:
                        cv2.rectangle(frame_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
        else:
            cv2.putText(frame_resize, "NORMAL", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("image", frame_resize)
        if cv2.waitKey(1) & 0xFF == (ord('q') or ord('Q')):
            camera.release()
            raise Exception("exit")
        return is_warning

class LifeJacketWraper(object):
    def __init__(self, gpu_id=0):
        """
        Wrapper of LifeJacketWraper, declaring queues and register recall
        :param gpu_id:
        """
        self.alertType = 2
        self.eventsByScene = {}
        for sceneId in config["scene_jacket"]:
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
        self.qn_in = config['frame_q_in']
        self.qn_out = config['frame_q_out']

        self.ex_in = config['frame_q_in']
        self.ex_out = config['frame_q_out']

        self.rtk_in = config['frame_q_in']
        self.rtk_out = config['frame_q_out']

        # Exchange declaration
        self.ch.exchange_declare(exchange=self.ex_in, exchange_type='direct', durable=True)
        self.ch.exchange_declare(exchange=self.ex_out, exchange_type='direct', durable=True)

        # Queue declaration
        self.q_in = self.ch.queue_declare(queue=self.qn_in, durable=True)
        self.q_out = self.ch.queue_declare(queue=self.qn_out, durable=True)

        # Bind queues to corresponding exchanges
        self.ch.queue_bind(exchange=self.ex_in, queue=self.qn_in, routing_key=self.rtk_in)
        self.ch.queue_bind(exchange=self.ex_out, queue=self.qn_out, routing_key=self.rtk_out)

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
        img = base64.b64decode(obj_json['picData'].encode())
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

    def running(self):
        def callback(ch, method, properties, body):
            start_time = time.time()
            obj_json = self.getJsonObj(body=body)
            sceneId = str(obj_json["cameraId"]) + "_" + obj_json["recorderId"]
            picId = obj_json['picId']
            # Getting json string from mq and get the image array
            timeID = self.getTimeStr(obj_json)
            print("%s %s %s %s"%(timeID, obj_json['recorderId'], obj_json['cameraId'], "#" * 8))

            img_opencv, h, w, c = self.getOpencvImg(obj_json)
            # Prediction
            is_alarm = self.detector.prediction(img_opencv, sceneId)
            if is_alarm:
                self.eventsByScene[sceneId]['eventId'] = picId + "|" + str(self.alertType)
                self.eventsByScene[sceneId]['eventPics'].append({'picId': picId, 'alertObjects': {}})
                response_dict = {
                     'protocol': '1.0.0',
                     'alertType': self.alertType,
                     'latestPicId': obj_json['picId'],
                     'eventId': self.eventsByScene[sceneId]['eventId'],
                     'eventPics': self.eventsByScene[sceneId]['eventPics'],
                     'Time01StampID': timeID,
                }
                # dumps json obj
                response_dict = json.dumps(response_dict, sort_keys=True, indent=2)
                print("response_dict=", response_dict)
                self.ch.basic_publish(exchange=self.ex_out, routing_key=self.qn_out, body=response_dict) 
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


if __name__ == '__main__':
    lifeJacketWrapper = LifeJacketWraper()
    lifeJacketWrapper.running()



