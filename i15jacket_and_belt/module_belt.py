# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 2020
@author: liujin
@function: recognize safetybelt
@version: V1.0
@modify: 2020/10/30
"""
import os
import sys
import torch
from torchvision import transforms
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from bisenet import BiSeNet
import matplotlib.pyplot as plt
from scipy import optimize
import importlib
import argparse
import pika
import base64
import json
import datetime
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

num = 1

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
        print(sceneId, ' in get_scene')
        """
        :param sceneId: integer ID of the camera scene
        :return: the existing scene
        """
        if sceneId in self.Scenes.keys():
            return self.Scenes[sceneId]
        else:
            return self.Scenes['default_belt']


class Scene(object):
    def __init__(self, sceneId, bw_image_dir):
        """
        :param sceneId: ID of this scene, IP Address concatenating integers from 1~32
        :param bw_image_dir: directory to store the black white image of safe & warn zones
        """
        self.mask = None
        self.position = None
        self.sceneId = sceneId
        warn_white_path = os.path.join(bw_image_dir, "%s.jpg" % sceneId)
        print("warn_white_path=", warn_white_path)
        if not os.path.exists(warn_white_path):
            print("[Warn] %s Dose not exist, use default.jpg instead."%(warn_white_path))
            warn_white_path = os.path.join(bw_image_dir, "default_belt.jpg")
        #self.get_mask_by_black_white_img(bw_image_path=warn_white_path)
        self.warn_polygons = self.get_polygon_by_black_white_img(warn_white_path)
        
    def get_polygon_by_black_white_img(self, bw_image_path):
        """
        :param bw_image_path: path to the black/white image indicating zones by white area
        :return: a list of polygons containing white region of the bw_image
        """
        bw_image = cv2.imread(bw_image_path)
        bw_image = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(bw_image, thresh=128, maxval=256, type=0)
        contours, hierarchy = cv2.findContours(thresh, mode=1, method=1)
        zone_polygons = [cv2.approxPolyDP(curve=contour, epsilon=5, closed=True) for contour in contours]
        return zone_polygons

    def point_zone_test(self, point, buffer=0):
        """
        :param point: Point in (x, y) format , test if a point is in safe zone of this scene
        :return: True if point in the polygon
        """
        for zone in self.warn_polygons:
            if cv2.pointPolygonTest(contour=zone, pt=point, measureDist=True) + buffer > 0:
                return True
        return False

    def get_cropped_image(self, frame):
        """
        :param point: frame , get crop image in safe zone of this scene
        :return:  cropped image
        """
        x = self.position[0]
        y = self.position[1]
        w= self.position[2]
        h = self.position[3]
        cropped_img = frame[y:y+h, x:x+w]
        cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=self.mask)
        return cropped_img

    def get_mask_by_black_white_img(self, bw_image_path):
        """
        :param bw_image_path: path to the black/white image indicating zones by white area
        :return: a list of polygons containing white region of the bw_image
        """
        # Get the black white image and chang it to 2-value image
        bw_image = cv2.imread(bw_image_path)
        bw_image = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(bw_image, thresh=128, maxval=256, type=0)
        contours, hierarchy = cv2.findContours(thresh, mode=1, method=1)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        self.position = (x, y, w, h)
        self.mask = thresh[y:y+h, x:x+w]

class SaftyBeltDetector():
    def __init__(self, gpu_id=0):
        self.debug = True
        self.num_classes = 3
        self.person_polygons = None
        gpu_id = "cuda:" + str(gpu_id)
        self.device = torch.device(gpu_id)
        self.model = BiSeNet(self.num_classes, backbone='resnet50', pretrained_base=False)
        self.model_half = BiSeNet(self.num_classes, backbone='resnet50', pretrained_base=False)
        params = torch.load("./model/bisenet_resnet50_pascal_voc.pth",map_location=self.device)
        self.model.load_state_dict(params)
        self.model_half.load_state_dict(params)

        self.model = self.model.to(self.device)
        self.model_half = self.model_half.to(self.device).half()

        self.model.eval()
        self.model_half.eval()
        print('Finished loading model!')

        self.sm = SceneManager(bw_image_dir="./scenes")
        for sceneId in config["scene_belt"]:
            self.sm.create_scene(sceneId=sceneId)

        self.size = (1920, 1080)
        # self.video_write = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'XVID'), 25, self.size)

    def point_warn_zone_test(self, point, buffer=0):
        """
        :param point: Point in (x, y) format , test if a point is in safe zone of this scene
        :return: True if point in the polygon
        """
        for zone in self.person_polygons:
            if cv2.pointPolygonTest(contour=zone, pt=point, measureDist=True) + buffer > 0:
                return True
        return False

    def function_1(self, x, a, b):
        return a*x + b
        
    def function_2(self, x, a, b, c):
        return a*x*x + b*x + c

    def get_cross_point(self, l1, l2):
        """
        :param l1=(a0,b0,c0)  l2=(a1,b1,c1)
        :return: cross point
        """
        d = l1[0] * l2[1] - l2[0] * l1[1]            #d=a0*b1-a1*b0
        x = (l1[1] * l2[2] - l2[1] * l1[2])*1.0 / d  #x = (b0*c1-b1*c0)/d
        y = (l1[2] * l2[0] - l2[2] * l1[0])*1.0 / d  #y = (a1*c0-a0*c1)/d
        return (x,y)
        
    def calc_intersection(self, outlines_bboxes, persons):
        outlines_x = outlines_bboxes[:, 0]
        outlines_y = outlines_bboxes[:, 1]*(-1)
        if self.debug:
            plt.scatter(outlines_x[:], outlines_y[:], 15, "green")
        a1, b1 = optimize.curve_fit(self.function_1, outlines_x, outlines_y)[0]
        a2, b2, c2 = optimize.curve_fit(self.function_2, outlines_x, outlines_y)[0]
        x1 = np.arange(0, self.width, 1)
        y1 = a1*x1 + b1
        x2 = np.arange(0, self.width, 1)
        y2 = a2*x2*x2 + b2*x2 + c2
        if self.debug:
            plt.plot(x1, y1, "blue")
            plt.plot(x2, y2, "blue")
        cx = int(np.sum(outlines_x)/outlines_bboxes.shape[0])
        cy = int(np.sum(outlines_y)/outlines_bboxes.shape[0])
        line_point_center = (cx,cy)
        line_point_center = np.array(line_point_center)
        if self.debug:
            plt.scatter(cx, cy, 50, "m")
        dis_max = 1000000
        j = 0
        for i in range(len(persons)):
            person_bboxes = persons[i]['zone'].reshape(persons[i]['zone'].shape[0], persons[i]['zone'].shape[2])
            cx_p = int(np.sum(person_bboxes[:, 0])/person_bboxes.shape[0])
            cy_p = int(np.sum(person_bboxes[:, 1]*(-1))/person_bboxes.shape[0])
            if self.debug:
                plt.scatter(cx_p, cy_p, 50, "m")
            person_point_center = (cx_p,cy_p)
            person_point_center = np.array(person_point_center)
            dis = np.sqrt(np.sum(np.square(line_point_center-person_point_center)))
            if dis <= dis_max:
                dis_max = dis
                j = i

        person_bboxes  = persons[j]['zone'].reshape(1, persons[j]['zone'].shape[0], persons[j]['zone'].shape[2])
        person_x = person_bboxes[0, :, 0]
        person_y = person_bboxes[0, :, 1]*(-1)
        person_x_min = person_bboxes[0, :, 0].min()
        person_x_max = person_bboxes[0, :, 0].max()
        if self.debug:
            plt.scatter(person_x[:], person_y[:], 15, "green")
        x1 = np.arange(person_x_min, person_x_max, 1)
        y1 = -(a1*x1 + b1)
        x2 = np.arange(person_x_min, person_x_max, 1)
        y2 = -(a2*x2*x2 + b2*x2 + c2)
        pts1 = np.vstack([x1, y1]).T
        pts2 = np.vstack([x2, y2]).T
        for point in pts1:
            point = tuple(point)
            if self.point_warn_zone_test(point):
                print("##########have cross point_1: ",point)
                persons[j]['b_crossed'] = True
                if self.debug:
                    plt.scatter(point[0], -point[1], 25, "red")
                break
                
        for point in pts2:
            point = tuple(point)
            if self.point_warn_zone_test(point):
                print("##########have cross point_2: ",point)
                persons[j]['b_crossed'] = True
                if self.debug:
                    plt.scatter(point[0], -point[1], 25, "red")
                break

    def intersection_realization(self, gray, frame, sceneId):
        scene = self.sm.get_scene(sceneId)
        cv2.polylines(frame, scene.warn_polygons, True, (0, 255, 255), 2)
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        ret, th_belt = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        gray[gray == 2] = 0
        ret, th_person = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY) 

        dilated_belt = cv2.dilate(th_belt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        erode_belt = cv2.erode(dilated_belt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        erode_belt = cv2.medianBlur(erode_belt, 5)

        dilated_person = cv2.dilate(th_person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        erode_person = cv2.erode(dilated_person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        erode_person = cv2.medianBlur(erode_person, 5)

        contours_belt, hierarchy = cv2.findContours(erode_belt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_person, hierarchy = cv2.findContours(erode_person,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if self.debug:
            belt_polygons = [cv2.approxPolyDP(curve=contour, epsilon=5, closed=True) for contour in contours_belt]
            cv2.drawContours(frame, belt_polygons, -1, (0, 255, 0), 2)

        #lines generate based on limitation factor
        length1 = len(contours_belt)
        line = []
        contours_belt_total = 0
        for i in range(length1):
            if len(contours_belt[i]) > 40:
                print("#####len contours_belt =", len(contours_belt[i]))
                line.append(contours_belt[i])
                contours_belt_total = contours_belt_total+len(contours_belt[i])
        lines = tuple(line)

        #person generate based on limitation factor
        length2 = len(contours_person)
        person = []
        for j in range(length2):
            x, y, w, h = cv2.boundingRect(contours_person[j])
            point = (int(x+w/2), int(y+h))
            b_in_zone = scene.point_zone_test(point)
            if b_in_zone == False:
                continue
            peron_area = cv2.contourArea(contours_person[j])
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            print("#####person:area=%d (10000), contours=%d h/w=%f (1.5)"%(peron_area, len(contours_person[j]), h/w))
            if len(contours_person[j]) > 100 and h/w > 1.5 and peron_area >10000:
                person.append(contours_person[j]) 
        persons = tuple(person)
        print('#'*10, 'have', len(persons), 'persons')
        self.person_polygons = [ cv2.convexHull(person) for person in persons]
        if self.debug:
            cv2.drawContours(frame, self.person_polygons, -1, (0, 0, 255), 2)

        outlines = []
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                point = (lines[i][j][0][0],lines[i][j][0][1])
                if self.point_warn_zone_test(point):
                    continue
                else:
                    outlines.append(lines[i][j])
        outlines = np.array(outlines)
        if len(outlines) <= 41 and len(persons) > 0:
            print("len outlines_bboxes = %d, persons=%d"%(len(outlines), len(persons)))
            plt.clf()
            return True
        elif len(persons) == 0:
            print("len outlines_bboxes = %d, persons=%d"%(len(outlines), len(persons)))
            plt.clf()
            return False
            
        outlines_bboxes  = outlines.reshape(1, outlines.shape[0], outlines.shape[2]) #n*1*2 => 1*n*2
        
        kmeans_outlines = []
        if len(persons) > 1:
            start = time.time()
            kmeans=KMeans(n_clusters=len(persons))
            kmeans.fit(np.squeeze(outlines_bboxes,0))
            end = time.time()
            print("kmeans time cost=%f ms"%((end-start)*1000))
            X = outlines_bboxes[0, :, 0]
            Y = outlines_bboxes[0, :, 1]
            outline = []
            for i in range(len(persons)):
                for j,value in enumerate(kmeans.labels_):
                    if value == i:
                        outline.append([X[j],Y[j]])
                kmeans_outlines.append(outline)
                outline = []
        else:
            kmeans_outlines = outlines_bboxes

        persons_map = {}
        for i in range(len(persons)):
            persons_map[i] = {'b_crossed':False, 'zone':persons[i]}
            
        for i in range(len(kmeans_outlines)):
            outlines = np.array(kmeans_outlines[i])
            outlines_hull = outlines.reshape(outlines.shape[0], 1, outlines.shape[1])
            x, y, w, h = cv2.boundingRect(outlines_hull) 
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            person_area = 0
            if w/h > 1:
                hull_line = cv2.convexHull(outlines_hull)
                for j in range(len(self.person_polygons)):
                    hull_persons = np.array(self.person_polygons[j])
                    person_area = cv2.contourArea(hull_persons)+person_area
                line_area = cv2.contourArea(hull_line)
                if line_area > person_area:
                    print("!!!!!![warning]:line_area=%d > person_area=%d"%(line_area, person_area))
                    plt.clf()
                    return False
            if self.debug:
                #colors = [np.random.randint(0, 255) for _ in range(3)]
                hull = cv2.convexHull(outlines_hull)
                cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
                
            self.calc_intersection(outlines, persons_map)
            
        print(" ")
        for i in range(len(persons_map)):
            if persons_map[i]['b_crossed'] == False:
                return True
                
        return False

    def preprocess(self, frame):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        try:
            img = Image.fromarray(img)
            img = transform(img)
            img = img.unsqueeze(0)
        except: 
            print('Warning, there is an image problem, save this picture')
            return torch.ones(1)
        return img

    def write_frame(self, frame_input):
        daytime = datetime.datetime.now().strftime('%Y-%m-%d')
        save_path = 'image_belt/'+daytime
        is_exist = os.path.exists(save_path)
        if not is_exist:
            os.umask(0)
            os.makedirs(save_path)
        hourtime = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        file_name = os.path.join(save_path,"%s-%s.jpg" % (daytime,hourtime))
        cv2.imwrite(file_name, frame_input)

    def prediction(self, frame, sceneId):
        #get crop image
        #scene = self.sm.get_scene(sceneId)
        img_org = frame.copy()
        #cropped_img = scene.get_cropped_image(frame)
        image =self.preprocess(frame)
        #image = image.to(self.device)
        image_half = image.to(self.device).half()
        '''
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = self.model(image)
        torch.cuda.synchronize()
        end = time.time()
        print("inference time=%f ms"%((end-start)*1000))
        '''
        torch.cuda.synchronize()
        start2 = time.time()
        with torch.no_grad():
            output_half = self.model_half(image_half)
        torch.cuda.synchronize()
        end2 = time.time()
        #print("inference_half time=%f ms"%((end2-start2)*1000))
        #print(torch.max(torch.abs(output_half[0] - output[0])))

        pred = torch.argmax(output_half[0], 1).squeeze(0).cpu().data.numpy()
        gray_value = np.uint8(pred)
        flag = self.intersection_realization(gray_value, frame, sceneId)
        if flag == False:
            cv2.putText(frame, "NORMAL", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 255, 0), 2)
        else:
            cv2.putText(frame, "WARNING", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.write_frame(img_org)
        #self.video_write.write(frame)
        imshow_name = "belt"+str(sceneId)
        cv2.imshow(imshow_name, frame)
        if cv2.waitKey(1) & 0xFF == (ord('q') or ord('Q')):
            exit(0)
        
        #if self.debug:
        if False:
            now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
            plt.title("test")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(0,self.width)
            plt.ylim(-self.height,0)
            global num
            pic_name = "./screenshots/"+str(sceneId)+"_"+now+"_"+str(num)+".png"
            plt.savefig(pic_name)
            #plt.show() 
            plt.clf()
            img_name = "./screenshots/"+str(sceneId)+"_"+now+"_"+str(num)+".jpg"
            cv2.imwrite(img_name,frame)
            num = num+1
        return flag, frame

class SafetyBeltWraper(object):
    def __init__(self, gpu_id=0):
        """
        Wrapper of beltWraper, declaring queues and register recall
        :param gpu_id:
        """
        self.alertType = 1
        self.eventsByScene = {}
        for sceneId in config["scene_belt"]:
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
        self.qn_in = config['safetyBelt_q_in']
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
        self.detector = SaftyBeltDetector(gpu_id=gpu_id)


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
        logging.getLogger('matplotlib.font_manager').disabled = True
        h = logging.handlers.QueueHandler(log_queue)  # Just the one handler needed
        root = logging.getLogger()
        root.addHandler(h)
        # send all messages, for demo; no other level or filter logic applied.
        root.setLevel(logging.DEBUG)

        
        

        def callback(ch, method, properties, body):


            start_time = time.time()
            obj_json = self.getJsonObj(body=body)
            sceneId = str(obj_json["placeid"])
            picId = sceneId
            # Getting json string from mq and get the image array
            timeID =  str(obj_json["time"])
            # print("%s %s %s"%(timeID, sceneId, "#" * 8))

            img_opencv, h, w, c = self.getOpencvImg(obj_json)
            # Prediction
            is_alarm, frame = self.detector.prediction(img_opencv, sceneId)

            global queue_rtsp_dict
            print(queue_rtsp_dict, '3'*20)
            print('is_alarm=', is_alarm, '----------------')
            if is_alarm:
                self.eventsByScene[sceneId]['eventId'] = picId + "|" + str(self.alertType)
                self.eventsByScene[sceneId]['eventPics'].append({'picId': picId, 'alertObjects': {}})
                response_dict = {
                     'protocol': '1.0.0',
                     'alertType': self.alertType,
                     # 'latestPicId': obj_json['picId'],
                     # 'eventId': self.eventsByScene[sceneId]['eventId'],
                     'eventPics': self.eventsByScene[sceneId]['eventPics'],
                     'Time01StampID': timeID,
                }
                # dumps json obj
                response_dict = json.dumps(response_dict, sort_keys=True, indent=2)
                print("response_dict=", response_dict)
                # self.ch.basic_publish(exchange=self.ex_out, routing_key=self.qn_out, body=response_dict)

                # add temp send 
                print('----------------', ' save image')


                queueid = int(sceneId)
                logger = logging.getLogger(str(queueid))
                now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
                img_name= str(sceneId) + '_'+ now + '.jpg'
                cv2.imwrite('./screenshots/' + img_name, frame)

                warning_signal = 'no-belt-in-area'
                print('----------------', logging.CRITICAL, warning_signal + '#' + img_name + '#' +queue_rtsp_dict.get(queueid, None)[5]
                            + '#' + queue_rtsp_dict.get(queueid, None)[6])
                logger.log(logging.CRITICAL, warning_signal + '#' + img_name + '#' +queue_rtsp_dict.get(queueid, None)[5]
                            + '#' + queue_rtsp_dict.get(queueid, None)[6])
                print('@after logger.log()')

                #  
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
                                       args=(log_queue, listener_configurer_belt))
    listener.start()

    

    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', default='0')
    args = parser.parse_args()

    beltWrapper = SafetyBeltWraper(gpu_id=int(args.gpu))
    beltWrapper.running(log_queue)


