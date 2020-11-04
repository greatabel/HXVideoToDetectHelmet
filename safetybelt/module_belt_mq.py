import os
import sys
import torch
from torchvision import transforms
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
            return self.Scenes['default']


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
            warn_white_path = os.path.join(bw_image_dir, "default.jpg")
        self.get_mask_by_black_white_img(bw_image_path=warn_white_path)

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
        self.debug = False
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
        self.video_write = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'XVID'), 25, self.size)

    def point_warn_zone_test(self, point, buffer=0):
        """
        :param point: Point in (x, y) format , test if a point is in safe zone of this scene
        :return: True if point in the polygon
        """
        for zone in self.person_polygons:
            if cv2.pointPolygonTest(contour=zone, pt=point, measureDist=True) + buffer > 0:
                return True
        return False

    def function(self, x, a, b):
        return a*x + b

    def get_cross_point(self, l1, l2):
        """
        :param l1=(a0,b0,c0)  l2=(a1,b1,c1)
        :return: cross point
        """
        d = l1[0] * l2[1] - l2[0] * l1[1]            #d=a0*b1-a1*b0
        x = (l1[1] * l2[2] - l2[1] * l1[2])*1.0 / d  #x = (b0*c1-b1*c0)/d
        y = (l1[2] * l2[0] - l2[2] * l1[0])*1.0 / d  #y = (a1*c0-a0*c1)/d
        return (x,y)

    def intersection_realization(self, gray, frame):
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
                line.append(contours_belt[i])
                contours_belt_total = contours_belt_total+len(contours_belt[i])
        lines = tuple(line) 

        #person generate based on limitation factor
        length2 = len(contours_person)
        person = []
        for j in range(length2):
            if len(contours_person[j]) > 100 :
                person.append(contours_person[j]) 

        persons = tuple(person)
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
        #print("======len outlines=",len(outlines))
        if len(outlines) <= 41 and len(persons) > 0:
            #print("len outlines_bboxes = %d, persons=%d"%(len(outlines), len(persons)))
            plt.clf()
            return True
        elif len(persons) == 0:
            print("len outlines_bboxes = %d, persons=%d"%(len(outlines), len(persons)))
            plt.clf()
            return False

        if self.debug:
            hull = cv2.convexHull(outlines)
            cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)

        outlines_bboxes  = outlines.reshape(1, outlines.shape[0], outlines.shape[2])
        outlines_x = outlines_bboxes[0, :, 0]
        outlines_y = outlines_bboxes[0, :, 1]*(-1)
        if self.debug:
            plt.scatter(outlines_x[:], outlines_y[:], 15, "green")
        a, b = optimize.curve_fit(self.function, outlines_x, outlines_y)[0]
        x = np.arange(0, 1920, 1)
        y = a*x + b
        line_func_param = (a,-1, b)
        if self.debug:
            plt.plot(x, y, "blue")

        have_cross_point = False
        for i in range(len(persons)):
            if have_cross_point:
                break
            person_bboxes  = persons[i].reshape(1, persons[i].shape[0], persons[i].shape[2])
            person_x = person_bboxes[0, :, 0]
            person_y = person_bboxes[0, :, 1]*(-1)
            person_x_min = person_bboxes[0, :, 0].min()
            person_x_max = person_bboxes[0, :, 0].max()
            if self.debug:
                plt.scatter(person_x[:], person_y[:], 15, "green")
            x = np.arange(person_x_min, person_x_max, 1)
            y = -(a*x + b)
            pts = np.vstack([x, y]).T
            for point in pts:
                point = tuple(point)
                if self.point_warn_zone_test(point):
                    print("have cross point: ",point)
                    have_cross_point = True
                    if self.debug:
                        plt.scatter(point[0], -point[1], 25, "red")
                    break
        if have_cross_point == False:
            print("not have cross point")
            plt.clf()
            return True
        if self.debug:
            plt.title("test")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(0,1920)
            plt.ylim(-1080,0)
            global num
            pic_name = "/media/liujin/disk/project/safetybelt_detect/pic/"+str(num)+".png"
            plt.savefig(pic_name)
            #plt.show() 
            plt.clf()

        if len(persons) > 0 and len(lines) > 0:
            lines = np.concatenate(lines, axis=0)
            persons = np.concatenate(persons,axis=0)

            hull1 = cv2.convexHull(lines)
            hull2 = cv2.convexHull(persons)

            if self.debug:
                pic_name = "/media/liujin/disk/project/safetybelt_detect/pic/"+str(num)+".jpg"
                cv2.imwrite(pic_name,frame )
                num = num+1

            line_bboxes  = hull1.reshape(1, hull1.shape[0], hull1.shape[2])
            person_bboxes  = hull2.reshape(1, hull2.shape[0], hull2.shape[2])#(1,x,y,2),x w  y h

            line_top = line_bboxes[0, :, 1].min() 
            person_top = person_bboxes[0, :, 1].min()

            im1 = np.zeros(gray.shape, dtype = "uint8")
            im2 =np.zeros(gray.shape, dtype = "uint8")
     
            line_mask = cv2.fillPoly(im1, line_bboxes, 255)
            person_mask = cv2.fillPoly(im2, person_bboxes,255)

            masked_and_person_line = cv2.bitwise_and(line_mask, person_mask)#用mask把相同的区域填充进去,也就是0，也就是黑
            and_area_person_line =np.sum(np.float32(np.greater(masked_and_person_line,0)))
            person_area =np.sum(np.float32(np.greater(person_mask,0)))
            IOU1 = and_area_person_line/person_area
            flag =(line_top < person_top)

            if IOU1 >= 0 and contours_belt_total >50 and flag:
                print("###IOU1=%f, contours_belt_total=%d, flag=%d"%(IOU1,contours_belt_total,flag))
                return False
            else:
                #print("###IOU1=%f, contours_belt_total=%d, flag=%d"%(IOU1,contours_belt_total,flag))
                return True
        elif len(persons) > 0 and len(lines) == 0:
            #print("###len(persons)=%d, len(lines)=%d"%(len(persons),len(lines)))
            return True
        elif len(persons) == 0:
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
        #get crop image
        scene = self.sm.get_scene(sceneId)
        cropped_img = scene.get_cropped_image(frame)
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
        flag = self.intersection_realization(gray_value, frame)
        if flag == False:
            cv2.putText(frame, "NORMAL", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 255, 0), 2)
        else:
            cv2.putText(frame, "WARNING", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #self.write_frame(frame)
        #self.video_write.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == (ord('q') or ord('Q')):
            exit(0)
        
        return flag

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
        self.qn_in = config['frame_q_in']
        self.qn_out = config['frame_q_out']

        self.ex_in = config['frame_q_in']
        self.ex_out = config['frame_q_out']

        self.rtk_in = config['frame_q_in']
        self.rtk_out = config['frame_q_out']

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

    def running(self):
        def callback(ch, method, properties, body):
            start_time = time.time()
            obj_json = self.getJsonObj(body=body)
            # sceneId = str(obj_json["cameraId"]) + "_" + obj_json["recorderId"]
            sceneId = str(obj_json["placeid"])
            # picId = obj_json['picId']
            picId = sceneId
            # Getting json string from mq and get the image array
            # timeID = self.getTimeStr(obj_json)
            timeID =  str(obj_json["time"])
            # print("%s %s %s %s"%(timeID, obj_json['recorderId'], obj_json['cameraId'], "#" * 8))

            img_opencv, h, w, c = self.getOpencvImg(obj_json)
            # Prediction
            is_alarm = self.detector.prediction(img_opencv, sceneId)
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', default='0')
    args = parser.parse_args()

    beltWrapper = SafetyBeltWraper(gpu_id=int(args.gpu))
    beltWrapper.running()


