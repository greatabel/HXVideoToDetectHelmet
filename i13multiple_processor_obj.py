import cv2

import time
import multiprocessing as mp

from gluoncv import model_zoo, data, utils
import mxnet as mx

import os

import logging
import logging.handlers
from random import choice, random


import csv
import ast
import i13process_frame
import i11qy_wechat

import pika
import json
import numpy
import base64
import i13rabbitmq_config
#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks


rtsp_file_path = 'i13rtsp_list.csv'
queue_rtsp_dict = {}
img_name = ''

# MXNet报Running performance tests to find the best convolution algorithm
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

def listener_configurer():
    # logging.getLogger("pika").propagate = False

    root = logging.getLogger()
    h = logging.handlers.RotatingFileHandler('temp.log', 'a', 3000000, 10)
    f = logging.Formatter('%(asctime)s %(processName)-8s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)

# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()

            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            if 'pika' in record.name:
                print('-'*30, record, '-'*30)
            elif 'pika' not in record.name:
                                # print('record','*'*20,record.name, record)
                logger = logging.getLogger(record.name)
                # logger.handle(record)  # No level or filter logic applied - just do it!
                warning_processor(logger, record)
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


queueid_warning_dict = {}
queueid_lastsendtime_dict = {}
def warning_processor(logger, record):
    global queueid_warning_dict, queueid_lastsendtime_dict
    queueid = record.name
    logger.handle(record)  # No level or filter logic applied - just do it!
    # recore.name 就是queueid 代表rtsp的获取队列id
    if queueid_warning_dict.get(record.name) is None:
        queueid_warning_dict[record.name] = [record.asctime]
    else:
        queueid_warning_dict[record.name].append(record.asctime)
        # 定期清空 记录时间的list， 容量达到50， 就清空到只剩下最新的1个
        if len(queueid_warning_dict[record.name]) > 50:
            del queueid_warning_dict[record.name][:len(queueid_warning_dict[record.name])-1]
            print('delete recordtime list')

    # print('\n', '-^-'*10, queueid_warning_dict)
    helmet_color = ''
    warning_signal, img_name, area, senduserids = record.msg.split('#') 

    if warning_signal == 'red-hat-in-area':
        # # 警告音 
        duration = 1  # seconds
        freq = 440  # Hz
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        helmet_color = '红色头盔'
    elif warning_signal == 'yellow-hat-in-area':
        # # 警告音 yellow
        duration = 0.5  # seconds
        freq = 660  # Hz
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        helmet_color = '黄色头盔'
    elif warning_signal == 'without-hat-in-area':
        helmet_color = '未佩戴头盔'

    msg = i13rabbitmq_config.FactoryName + area + ' 发生 ' + helmet_color + '非授权进入区域'

    # timelimit 为在限制区域时间存在达到多少秒后，才会发送消息报警
    timelimit = 2
    # time_span_limit 代表在这个时间内只能发一次消息报警
    time_span_limit = 180
    if len(queueid_warning_dict[record.name]) >= timelimit:
        sendmsg_flag = i13process_frame.proces_timelist(queueid_warning_dict[record.name], timelimit)
        
        now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        if queueid_lastsendtime_dict.get(record.name) is not None:

            timespan_flag = i13process_frame.compare_time(now, queueid_lastsendtime_dict.get(record.name), time_span_limit)
            # 时间间隔没到 也不能发送
            if timespan_flag == False:
                sendmsg_flag = False
        if sendmsg_flag:
            i11qy_wechat.send_text_and_image_wechat(img_name, msg, senduserids)


            queueid_lastsendtime_dict[record.name] = now


class Hat_and_Person_Detector():
    def __init__(self, processid, log_queue):
        print('processid', processid, '-^-'*5)
        self.log_queue = log_queue

        self.log_queue = log_queue
        self.args = i13process_frame.parse_args()
        print('是否使用GPU:', self.args.gpu)
        if self.args.gpu:
            print('gpu_num:', self.args.gpu_num)
            self.ctx = mx.gpu(self.args.gpu_num)
        else:
            self.ctx = mx.cpu()
        # ctx = mx.cpu()

        self.net = model_zoo.get_model(self.args.network, pretrained=False)
        
        classes = ['hat', 'person']
        for param in self.net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
        self.net.reset_class(classes)
        self.net.collect_params().reset_ctx(self.ctx)

        if self.args.network == 'yolo3_darknet53_voc':
            self.net.load_parameters('darknet.params',ctx=self.ctx)
            print('use darknet to extract feature')
        elif self.args.network == 'yolo3_mobilenet1.0_voc':
            self.net.load_parameters('mobilenet1.0.params',ctx=self.ctx)
            print('use mobile1.0 to extract feature')
        else:
            self.net.load_parameters('mobilenet0.25.params',ctx=self.ctx)
            print('use mobile0.25 to extract feature')
            print('#'*20)

        



    def process(self, frame,  rect, default_enter_rule, queueid=None):
        h = logging.handlers.QueueHandler(self.log_queue)  # Just the one handler needed
        root = logging.getLogger()
        root.addHandler(h)
        # send all messages, for demo; no other level or filter logic applied.
        root.setLevel(logging.DEBUG)

        logger = logging.getLogger(str(queueid))
        if frame is not None:
            print('here0', rect, default_enter_rule, type(frame))
            cv2.imwrite("filename_process.png", frame)

            # new_frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            new_frame = mx.nd.array(frame).astype('uint8')

            # img_float32 = numpy.float32(frame)
            # lab_image = cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB)
            cv2.imwrite("filename_R.png", frame)
            # new_frame = mx.nd.array(cv2.cvtColor(lab_image, cv2.COLOR_RGB2BGR)).astype('uint8')

            print('here1')
            # x, orig_img = data.transforms.presets.yolo.load_test(frame, short=args.short)
            # x, orig_img = data.transforms.presets.yolo.transform_test(new_frame, short=args.short)
            x, orig_img = data.transforms.presets.yolo.transform_test(new_frame,  short=512, max_size=2000)
            # print('Shape of pre-processed image:', x.shape)
            x = x.as_in_context(self.ctx)
            box_ids, scores, bboxes = self.net(x)
            print('here2')
            # render_as_image(bboxes)
            # ---
            # if isinstance(bboxes[0], mx.nd.NDArray):

            #     bboxes_a = bboxes[0].asnumpy()
            # for i, bbox in enumerate(bboxes_a):
            #     xmin, ymin, xmax, ymax = [int(x) for x in bbox]
            #     print(xmin, ymin, xmax, ymax)
            #     if xmin > -1 :
            #         cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), 122, 2)



            # for idx in range(len(bboxes.asnumpy())):
            #     x, y, w, h = cv2.boundingRect(bboxes.asnumpy()[idx])
            #     mask[y:y+h, x:x+w] = 0
            #     print("Box {0}: ({1},{2}), ({3},{4}), ({5},{6}), ({7},{8})".format(idx,x,y,x+w,y,x+w,y+h,x,y+h))
            #     cv2.drawContours(mask, bboxes.asnumpy(), idx, (255, 255, 255), -1)
            #     r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

            #----


            # ax = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes,thresh=args.threshold)
            x, warning_signal = i13process_frame.forked_version_cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], 
                                            class_names=self.net.classes,thresh=self.args.threshold, hx_rect=rect, default_enter_rule=default_enter_rule)
            # x = origin_cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], 
            #                                 class_names=net.classes,thresh=args.threshold)

            # print('#'*10, type(bboxes), bboxes.shape)
            # 让窗口可以调整
            print('here3')
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow('image', x)
            
            print('processing:', queueid)
            if warning_signal is not None:
                print('@'*20, ' save image')
                now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
                img_name= str(queueid) + '_'+ now + '.jpg'
                cv2.imwrite('screenshots/' + img_name,x)
                # 发送企业维新消息
                # print('img_name ',img_name, 'queue_rtsp_dict=', queue_rtsp_dict )
                # print('*-*-'*20, '\n')
                # i11qy_wechat.send_text_and_image_wechat(img_name, queue_rtsp_dict.get(queueid, None)[5]+'发生非授权头盔进入区域',
                #     queue_rtsp_dict.get(queueid, None)[6])
                print('@'*30, queue_rtsp_dict)
                print('#'*5, logging.CRITICAL, warning_signal + '#' + img_name + '#' +queue_rtsp_dict.get(queueid, None)[5]
                            + '#' + queue_rtsp_dict.get(queueid, None)[6])
                logger.log(logging.CRITICAL, warning_signal + '#' + img_name + '#' +queue_rtsp_dict.get(queueid, None)[5]
                            + '#' + queue_rtsp_dict.get(queueid, None)[6])
                print('@after logger.log()')
            # if cv2.waitKey(1) == 27:
            #         break
        else:
            print('skip frame from queueid=', queueid)

def receiver(host, processid, log_queue):
    # at yangxin shuini factory
    # host = '10.248.68.249'
    
    # host = '127.0.0.1'
    credentials = pika.PlainCredentials('test', 'test')
    parameters = pika.ConnectionParameters(host,
                                       5672,
                                       '/',
                                       credentials)
    detector = Hat_and_Person_Detector(processid, log_queue)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(
        queue='hello',
        arguments=i13rabbitmq_config.ARGUMENTS,
        )

    # channel.basic_consume(on_message_callback=lambda ch, method, properties, body: image_get_v0(ch, 
    #                         method, properties, body, processid=processid, detector=detector),
    #                       queue='hello',
    #                       auto_ack=True)
    channel.basic_consume(on_message_callback=lambda ch, method, properties, body: image_get_v0(ch, 
                            method, properties, body, processid=processid, detector=detector),
                          queue='hello',
                           auto_ack=False)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()






# def image_get_v0(quelist, window_name, log_queue):
def image_get_v0(ch, method, properties, body, processid, detector):



    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    print('************************************')


    print('in image_get_v0 ', ch, method, properties, 'processid==>', processid)
    msg = json.loads(body)
    frame = None
    if msg is not None:
        img = base64.b64decode(msg['img'].encode())
        
        # get image array
        frame = cv2.imdecode(numpy.fromstring(img, numpy.uint8), 1)

        # frame = numpy.asarray(msg["img"])
        queueid = int(msg['placeid'])
        # print(" [x] Received %r" % msg)
        # imgdata = base64.b64decode(msg['img'])
        print(msg['placeid'], '@'*10, msg['time'])
        print(type(frame), '#'*10)
        cv2.imwrite("filename.png", frame)

        print('-----------------------')

        # print(queue_rtsp_dict, queueid, type(queueid))
        # frame, queueid = q.get()
        rect = None 
        if queue_rtsp_dict.get(queueid, None)[7] != None and \
             queue_rtsp_dict.get(queueid, None)[7].strip() != '':
            print('#'*30, 'here')
            rect = ast.literal_eval(queue_rtsp_dict.get(queueid, None)[7])
        default_enter_rule = queue_rtsp_dict.get(queueid, None)[8]
        detector.process(frame, rect, default_enter_rule, queueid)





    # args = i13process_frame.parse_args()
    # print('是否使用GPU:', args.gpu)
    # if args.gpu:
    #     ctx = mx.gpu()
    # else:
    #     ctx = mx.cpu()
    # # ctx = mx.cpu()

    # net = model_zoo.get_model(args.network, pretrained=False)
    
    # classes = ['hat', 'person']
    # for param in net.collect_params().values():
    #     if param._data is not None:
    #         continue
    #     param.initialize()
    # net.reset_class(classes)
    # net.collect_params().reset_ctx(ctx)

    # if args.network == 'yolo3_darknet53_voc':
    #     net.load_parameters('darknet.params',ctx=ctx)
    #     print('use darknet to extract feature')
    # elif args.network == 'yolo3_mobilenet1.0_voc':
    #     net.load_parameters('mobilenet1.0.params',ctx=ctx)
    #     print('use mobile1.0 to extract feature')
    # else:
    #     net.load_parameters('mobilenet0.25.params',ctx=ctx)
    #     print('use mobile0.25 to extract feature')
    #     print('#'*20)


    # level = choice(LEVELS)
    # message = choice(MESSAGES)
    # logger.log(level, message)
    # print('*** rect=', rect, type(rect))

    # cv2.imshow(ip, frame)
    # cv2.waitKey(1)

    

        
    # frame = '1.jpg'
    # x, orig_img = data.transforms.presets.yolo.load_test(frame, short=args.short)
    # x = x.as_in_context(ctx)
    # box_ids, scores, bboxes = net(x)
    # ax = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes,thresh=args.threshold)
    # cv2.imshow('image', orig_img[...,::-1])
    # cv2.waitKey(0)
    # cv2.imwrite(frame.split('.')[0] + '_result.jpg', orig_img[...,::-1])
    # cv2.destroyAllWindows()

    # Image pre-processing




def run_multi_camera(camera_ip_l):
    global queue_rtsp_dict
    for line in camera_ip_l:
        print('line:', line, type(line))
        queue_rtsp_dict[int(line[0])] = line[1:]
    log_queue = mp.Queue(-1)
    listener = mp.Process(target=listener_process,
                                       args=(log_queue, listener_configurer))
    listener.start()

    # mp.set_start_method(method='spawn')  # init0
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]
    processes = []



    # -------------------- start ai processes
    num_of_ai_process = 8

    for i in range(0, num_of_ai_process):
        print('ai process', i)
        processes.append(mp.Process(target=receiver, args=(i13rabbitmq_config.Where_This_Server_ReadFrom,
                                                           i, log_queue)))
    # -------------------- end   ai processes

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


def load_rtsp_list():
    with open(rtsp_file_path, newline='') as f:
        reader = csv.reader(f)
        rtsp_list = list(reader)
    # print(data, '#'*10, data[0],'\n', data[0][0])
    return rtsp_list

if __name__ == '__main__':
    camera_ip_l = load_rtsp_list()
    # python3 i13multiple_processor_obj.py --gpu=True --network=yolo3_mobilenet0.25_voc
    run_multi_camera(camera_ip_l) 
