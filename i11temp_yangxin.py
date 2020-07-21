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
import i11process_frame
import i11qy_wechat
#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks


rtsp_file_path = 'i11rtsp_list.csv'
queue_rtsp_dict = {}
img_name = ''


def listener_configurer():
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
        # 定期清空 记录时间的list， 容量达到30， 就清空到只剩下最新的1个
        if len(queueid_warning_dict[record.name]) > 30:
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
        helmet_color = '红色'
    elif warning_signal == 'yellow-hat-in-area':
        # # 警告音 yellow
        duration = 0.5  # seconds
        freq = 660  # Hz
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        helmet_color = '黄色'

    msg = area + ' 发生 ' + helmet_color + '非授权头盔进入区域'

    # timelimit 为在限制区域时间存在达到多少秒后，才会发送消息报警
    timelimit = 5
    # time_span_limit 代表在这个时间内只能发一次消息报警
    time_span_limit = 180
    if len(queueid_warning_dict[record.name]) >=5:
        sendmsg_flag = i11process_frame.proces_timelist(queueid_warning_dict[record.name], timelimit)
        
        now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        if queueid_lastsendtime_dict.get(record.name) is not None:

            timespan_flag = i11process_frame.compare_time(now, queueid_lastsendtime_dict.get(record.name), time_span_limit)
            if timespan_flag == False:
                sendmsg_flag = False
        if sendmsg_flag:
            i11qy_wechat.send_text_and_image_wechat(img_name, msg, senduserids)


            queueid_lastsendtime_dict[record.name] = now

def image_put(q, queueid):
    name = queue_rtsp_dict.get(queueid, None)[0]
    pwd = queue_rtsp_dict.get(queueid, None)[1]
    ip = queue_rtsp_dict.get(queueid, None)[2]
    channel = queue_rtsp_dict.get(queueid, None)[3]
    camera_corp = queue_rtsp_dict.get(queueid, None)[4]


    # 大华的情况 ：
    if camera_corp == 'dahua':
        full_vedio_url = "rtsp://%s:%s@%s/cam/realmonitor?channel=%s&subtype=0" % (name, pwd, ip, channel)
    # 网络摄像头是海康:
    elif camera_corp == 'hik':
        full_vedio_url = "rtsp://%s:%s@%s/Streaming/Channels/%s" % (name, pwd, ip, channel)
    full_vedio_url = i11process_frame.deal_specialchar_in_url(full_vedio_url)
    cap = cv2.VideoCapture(full_vedio_url)
    # if cap.isOpened():
    #     print('HIKVISION')
    # else:
    #     cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (name, pwd, ip, channel))
    #     print('DaHua')

    # 通过timeF控制多少帧数真正读取1帧到队列中
    timeF = 15
    count = 1 
    while True:
        res, frame = cap.read()
        if count % timeF == 0:
            # print('pick=', count)
            q.put((cap.read()[1], queueid))
            q.get() if q.qsize() > 1 else time.sleep(0.01)
        count += 1
        # print('count=', count)

# def image_get(quelist, window_name):
#     cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
#     while True:
#         for q in quelist:
#             frame = q.get()
#             cv2.imshow(window_name, frame)
#             cv2.waitKey(1)

def image_get_v0(quelist, window_name, log_queue):

    h = logging.handlers.QueueHandler(log_queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)




    args = i11process_frame.parse_args()
    print('是否使用GPU:', args.gpu)
    if args.gpu:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
    # ctx = mx.cpu()

    net = model_zoo.get_model(args.network, pretrained=False)
    
    classes = ['hat', 'person']
    for param in net.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()
    net.reset_class(classes)
    net.collect_params().reset_ctx(ctx)

    if args.network == 'yolo3_darknet53_voc':
        net.load_parameters('darknet.params',ctx=ctx)
        print('use darknet to extract feature')
    elif args.network == 'yolo3_mobilenet1.0_voc':
        net.load_parameters('mobilenet1.0.params',ctx=ctx)
        print('use mobile1.0 to extract feature')
    else:
        net.load_parameters('mobilenet0.25.params',ctx=ctx)
        print('use mobile0.25 to extract feature')
        print('#'*20)


    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    while True:
        for q in quelist:

            frame, queueid = q.get()
            rect = ast.literal_eval(queue_rtsp_dict.get(queueid, None)[7])


            logger = logging.getLogger(str(queueid))
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
            new_frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

            # x, orig_img = data.transforms.presets.yolo.load_test(frame, short=args.short)
            # x, orig_img = data.transforms.presets.yolo.transform_test(new_frame, short=args.short)
            x, orig_img = data.transforms.presets.yolo.transform_test(new_frame,  short=512, max_size=2000)
            # print('Shape of pre-processed image:', x.shape)
            x = x.as_in_context(ctx)
            box_ids, scores, bboxes = net(x)

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
            x, warning_signal = i11process_frame.forked_version_cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], 
                                            class_names=net.classes,thresh=args.threshold, hx_rect=rect)
            # x = origin_cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], 
            #                                 class_names=net.classes,thresh=args.threshold)

            # print('#'*10, type(bboxes), bboxes.shape)
            # 让窗口可以调整
            
            cv2.imshow('image', orig_img[...,::-1])
            print('processing:', window_name)
            if warning_signal is not None:
                print('@'*20, ' save image')
                now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
                img_name= str(queueid) + '_'+ now + '.jpg'
                cv2.imwrite('screenshots/' + img_name, orig_img[...,::-1])
                # 发送企业维新消息
                # print('img_name ',img_name, 'queue_rtsp_dict=', queue_rtsp_dict )
                # print('*-*-'*20, '\n')
                # i11qy_wechat.send_text_and_image_wechat(img_name, queue_rtsp_dict.get(queueid, None)[5]+'发生非授权头盔进入区域',
                #     queue_rtsp_dict.get(queueid, None)[6])

                logger.log(logging.CRITICAL, warning_signal + '#' + img_name + '#' +queue_rtsp_dict.get(queueid, None)[5]
                            + '#' + queue_rtsp_dict.get(queueid, None)[6])
            if cv2.waitKey(1) == 27:
                    break


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_multi_camera(camera_ip_l):
    global queue_rtsp_dict
    log_queue = mp.Queue(-1)
    listener = mp.Process(target=listener_process,
                                       args=(log_queue, listener_configurer))
    listener.start()

    # mp.set_start_method(method='spawn')  # init0
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]
    processes = []



    queueid = 0
    for queue, camera_ip in zip(queues, camera_ip_l):
        # rect = ast.literal_eval(camera_ip[7])
        # print(camera_ip, camera_ip[0], '##', rect, type(rect))
        processes.append(mp.Process(target=image_put, 
            args=(queue, queueid)))

        queue_rtsp_dict[queueid] = camera_ip
        queueid += 1
        # processes.append(mp.Process(target=image_get, args=(queue, camera_ip[2])))


    # -------------------- start ai processes
    num_of_ai_process = 3
    chunk_queues = list(chunks(queues, int(len(queues)/num_of_ai_process)))
    print(chunk_queues)
    for i in range(0, num_of_ai_process):
        print('ai process', i)
        processes.append(mp.Process(target=image_get_v0, args=(chunk_queues[i], str(i),log_queue)))
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
    # python3 i11temp_yangxin.py --gpu=True --network=yolo3_mobilenet0.25_voc
    run_multi_camera(camera_ip_l) 
