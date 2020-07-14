import cv2

import time
import multiprocessing as mp

from gluoncv import model_zoo, data, utils
import mxnet as mx
import i11process_frame
#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks


def image_put(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    if cap.isOpened():
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (name, pwd, ip, channel))
        print('DaHua')

    # 通过timeF控制多少帧数真正读取1帧到队列中
    timeF = 15
    count = 1 
    while True:
        res, frame = cap.read()
        if count % timeF == 0:
            # print('pick=', count)
            q.put(cap.read()[1])
            q.get() if q.qsize() > 1 else time.sleep(0.01)
        count += 1
        # print('count=', count)


def image_get(quelist, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        for q in quelist:
            frame = q.get()
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

def image_get_v0(quelist, window_name, rect):

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


    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    while True:
        for q in quelist:

            frame = q.get()
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
            x = i11process_frame.forked_version_cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], 
                                            class_names=net.classes,thresh=args.threshold, hx_rect=rect)
            # x = origin_cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], 
            #                                 class_names=net.classes,thresh=args.threshold)

            # print('#'*10, type(bboxes), bboxes.shape)
            # 让窗口可以调整
            
            # cv2.imshow('image', orig_img[...,::-1])
            print('processing:', window_name)
            if cv2.waitKey(1) == 27:
                    break


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_multi_camera():
    # user_name, user_pwd = "admin", "password"
    camera_ip_l = [
            ('admin', 'yxgl$666','192.168.200.182:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.183:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.190:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.204:554',1, 'hik'), 
            ('admin', '12345','192.168.200.76:554',1, 'hik'), 
            ('admin', '12345','192.168.200.91:554',1, 'hik'), 
            ('admin', '12345','192.168.200.95:554',1, 'hik'), 
            ('admin', '12345','192.168.200.97:554',1, 'hik'), 
            ('admin', '12345','192.168.200.95:554',1, 'hik'), 
            ('admin', '12345','192.168.200.97:554',1, 'hik'), 
            
            ('admin', 'yxgl$666','192.168.200.182:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.183:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.190:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.204:554',1, 'hik'), 
            ('admin', '12345','192.168.200.76:554',1, 'hik'), 
            ('admin', '12345','192.168.200.91:554',1, 'hik'), 
            ('admin', '12345','192.168.200.95:554',1, 'hik'), 
            ('admin', '12345','192.168.200.97:554',1, 'hik'), 
            ('admin', '12345','192.168.200.95:554',1, 'hik'), 
            ('admin', '12345','192.168.200.97:554',1, 'hik'), 
            
            ('admin', 'yxgl$666','192.168.200.182:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.183:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.190:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.204:554',1, 'hik'), 
            ('admin', '12345','192.168.200.76:554',1, 'hik'), 
            ('admin', '12345','192.168.200.91:554',1, 'hik'), 
            ('admin', '12345','192.168.200.95:554',1, 'hik'), 
            ('admin', '12345','192.168.200.97:554',1, 'hik'), 
            ('admin', '12345','192.168.200.95:554',1, 'hik'), 
            ('admin', '12345','192.168.200.97:554',1, 'hik'), 
            
            ('admin', 'yxgl$666','192.168.200.182:554',1, 'hik'), 
            ('admin', 'yxgl$666','192.168.200.183:554',1, 'hik'), 
            # ('admin', 'yxgl$666','192.168.200.190:554',1, 'hik'), 
            # ('admin', 'yxgl$666','192.168.200.204:554',1, 'hik'), 
            # ('admin', '12345','192.168.200.76:554',1, 'hik'), 
            # ('admin', '12345','192.168.200.91:554',1, 'hik'), 
            # ('admin', '12345','192.168.200.95:554',1, 'hik'), 
            # ('admin', '12345','192.168.200.97:554',1, 'hik'), 
            # ('admin', '12345','192.168.200.95:554',1, 'hik'), 
            # ('admin', '12345','192.168.200.97:554',1, 'hik'), 
            
            ]
    # camera_ip_l = [
    #     ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     ["admin", "admin123", "10.248.10.100:554", 3],
    #     ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     ["admin", "admin123", "10.248.10.100:554", 3],
    #     ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     ["admin", "admin123", "10.248.10.100:554", 3],
    #     ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     ["admin", "admin123", "10.248.10.100:554", 3],
    #     ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     ["admin", "admin123", "10.248.10.100:554", 3],

    #     ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     ["admin", "admin123", "10.248.10.100:554", 3],
    #     ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     ["admin", "admin123", "10.248.10.100:554", 3],
    #     ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],

    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],

    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],

    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],
    #     # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
    #     # ["admin", "admin123", "10.248.10.100:554", 3],


    #     # 把你的摄像头的地址放到这里，如果是ipv6，那么需要加一个中括号。
    # ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]
    processes = []



    
    for queue, camera_ip in zip(queues, camera_ip_l):
        print(camera_ip, camera_ip[0], '#', camera_ip[1])
        processes.append(mp.Process(target=image_put, args=(queue, camera_ip[0], camera_ip[1], camera_ip[2], camera_ip[3])))
        # processes.append(mp.Process(target=image_get, args=(queue, camera_ip[2])))



    # -------------------- start ai processes
    num_of_ai_process = 8
    chunk_queues = list(chunks(queues, int(len(queues)/num_of_ai_process)))
    print(chunk_queues)
    for i in range(0, num_of_ai_process):
        print('ai process', i)
        processes.append(mp.Process(target=image_get_v0, args=(chunk_queues[i], str(i), None)))
    # -------------------- end   ai processes

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()



if __name__ == '__main__':
    print('here')
    # python3 i11temp_yangxin.py --gpu=True --network=yolo3_mobilenet0.25_voc
    run_multi_camera() 
