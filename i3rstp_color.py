import multiprocessing as mp
import cv2
import time


from gluoncv import model_zoo, data, utils
#from matplotlib import pyplot as plt
import mxnet as mx
import cv2
import argparse
from imutils import paths
from imutils.video import FileVideoStream

import numpy as np
from matplotlib import pyplot as plt
import webcolors


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_voc',
                        #use yolo3_darknet53_voc, yolo3_mobilenet1.0_voc, yolo3_mobilenet0.25_voc 
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--short', type=int, default=416,
                        help='Input data shape for evaluation, use 320, 416, 512, 608, '                  
                        'larger size for dense object and big size input')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='confidence threshold for object detection')

    parser.add_argument('--gpu', type=bool, default=False,
                        help='use gpu or cpu.')
    
    args = parser.parse_args()
    return args


def image_put(q, user, pwd, ip, channel=3):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel))
    if cap.isOpened():
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))
        print('DaHua')

    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


# def render_as_image(a):
#     img = a.asnumpy() # convert to numpy array
#     img = img.transpose((1, 2, 0))  # Move channel to the last dimension
#     img = img.astype(np.uint8)  # use uint8 (0-255)

#     plt.imshow(img)
#     # plt.show()
#     import random
#     int_name = str(random.randint(1,10))
#     plt.savefig(int_name + '.png')

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def origin_cv_plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, colors=None,
                 absolute_coordinates=True, scale=1.0):

    

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if len(bboxes) < 1:
        return img

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height
    else:
        bboxes *= scale


    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        bcolor = [x * 255 for x in colors[cls_id]]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, 2)

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''
        if class_name or score:
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(img, '{:s} {:s}'.format(class_name, score),
                        (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(scale/2, 2),
                        bcolor, min(int(scale), 5), lineType=cv2.LINE_AA)

    return img

#  fork from : https://gluon-cv.mxnet.io/_modules/gluoncv/utils/viz/bbox.html
def forked_version_cv_plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, colors=None,
                 absolute_coordinates=True, scale=1.0):
    """Visualize bounding boxes with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).
    scale : float
        The scale of output image, which may affect the positions of boxes

    Returns
    -------
    numpy.ndarray
        The image with detected results.

    """
    


    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if len(bboxes) < 1:
        return img

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height
    else:
        bboxes *= scale


    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
                # print(cls_id, ' @-> ', colors[cls_id])
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
                # print(cls_id, '-> ', colors[cls_id])
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        # ---- 裁减到识别区域
        
        crop_img = img[ymin:ymax, xmin+int((xmax-xmin)/2):xmax]


        mydata = np.reshape(crop_img, (-1,3))
        # print(data.shape)
        colorname = ''
        dominant_color = None
        if mydata.shape[0] > 0:
            mydata = np.float32(mydata)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv2.KMEANS_RANDOM_CENTERS
            compactness,mylabels,centers = cv2.kmeans(mydata,1,None,criteria,10,flags)
            colorname = closest_colour(centers[0].astype(np.int32))
            dominant_color = centers[0].astype(np.int32)
            print('Dominant color is: bgr({})'.format(dominant_color))
            print('Dominant color is: bgr({})'.format(colorname))

            # 因为centers传出的是bgr，需要改变顺序为 rgb
            # t = cv2.cvtColor(centers[0], cv2.COLOR_BGR2RGB).astype(np.int32)[0][0]
            # print('# color is: bgr({})'.format(t))
            # colorname = closest_colour(centers[0].astype(np.int32))

            # print('Dominant color is: bgr({})'.format(colorname))

        # from PIL import Image
        # new_img = Image.fromarray(crop_img, 'RGB')

        # plt.imshow(crop_img)
        # # plt.show()
        # import random
        # int_name = str(random.randint(1,10))
        # plt.savefig(int_name + '.png')
        # 识别颜色
        # from colorthief import ColorThief
        # color_thief = ColorThief(new_img)
        # # get the dominant color
        # dominant_color = color_thief.get_color(quality=1)
        # print(dominant_color, type(dominant_color))

        # # ----
        bcolor = [x * 255 for x in colors[cls_id]]
        

        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, 2)

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''

        if class_name == 'person':
            #天蓝色
            bcolor = (12, 203, 232)
        elif class_name == 'hat':
            if colorname in ('olivedrab', 'yellow', 'sienna'):               
                # 黄色
                bcolor = (255,255,0)
            elif colorname in ('saddlebrown', 'red'):
                # 红色
                bcolor = (255, 0, 0)
            elif colorname == 'darkolivegreen':
                if dominant_color is not None:
                    if (dominant_color[0] > 100 and dominant_color[1] > 70) or \
                       (dominant_color[0] < 100 and dominant_color[2] < 50):
                        bcolor = bcolor = (255,255,0)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, 2)

        if class_name or score:
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(img, '{:s} {:s}'.format(class_name, score),
                        (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(scale/2, 2),
                        bcolor, min(int(scale), 5), lineType=cv2.LINE_AA)

    return img

def image_get(q, window_name):
    frame_index = 0


    args = parse_args()
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

    # cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        # cv2.imshow(window_name, frame)
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
        count = 0

            



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
        x = forked_version_cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], 
                                        class_names=net.classes,thresh=args.threshold)
        # x = origin_cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], 
        #                                 class_names=net.classes,thresh=args.threshold)

        # print('#'*10, type(bboxes), bboxes.shape)
        # 让窗口可以调整
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow('image', orig_img[...,::-1])

        # count += 8
        # cap.set(1, count)

    # cv2.waitKey(0)

    # cv2.imwrite(frame.split('.')[0] + '_result.jpg', orig_img[...,::-1])

    # cv2.destroyAllWindows()

    # frame = imutils.resize(frame, width=min(800, frame.shape[1]))
    # (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.15)
    # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
    # for (x, y, w, h) in pick:
    #     cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
    #     newpath = os.path.join('myimages/' , str(frame_index) + ".jpg")
    #     cv2.imwrite(newpath,frame[y:h,x:w])

    # frame_index = frame_index + 1
    # print('#'*20, frame_index)

    # cv2.imshow("frame",frame)

        if cv2.waitKey(1) == 27:
                break



def run_single_camera():
    user_name, user_pwd, camera_ip = "admin", "admin123", "10.248.10.133:554"

    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=2)
    processes = [mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)),
                 mp.Process(target=image_get, args=(queue, camera_ip))]

    [process.start() for process in processes]
    [process.join() for process in processes]



if __name__ == '__main__':
    run_single_camera()

    pass
