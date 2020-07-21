import multiprocessing as mp
import cv2
import time
from datetime import datetime

from gluoncv import model_zoo, data, utils
#from matplotlib import pyplot as plt
import mxnet as mx

import argparse
from imutils import paths
from imutils.video import FileVideoStream

import numpy as np
from matplotlib import pyplot as plt
import webcolors
from sklearn.cluster import KMeans
import os
import urllib
import logging


def proces_timelist(timelist, timelimit):
    timestamps = []
    for str_time in timelist:
        timestamp = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S,%f')
        print(timestamp, type(timestamp))
        timestamps.append(timestamp)
    max_time = max(timestamps)
    min_time = min(timestamps)
    interval = max_time - min_time
    # print(type(interval), '$'*20, interval.total_seconds())
    # print('interval:', interval)
    if interval.total_seconds() > timelimit:
        return True
    else:
        return False


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


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def forked_version_cv_plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, colors=None,
                 absolute_coordinates=True, scale=1.0, hx_rect=None):
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

    hx_rect: huaxin added , only check objects in this rect.
    Returns
    -------
    numpy.ndarray
        The image with detected results.

    """

    # 决定是否是异常状况，是否保存截图的信号量
    warning_signal = None


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

        # print('hx_rect=', hx_rect, '#'*10, xmin, ymin, xmax, ymax )

        # ---- 裁减检测到的情况出现在我们划定的识别区域 ----
        if hx_rect is None or (hx_rect[0] < xmin and hx_rect[1] < ymin
            and hx_rect[2] > xmax and hx_rect[3] > ymax):
        
            crop_img = img[ymin:ymax-int(int((ymax-ymin)/2)), xmin+int((xmax-xmin)/4):xmax-int((xmax-xmin)/4)]
            # crop_img = img[ymin:ymax, xmin:xmax]



            # print('crop_img.shape=', crop_img.shape)
            # mydata = np.reshape(crop_img, (-1,3))

            colorname = ''
            dominant_color = None
            # if mydata.shape[0] > 0:
            #     mydata = np.float32(mydata)

            #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            #     flags = cv2.KMEANS_RANDOM_CENTERS
            #     compactness,mylabels,centers = cv2.kmeans(mydata,1,None,criteria,10,flags)
            #     colorname = closest_colour(centers[0].astype(np.int32))
            #     dominant_color = centers[0].astype(np.int32)
            #     print('Dominant color is: bgr({})'.format(dominant_color))
            #     print('Dominant color is: bgr({})'.format(colorname))

            #reshaping to a list of pixels
            crop_img = crop_img.reshape((crop_img.shape[0] * crop_img.shape[1], 3))
            if len(crop_img) > 0:
                
                #using k-means to cluster pixels
                kmeans = KMeans(n_clusters = 1)
                kmeans.fit(crop_img)
                dominant_color = kmeans.cluster_centers_.astype(np.int32)[0]
                colorname = closest_colour(dominant_color)
                print(dominant_color, colorname)

                # differences = [[color_difference(dominant_color, target_value), target_name] for target_name, target_value in TARGET_COLORS.items()]
                # differences.sort()  # sorted by the first element of inner lists
                # my_color_name = differences[0][1]

                # print(my_color_name)
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
            # bcolor = [x * 255 for x in colors[cls_id]]
            bcolor = (255, 255, 255) 
            

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
                if colorname in ('olivedrab', 'yellow', 'sienna','goldenrod', 'gold','palegoldenrod',
                 'darkgoldenrod','greenyellow','khaki','darkkhaki','blanchedalmond', 'wheat'):               
                    # 黄色
                    bcolor = (255,255,0)
                    # 警告音 
                    # duration = 0.5  # seconds
                    # freq = 660  # Hz
                    # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                    # logger.log(logging.CRITICAL, 'yellow-hat-in-area')
                    warning_signal = 'yellow-hat-in-area'
                    # print('#'*10)

                elif colorname in ('saddlebrown', 'red', 'maroon','darkred','indianred','firebrick','brown','crimson'):
                    # 红色
                    bcolor = (255, 0, 0)
                    # 警告音 
                    # duration = 1  # seconds
                    # freq = 440  # Hz
                    # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                    # logger.log(logging.CRITICAL, 'red-hat-in-area')
                    warning_signal = 'red-hat-in-area'
                    # print('#'*20)

                # elif colorname == 'darkolivegreen':
                #     if dominant_color is not None:
                #         if (dominant_color[0] > 100 and dominant_color[1] > 70) or \
                #            (dominant_color[0] < 100 and dominant_color[2] < 50):
                #            # seem as yellow
                #             bcolor = (255,255,0)
                # elif colorname == 'darkslategray':
                #     if dominant_color is not None:
                #         if (dominant_color[0] < 65 and dominant_color[1] <= 50):
                #             # as yellow
                #             bcolor = (255,255,0)
                #         if (dominant_color[0] > 80 and dominant_color[1] >= 50):
                #             # as red
                #             bcolor = (255,0,0)                                 
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, 2)

            if class_name or score:
                y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                cv2.putText(img, '{:s} {:s}'.format(class_name, score),
                            (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(scale/2, 2),
                            bcolor, min(int(scale), 5), lineType=cv2.LINE_AA)
        else:
            print('裁减检测到的情况出现在我们划定的识别区域之外')

    return img, warning_signal