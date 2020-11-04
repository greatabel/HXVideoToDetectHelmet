import logging
import logging.handlers
import csv
import time

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i11qy_wechat
import i13process_frame


rtsp_file_path = '../i13rtsp_list.csv'
def load_rtsp_list():
    with open(rtsp_file_path, newline='') as f:
        reader = csv.reader(f)
        rtsp_list = list(reader)
    # print(data, '#'*10, data[0],'\n', data[0][0])
    return rtsp_list


def listener_configurer():
    # logging.getLogger("pika").propagate = False

    root = logging.getLogger()
    h = logging.handlers.RotatingFileHandler('jacket_temp.log', 'a', 3000000, 10)
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



    msg = area + ' 发生未佩戴救生衣进入区域'

    # timelimit 为在限制区域时间存在达到多少秒后，才会发送消息报警
    timelimit = 8
    # time_span_limit 代表在这个时间内只能发一次消息报警
    time_span_limit = 180
    if len(queueid_warning_dict[record.name]) >= 8:
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