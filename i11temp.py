import cv2

import time
import multiprocessing as mp

def image_put(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    if cap.isOpened():
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (name, pwd, ip, channel))
        print('DaHua')

    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)

def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

def run_multi_camera():
    # user_name, user_pwd = "admin", "password"

    camera_ip_l = [
        ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        ["admin", "admin123", "10.248.10.100:554", 3],
        ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        ["admin", "admin123", "10.248.10.100:554", 3],
        ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],

        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],

        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],

        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],

        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],
        # ["admin", "admin123", "10.248.10.100:554", 1],  # ipv4
        # ["admin", "admin123", "10.248.10.100:554", 3],


        # 把你的摄像头的地址放到这里，如果是ipv6，那么需要加一个中括号。
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]
    processes = []



    
    for queue, camera_ip in zip(queues, camera_ip_l):
        print(camera_ip, camera_ip[0], '#', camera_ip[1])
        processes.append(mp.Process(target=image_put, args=(queue, camera_ip[0], camera_ip[1], camera_ip[2], camera_ip[3])))
        # processes.append(mp.Process(target=image_get, args=(queue, camera_ip[2])))



    # -------------------- start ai processes

    for i in range(0, 2):
        print('ai process', i)
        processes.append(mp.Process(target=image_get, args=(queues[i], str(i))))
    # -------------------- end   ai processes

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()



if __name__ == '__main__':
    print('here')
    run_multi_camera() 
