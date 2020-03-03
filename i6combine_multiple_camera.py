import multiprocessing
import random
import time
import cv2
import numpy as np
import urllib.request


pre_video_url = 'http://0.0.0.0:5000/video_feed'

def show_video(video_url):
    stream = urllib.request.urlopen(video_url)
    total_bytes = b''
    while True:
        total_bytes += stream.read(1024)
        b = total_bytes.find(b'\xff\xd9') # JPEG end
        if not b == -1:
            a = total_bytes.find(b'\xff\xd8') # JPEG start
            jpg = total_bytes[a:b+2] # actual image
            total_bytes= total_bytes[b+2:] # other informations
            
            # decode to colored image ( another option is cv2.IMREAD_GRAYSCALE )
            img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR) 
            # print(type(img))
            # cv2.imshow('Window name',cv2.flip(img, -1)) # display image while receiving data
            cv2.imshow('frame',img)
            if cv2.waitKey(1) ==27: # if user hit esc            
                break
    cv2.destroyWindow()

class producer(multiprocessing.Process):
    def __init__(self, queue, number):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.number = number

    def run(self):
        global my_queue_len
        print(self.number, ' run')
        # for i in range(3):
        #     url = pre_video_url + str(i)


        #     item = random.randint(0, 256)
        #     self.queue.put(item) 
        #     print ("模拟摄像头检测 Process Producer : 消息 %d 添加到 queue %s"\
        #            % (item,self.name))
        #     time.sleep(1)



class consumer(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue

    def run(self):

        while True:
            if (self.queue.empty()):
                print("队列已空")
                break
            else :
                time.sleep(0.5)
                item = self.queue.get()

                print ('Process调用程序 : 消息 %d 通知到相应人员 from by %s \n'\
                       % (item, self.name))
                time.sleep(1)

if __name__ == '__main__':
        queue = multiprocessing.Queue()

        producers = []
        for i in range(3):
            print('#',i)
            producers.append(producer(queue,i))
        # process_producer = producer(queue, 0)
        process_consumer = consumer(queue)
        for p in producers:
            p.start()
        # process_producer.start()
        process_consumer.start()
        for p in producers:
            p.join()
        # process_producer.join()
        process_consumer.join()