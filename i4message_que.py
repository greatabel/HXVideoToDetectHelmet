import multiprocessing
import random
import time



class producer(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue

    def run(self):
        global my_queue_len
        for i in range(15):

            item = random.randint(0, 256)
            itemObj = { 'name': 'abel' + str(item), 'id': item}
            self.queue.put(itemObj) 
            print ("模拟摄像头检测 Process Producer : 消息 %d 添加到 queue %s"\
                   % (item,self.name))
            time.sleep(1)



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

                print ('Process调用程序 : 消息%s # %d 通知到相应人员 from by %s \n'\
                       % (item['name'], item['id'], self.name))
                time.sleep(1)

if __name__ == '__main__':
        queue = multiprocessing.Queue()
        process_producer = producer(queue)
        process_consumer = consumer(queue)

        process_producer.start()
        time.sleep(1)
        process_consumer.start()

        process_producer.join()
        process_consumer.join()