import multiprocessing
import random
import time



class producer(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue

    def run(self):
        global my_queue_len
        for i in range(5):

            item = random.randint(0, 256)
            self.queue.put(item) 
            print ("Process Producer : 消息 %d 添加到 queue %s"\
                   % (item,self.name))
            time.sleep(1)



class consumer(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue

    def run(self):

        while True:
            if (self.queue.empty()):
                print("the queue is empty")
                break
            else :
                time.sleep(0.5)
                item = self.queue.get()

                print ('Process Consumer : 消息 %d 通知到相应人员 from by %s \n'\
                       % (item, self.name))
                time.sleep(1)

if __name__ == '__main__':
        queue = multiprocessing.Queue()
        process_producer = producer(queue)
        process_consumer = consumer(queue)

        process_producer.start()
        process_consumer.start()

        process_producer.join()
        process_consumer.join()