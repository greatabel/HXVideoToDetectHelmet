import pika
import json
from json import JSONEncoder
import numpy
import time
import cv2


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def sender(host, img, queueid=None):
	print(type(img), 'in sender')
	connection = pika.BlockingConnection(pika.ConnectionParameters(
	        host=host))
	channel = connection.channel()

	channel.queue_declare(queue='hello')

	# myfile = 'test0.jpg'
	# img = None
	# with open(myfile, "rb") as image:
	#     img = base64.b64encode(image.read())
	#     print(type(img), '#'*10)
	now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
	msg = {
	    'placeid': queueid,
	    'time': now,
	    'img': img

	}
	print(type(msg), '@'*10)
	channel.basic_publish(exchange='',
	                      routing_key='hello',
	                      body=json.dumps(msg, cls=NumpyArrayEncoder) 
	                      )
	print(" [x] Sent msg'")
	connection.close()




# def my_callback_with_extended_args(ch, method, properties, body, host, log_queue):
#     print('#'*20)
#     print(type(log_queue))
#     print(ch, method, properties, host)
#     msg = json.loads(body)
#     numpy_data = numpy.asarray(msg["img"])
#     # print(" [x] Received %r" % msg)
#     # imgdata = base64.b64decode(msg['img'])
#     print(msg['placeid'], '@'*10, msg['time'])
#     print(type(numpy_data), '#'*10)
#     cv2.imwrite("filename.png", numpy_data)


if __name__ == "__main__":
	numpyArrayOne = numpy.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
	sender('localhost', numpyArrayOne)