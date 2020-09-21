import pika
import json
from json import JSONEncoder
import base64
import numpy


connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')

numpyArrayOne = numpy.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

msg = {
    'placeid': 2,
    'time': '2020-09-21:10:58:59',
    'ai_type': 'hat_detection',
    'img': numpyArrayOne

}
print(type(msg), '@'*10)
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body=json.dumps(msg, cls=NumpyArrayEncoder)
                      )
print(" [x] Sent msg'")
connection.close()