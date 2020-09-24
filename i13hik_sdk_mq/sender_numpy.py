import pika
import json
from json import JSONEncoder
import base64
import numpy as np
import cv2
import time
import os 


while True:
  connection = pika.BlockingConnection(pika.ConnectionParameters(
          host='localhost'))
  channel = connection.channel()

  channel.queue_declare(
          queue='hello',
          arguments={'x-max-length': 5, "x-queue-mode": "lazy"},

          )


  # numpyArrayOne = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])

  image_name = os.path.join(os.getcwd(),'test0.jpg')
  
  img = cv2.imread(image_name)
  _, img_encode = cv2.imencode('.jpg', img)
  np_data = np.array(img_encode)
  str_data = np_data.tostring()
  b64_bytes = base64.b64encode(str_data)

  picData_string = b64_bytes.decode()
  print('picData_string', picData_string)
  # class NumpyArrayEncoder(JSONEncoder):
  #     def default(self, obj):
  #         if isinstance(obj, numpy.ndarray):
  #             return obj.tolist()
  #         return JSONEncoder.default(self, obj)

  msg = {
      'placeid': 2,
      'time': '2020-09-21:10:58:59',
      'ai_type': 'hat_detection',
      'img': picData_string,

  }
  json1 = json.dumps(msg)
  # print(type(msg), '@'*10)
  import sys
  s = sys.getsizeof(msg)
  print(s, s/1024, s/1048576)
  s = sys.getsizeof(json1)
  print(s, s/1024, s/1048576)
  channel.basic_publish(exchange='',
                        routing_key='hello',
                        body=json1
                        )
  time.sleep(0.5)
  print(" [x] Sent msg'")

  connection.close()