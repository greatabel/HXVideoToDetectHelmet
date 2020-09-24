import pika
import json

#https://stackoverflow.com/questions/50404273/python-tutorial-code-from-rabbitmq-failing-to-run

import numpy as np
import cv2
import base64 

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()


channel.queue_declare(
        queue='hello',
        arguments={'x-max-length': 5, "x-queue-mode": "lazy"},
        )



# def callback(ch, method, properties, body):
#     msg = json.loads(body)
#     numpy_data = numpy.asarray(msg["img"])
#     # print(" [x] Received %r" % msg)
#     # imgdata = base64.b64decode(msg['img'])
#     print(msg['placeid'], '@'*10, msg['time'])
#     print(type(numpy_data), '#'*10, numpy_data)
#     cv2.imwrite("filename.png", numpy_data)


    # cv2.namedWindow('f', flags=cv2.WINDOW_NORMAL)
    # cv2.imshow('f',numpy_data)
    # cv2.imshow('received_test', numpy_data)
    # with open(filename, 'wb') as f:
    #     f.write(imgdata)

# channel.basic_consume(on_message_callback=callback,
#                       queue='hello',
#                       auto_ack=True)
def my_callback_with_extended_args(ch, method, properties, body, host):
    print('#'*20)
    print(ch, method, properties, host)
    msg = json.loads(body)
    # numpy_data = numpy.asarray(msg["img"])
    # get image bytes string
    img = base64.b64decode(msg['img'].encode())
    print(msg, 'img=', img)
    # get image array
    img_opencv = cv2.imdecode(np.fromstring(img, np.uint8), 1)
    print('img_opencv=', img_opencv, type(img_opencv))
    cv2.imwrite("filename.png", img_opencv)
    # channel.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(on_message_callback=lambda ch, method, properties, body: my_callback_with_extended_args(ch, method, properties, body, host="abel"),
                      queue='hello',
                      auto_ack=False)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()