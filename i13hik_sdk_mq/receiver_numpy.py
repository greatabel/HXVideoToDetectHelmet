import pika
import json

#https://stackoverflow.com/questions/50404273/python-tutorial-code-from-rabbitmq-failing-to-run

import numpy
import cv2

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()


channel.queue_declare(queue='hello')

def callback(ch, method, properties, body):
    msg = json.loads(body)
    numpy_data = numpy.asarray(msg["img"])
    # print(" [x] Received %r" % msg)
    # imgdata = base64.b64decode(msg['img'])
    print(type(numpy_data), '#'*10, numpy_data)
    cv2.imwrite("filename.png", numpy_data)


    # cv2.namedWindow('f', flags=cv2.WINDOW_NORMAL)
    # cv2.imshow('f',numpy_data)
    # cv2.imshow('received_test', numpy_data)
    # with open(filename, 'wb') as f:
    #     f.write(imgdata)

channel.basic_consume(on_message_callback=callback,
                      queue='hello',
                      auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()