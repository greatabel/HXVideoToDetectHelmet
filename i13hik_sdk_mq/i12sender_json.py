import pika
import simplejson as json
import base64


connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')

myfile = 'test0.jpg'
img = None
with open(myfile, "rb") as image:
    img = base64.b64encode(image.read())
    print(type(img), '#'*10)
msg = {
    'placeid': 2,
    'time': '2020-09-21:10:58:59',
    'ai_type': 'hat_detection',
    'img': img

}
print(type(msg), '@'*10)
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body=json.dumps(msg)
                      )
print(" [x] Sent msg'")
connection.close()