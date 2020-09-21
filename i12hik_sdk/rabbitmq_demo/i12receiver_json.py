import pika
import simplejson as json
import base64
#https://stackoverflow.com/questions/50404273/python-tutorial-code-from-rabbitmq-failing-to-run

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()


channel.queue_declare(queue='hello')
filename = 'received_test.jpg'
def callback(ch, method, properties, body):
    msg = json.loads(body)
    print(" [x] Received %r" % msg)
    imgdata = base64.b64decode(msg['img'])
    print(type(imgdata), '#'*10)
    # with open(filename, 'wb') as f:
    #     f.write(imgdata)

channel.basic_consume(on_message_callback=callback,
                      queue='hello',
                      auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()