import pika

#https://stackoverflow.com/questions/50404273/python-tutorial-code-from-rabbitmq-failing-to-run

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()


channel.queue_declare(queue='hello0')

def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)

channel.basic_consume(on_message_callback=callback,
                      queue='hello0',
                      auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()