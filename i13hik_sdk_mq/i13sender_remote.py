import pika


host = '10.248.68.249'
credentials = pika.PlainCredentials('test', 'test')
parameters = pika.ConnectionParameters(host,
                                   5672,
                                   '/',
                                   credentials)

connection = pika.BlockingConnection(parameters)

channel = connection.channel()

channel.queue_declare(queue='hello0')

channel.basic_publish(exchange='',
                  routing_key='hello0',
                  body='Hello W0rld from macbook pro#!')
print(" [x] Sent 'Hello World!'")
connection.close()