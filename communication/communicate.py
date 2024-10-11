import pika
import time
TIMEOUT_SECONDS = 5


def wake_up_service(message, service_name_to, service_name_from, queue_name):
    '''
    This function sends a message to a service to wake it up.

    Parameters:
    - message: The message to be sent
    - service_name_to: The name of the service to send the message to
    - service_name_from: The name of the service sending the message
    - queue_name: The name of the queue to send the message to

    Returns:
    - None
    '''
    # Connect to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    # Declare a queue
    channel.queue_declare(queue=queue_name)

    channel.basic_publish(
        exchange='', routing_key=queue_name, body=message)
    try:
        service_indicator = service_name_from.upper()[0]
    except:
        service_indicator = "X"

    print(f" [{service_indicator}] Sent '{message}' to {service_name_to}")


def default_callback(ch, method, properties, body):
    '''
    Default callback function for receiving messages.
    '''
    print(f" [x] Received {body.decode()}")


def wait_for_service(service_name, queue_name, callback=None):
    '''
    This function waits for a message from a service.

    Parameters:
    - service_name: The name of the service waiting for the message
    - queue_name: The name of the queue to wait for the message
    - callback: The function to call when the message is received

    Returns:
    - None
    '''
    while True:
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters('rabbitmq'))
            channel = connection.channel()
            channel.queue_declare(queue=queue_name)
            break  # If connection is successful, exit the loop
        except pika.exceptions.AMQPConnectionError:
            print(
                f"RabbitMQ not available yet, retrying in {TIMEOUT_SECONDS} seconds...")
            time.sleep(TIMEOUT_SECONDS)

    channel.basic_consume(
        queue=queue_name, on_message_callback=callback, auto_ack=True)

    try:
        service_indicator = service_name.upper()[0]
    except:
        service_indicator = "X"

    print(f" [{service_indicator}] Waiting on {queue_name}. To exit press CTRL+C")
    channel.start_consuming()
