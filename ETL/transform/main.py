import os
from communication.communicate import wake_up_service, wait_for_service
import json

EXTRACT_SERVICE_NAME = os.getenv('EXTRACT_SERVICE_NAME', 'extract')
LOAD_SERVICE_NAME = os.getenv('LOAD_SERVICE_NAME', 'load')
TRANSFORM_SERVICE_NAME = os.getenv('TRANSFORM_SERVICE_NAME', 'transform')
INDICATOR = TRANSFORM_SERVICE_NAME.upper()[0]
TRANSFORM_QUEUE = os.getenv('TRANSFORM_QUEUE', 'transform_queue')
LOADING_QUEUE = os.getenv('LOADING_QUEUE', 'loading_queue')


def notify_load_service(ch, method, properties, body):
    '''
    This function transforms incoming data and notifies the load service.

    Parameters:
    - ch: The channel
    - method: The method
    - properties: The properties
    - body: The body

    Returns:
    - None
    '''

    try:
        payload = json.loads(body)
        active_file_name = payload["file_name"]
        active_dataset = payload["dataset"]
        print(f" [{INDICATOR}] Received {active_file_name}")
        # print("Transforming data...")
        # TODO: Act as transform service
        # Data are transformed
        print(f" [{INDICATOR}] Data transformed successfully")

        payload = json.dumps(
            {'file_name': active_file_name, 'dataset': active_dataset})
        wake_up_service(message=payload, service_name_to=LOAD_SERVICE_NAME, service_name_from=TRANSFORM_SERVICE_NAME,
                        queue_name=LOADING_QUEUE)
    except Exception as e:
        print(f" [{INDICATOR}] Something went wrong! {e}")


wait_for_service(service_name=TRANSFORM_SERVICE_NAME,
                 queue_name=TRANSFORM_QUEUE, callback=notify_load_service)
