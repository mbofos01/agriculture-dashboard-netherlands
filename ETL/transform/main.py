import os
from communication.communicate import wake_up_service, wait_for_service, log_action
import json
import pandas as pd

EXTRACT_SERVICE_NAME = os.getenv('EXTRACT_SERVICE_NAME', 'extract')
LOAD_SERVICE_NAME = os.getenv('LOAD_SERVICE_NAME', 'load')
TRANSFORM_SERVICE_NAME = os.getenv('TRANSFORM_SERVICE_NAME', 'transform')
INDICATOR = TRANSFORM_SERVICE_NAME.upper()[0]
TRANSFORM_QUEUE = os.getenv('TRANSFORM_QUEUE', 'transform_queue')
LOADING_QUEUE = os.getenv('LOADING_QUEUE', 'loading_queue')
FAOSTAT_INDICATOR = "QCL"
CBS_INDICATOR = "CBS"


def transform_faostat(filename):
    '''
    This function transforms the FAOSTAT dataset.

    Parameters:
    - filename: The filename of the dataset

    Returns:
    - filename: The filename of the transformed dataset
    '''
    FAOSTAT = pd.read_csv(filename)

    # REMOVE ALL THE CROPS THAT HAVE MORE THAN 30% ZERO VALUES
    zero_percentage = FAOSTAT.groupby('Item')['Value'].apply(
        lambda x: (x == 0).sum() / len(x) * 100)
    items_to_keep = zero_percentage[zero_percentage <= 30].index

    # SAVE THE REDUCED CROP DATASET
    FAOSTAT = FAOSTAT[FAOSTAT['Item'].isin(items_to_keep)]
    filename = "/data/ACTIVE_FAOSTAT.csv"
    FAOSTAT.to_csv(filename, index=False)

    return filename

def transform_cbs(filename):
    # TODO: Transform CBS data
    return filename

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
        log_action(TRANSFORM_SERVICE_NAME,
                   f"Received {active_file_name} for {active_dataset}")
        # TODO: Act as transform service for weather data
        if active_dataset == FAOSTAT_INDICATOR:
            active_file_name = transform_faostat(active_file_name)
        elif active_dataset == CBS_INDICATOR:
            active_file_name = transform_cbs(active_file_name)
        # Data are transformed
        log_action(TRANSFORM_SERVICE_NAME,
                   f"Transformed {active_file_name} for {active_dataset}")

        payload = json.dumps(
            {'file_name': active_file_name, 'dataset': active_dataset})
        wake_up_service(message=payload, service_name_to=LOAD_SERVICE_NAME, service_name_from=TRANSFORM_SERVICE_NAME,
                        queue_name=LOADING_QUEUE)
    except Exception as e:
        log_action(TRANSFORM_SERVICE_NAME, f"Something went wrong! {e}")


wait_for_service(service_name=TRANSFORM_SERVICE_NAME,
                 queue_name=TRANSFORM_QUEUE, callback=notify_load_service)
