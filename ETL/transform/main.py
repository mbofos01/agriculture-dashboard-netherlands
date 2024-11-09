import os
from communication.communicate import wake_up_service, wait_for_service, log_action
import json
import pandas as pd
from weather_transformation import transform_daily_data, transform_hourly_data

EXTRACT_SERVICE_NAME = os.getenv('EXTRACT_SERVICE_NAME', 'extract')
LOAD_SERVICE_NAME = os.getenv('LOAD_SERVICE_NAME', 'load')
TRANSFORM_SERVICE_NAME = os.getenv('TRANSFORM_SERVICE_NAME', 'transform')
INDICATOR = TRANSFORM_SERVICE_NAME.upper()[0]
TRANSFORM_QUEUE = os.getenv('TRANSFORM_QUEUE', 'transform_queue')
LOADING_QUEUE = os.getenv('LOADING_QUEUE', 'loading_queue')
FAOSTAT_INDICATOR = "QCL"
CBS_INDICATOR = "CBS"
WEATHER_INDICATOR = "Weather"


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
    '''
    CBS data do not need any transformation.

    Parameters:
    - filename: The filename of the dataset

    Returns:
    - filename: The filename of the transformed dataset
    '''
    return filename


def transform_weather_data(filename_daily, filename_hourly):
    '''
    This function transforms the weather dataset.

    Parameters:
    - filename_daily: The filename of the daily dataset
    - filename_hourly: The filename of the hourly dataset

    Returns:
    - filename_annual: The filename of the annual dataset
    - filename_monthly: The filename of the monthly dataset
    '''
    daily = pd.read_csv(filename_daily)
    hourly = pd.read_csv(filename_hourly)

    monthly = transform_daily_data(daily)
    final_annual, final_monthly = transform_hourly_data(hourly, monthly)

    final_annual.to_csv("/data/ACTIVE_WEATHER_ANNUAL.csv", index=False)
    final_monthly.to_csv("/data/ACTIVE_WEATHER_MONTHLY.csv", index=False)

    return "/data/ACTIVE_WEATHER_ANNUAL.csv", "/data/ACTIVE_WEATHER_MONTHLY.csv"


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
        active_dataset = payload["dataset"]
        if active_dataset == WEATHER_INDICATOR:
            active_file_name_daily = payload["file_name_daily"]
            active_file_name_hourly = payload["file_name_hourly"]
            log_action(TRANSFORM_SERVICE_NAME,
                       f"Received {active_file_name_daily} and {active_file_name_hourly} for {active_dataset}")
        else:
            active_file_name = payload["file_name"]
            log_action(TRANSFORM_SERVICE_NAME,
                       f"Received {active_file_name} for {active_dataset}")
        if active_dataset == FAOSTAT_INDICATOR:
            active_file_name = transform_faostat(active_file_name)
        elif active_dataset == CBS_INDICATOR:
            active_file_name = transform_cbs(active_file_name)
        elif active_dataset == WEATHER_INDICATOR:
            active_file_name, final_dataset_monthly = transform_weather_data(
                active_file_name_daily, active_file_name_hourly)
        # Data are transformed
        log_action(TRANSFORM_SERVICE_NAME,
                   f"Transformed {active_file_name} for {active_dataset}")

        if active_dataset == WEATHER_INDICATOR:
            payload = json.dumps(
                {'file_name': active_file_name, 'dataset': f'{active_dataset}-ANNUAL'})
            wake_up_service(message=payload, service_name_to=LOAD_SERVICE_NAME, service_name_from=TRANSFORM_SERVICE_NAME,
                            queue_name=LOADING_QUEUE)

            payload = json.dumps(
                {'file_name': final_dataset_monthly, 'dataset': f'{active_dataset}-MONTHLY'})

        else:
            payload = json.dumps(
                {'file_name': active_file_name, 'dataset': active_dataset})
            wake_up_service(message=payload, service_name_to=LOAD_SERVICE_NAME, service_name_from=TRANSFORM_SERVICE_NAME,
                            queue_name=LOADING_QUEUE)
    except Exception as e:
        log_action(TRANSFORM_SERVICE_NAME, f"Something went wrong! {e}")


# Wait for the extract service - Start the process
wait_for_service(service_name=TRANSFORM_SERVICE_NAME,
                 queue_name=TRANSFORM_QUEUE, callback=notify_load_service)
