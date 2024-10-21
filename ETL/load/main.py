from communication.communicate import wake_up_service, wait_for_service, log_action
import os
import pandas as pd
from sqlalchemy import create_engine, text
import json

EXTRACT_SERVICE_NAME = os.getenv('EXTRACT_SERVICE_NAME', 'extract')
SERVER_SERVICE_NAME = os.getenv('SERVER_SERVICE_NAME', 'server')
LOAD_SERVICE_NAME = os.getenv('LOAD_SERVICE_NAME', 'load')
INDICATOR = LOAD_SERVICE_NAME.upper()[0]
TRANSFORM_QUEUE = os.getenv('TRANSFORM_QUEUE', 'transform_queue')
LOADING_QUEUE = os.getenv('LOADING_QUEUE', 'loading_queue')
SERVER_QUEUE = os.getenv('SERVER_QUEUE', 'server_queue')
FAOSTAT_INDICATOR = "QCL"
CBS_INDICATOR = "CBS"

engine = create_engine(
    "postgresql://student:infomdss@database:5432/dashboard")


def load_data_to_database(ch, method, properties, body):
    '''
    This function loads data to the database.

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
        log_action(LOAD_SERVICE_NAME,
                   f"Received {active_file_name} for {active_dataset}")
        log_action(LOAD_SERVICE_NAME, f"Loading data...")
        with engine.connect() as conn:
            if active_dataset == FAOSTAT_INDICATOR:
                result = conn.execute(text("DROP TABLE IF EXISTS QCL CASCADE;"))
            elif active_dataset == CBS_INDICATOR:
                result = conn.execute(text("DROP TABLE IF EXISTS CBS CASCADE;"))

        log_action(LOAD_SERVICE_NAME, "Connecting with PostgreSQL...")
        data_frame = pd.read_csv(active_file_name, delimiter=",")
    
        if active_dataset == FAOSTAT_INDICATOR:
            data_frame.to_sql("QCL", engine, if_exists="replace", index=True)
        elif active_dataset == CBS_INDICATOR:
            data_frame.to_sql("CBS", engine, if_exists="replace", index=True)
            
        log_action(LOAD_SERVICE_NAME, "Data loaded successfully!")

        payload = json.dumps(
            {'status': "Data updated successfully", 'file_name': active_file_name, 'dataset': active_dataset})
        wake_up_service(message=payload,
                        service_name_to=SERVER_SERVICE_NAME,
                        service_name_from=LOAD_SERVICE_NAME,
                        queue_name=SERVER_QUEUE)
    except Exception as e:
        log_action(LOAD_SERVICE_NAME, f"Something went wrong: {e}")


wait_for_service(service_name=LOAD_SERVICE_NAME,
                 queue_name=LOADING_QUEUE, callback=load_data_to_database)
