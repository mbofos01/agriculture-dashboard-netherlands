from communication.communicate import wake_up_service, wait_for_service
import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect, Table

EXTRACT_SERVICE_NAME = os.getenv('EXTRACT_SERVICE_NAME', 'extract')
LOAD_SERVICE_NAME = os.getenv('LOAD_SERVICE_NAME', 'load')
INDICATOR = LOAD_SERVICE_NAME.upper()[0]
TRANSFORM_QUEUE = os.getenv('TRANSFORM_QUEUE', 'transform_queue')
LOADING_QUEUE = os.getenv('LOADING_QUEUE', 'loading_queue')

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
        active_file_name = body.decode()
        print(f" [{INDICATOR}] Received {active_file_name}")
        print(f" [{INDICATOR}] Loading data...")
        with engine.connect() as conn:
            result = conn.execute(text("DROP TABLE IF EXISTS QCL CASCADE;"))

        print(f" [{INDICATOR}] Connecting with PostgreSQL...")
        data_frame = pd.read_csv(active_file_name, delimiter=",")

        data_frame.to_sql("QCL", engine, if_exists="replace", index=True)
        print(f" [{INDICATOR}] Data Loaded Succesfully")
    except Exception as e:
        print(f" [{INDICATOR}] Something went wrong! {e}")


wait_for_service(service_name=LOAD_SERVICE_NAME,
                 queue_name=LOADING_QUEUE, callback=load_data_to_database)
