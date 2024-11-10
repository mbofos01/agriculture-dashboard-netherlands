import cbsodata
import pandas as pd
from datetime import datetime
import os
from sqlalchemy import create_engine
import database_communication as db
from communication.communicate import wake_up_service, log_action
import json

# These variables are used to fetch data
CODE = 'CBS'
DATASET = '7100ENG'
PREFIX = 'netherlands_cbs_data'
ROOT_DIR = "/data"

# These variables are used to communicate with other services
EXTRACT_SERVICE_NAME = os.getenv('EXTRACT_SERVICE_NAME', 'extract')
INDICATOR = EXTRACT_SERVICE_NAME.upper()[0]
TRANSFORM_SERVICE_NAME = os.getenv('TRANSFORM_SERVICE_NAME', 'transform')
TRANSFORM_QUEUE = os.getenv('TRANSFORM_QUEUE', 'transform_queue')


def get_latest_dataset(directory=ROOT_DIR, prefix=PREFIX):
    '''
    This function returns the latest dataset in a directory. Based on the os attached date.

    Parameters:
    - directory: The directory to search in
    - prefix: The prefix of the file to search for

    Returns:
    - The name of the latest dataset
    '''
    files = [f for f in os.listdir(directory) if f.startswith(
        prefix) and f.endswith(".csv")]

    if not files:
        return None

    latest_file = max(
        files,
        key=lambda f: os.path.getmtime(os.path.join(directory, f))
    )

    return latest_file


def get_latest_index(directory=ROOT_DIR, prefix=PREFIX):
    '''
    This function returns the latest index of a dataset in a directory.

    Parameters:
    - directory: The directory to search in
    - prefix: The prefix of the file to search for

    Returns:
    - The latest index of the dataset
    '''
    try:
        previous_data = pd.read_csv(os.path.join(
            directory, get_latest_dataset(directory, prefix)))

        return previous_data.shape[0]
    except:
        return 0


def read_fao_threshold():
    '''
    This function reads the FAOSTAT threshold from a file.

    Returns:
    - The FAOSTAT threshold
    '''
    try:
        with open('/data/faostat_end_year.txt', 'r') as file:
            return file.read()
    except:
        return None


# Download data from CBS
DATA = pd.DataFrame(cbsodata.get_data(DATASET))
DATA['Regions'] = DATA['Regions'].str.strip()

fao_threshold = read_fao_threshold()
DATA = DATA[DATA['Periods'] <= fao_threshold]

engine = create_engine(
    "postgresql://student:infomdss@database:5432/dashboard")

if db.table_exists(engine, 'CBS') == False and get_latest_dataset() is not None:
    log_action(EXTRACT_SERVICE_NAME,
               "Table does not exist, but data is available locally")
    payload = json.dumps(
        {'file_name': "/data/" + get_latest_dataset(), 'dataset': 'CBS'})
    wake_up_service(
        message=payload, service_name_from=EXTRACT_SERVICE_NAME, service_name_to=TRANSFORM_SERVICE_NAME, queue_name=TRANSFORM_QUEUE)
    exit(1)

try:
    LATEST_INDEX = get_latest_index()
except:
    LATEST_INDEX = 0

# Check if new data is available
if DATA is None or DATA.empty:
    log_action(EXTRACT_SERVICE_NAME, "No CBS data available")
elif DATA.shape[0] == LATEST_INDEX:
    log_action(EXTRACT_SERVICE_NAME, "No new CBS data available")
elif DATA.shape[0] < LATEST_INDEX:
    log_action(EXTRACT_SERVICE_NAME, "CBS data is missing")
elif DATA.shape[0] > LATEST_INDEX:
    log_action(EXTRACT_SERVICE_NAME, "New CBS data available")
    log_action(EXTRACT_SERVICE_NAME,
           f'FAOSTAT threshold is {fao_threshold} dropping CBS data after this year')
    TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
    DATA.to_csv(f'{ROOT_DIR}/{PREFIX}_{TIMESTAMP}.csv', index=False)
    payload = json.dumps(
        {'file_name': f'{ROOT_DIR}/{PREFIX}_{TIMESTAMP}.csv', 'dataset': 'CBS'})
    wake_up_service(
        message=payload, service_name_from=EXTRACT_SERVICE_NAME, service_name_to=TRANSFORM_SERVICE_NAME, queue_name=TRANSFORM_QUEUE)
else:
    log_action(EXTRACT_SERVICE_NAME,
               "Something went wrong with CBS data extraction")
