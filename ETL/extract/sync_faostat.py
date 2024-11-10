import faostat
import pandas as pd
from datetime import datetime
import os
from sqlalchemy import create_engine
import database_communication as db
from communication.communicate import wake_up_service, log_action
import json
from weather_extraction import get_daily_data, get_hourly_data, transform_date

# These variables are used to fetch data
CODE = 'QCL'
NETHERLANDS = 'Netherlands (Kingdom of the)'
PREFIX = 'netherlands_qcl_data'
ROOT_DIR = "/data"
PRODUCTION_QUANTITY = 'Production Quantity'
CROPS_PRIMARY = 'Crops, primary > (List)'

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


def save_faostat_end_year(year):
    '''
    This function saves the end year of the FAOSTAT data in a file.
    
    Parameters:
    - year: The end year of the FAOSTAT data
    
    Returns:
    - None
    '''
    with open('/data/faostat_end_year.txt', 'w') as f:
        f.write(str(year))
    

# Define the details you want to fetch
AREA_TUPLES = faostat.get_par(CODE, 'area')
ELEMENT_TUPLES = faostat.get_par(CODE, 'element')
ITEM_TUPLES = faostat.get_par(CODE, 'item')

MY_PARAMS = {'area': AREA_TUPLES[NETHERLANDS], 'element': [
    ELEMENT_TUPLES[PRODUCTION_QUANTITY]], 'item': ITEM_TUPLES[CROPS_PRIMARY]}
DATA = faostat.get_data_df(CODE, pars=MY_PARAMS, strval=False)

engine = create_engine(
    "postgresql://student:infomdss@database:5432/dashboard")

if db.table_exists(engine, 'QCL') == False and get_latest_dataset() is not None:
    log_action(EXTRACT_SERVICE_NAME,
               "Table does not exist, but data is available locally")
    payload = json.dumps(
        {'file_name': "/data/" + get_latest_dataset(), 'dataset': 'QCL'})

    save_faostat_end_year(DATA['Year'].max())
    wake_up_service(
        message=payload, service_name_from=EXTRACT_SERVICE_NAME, service_name_to=TRANSFORM_SERVICE_NAME, queue_name=TRANSFORM_QUEUE)
    exit(1)

try:
    LATEST_INDEX = get_latest_index()
except:
    LATEST_INDEX = 0

# Check if new data is available
if DATA is None or DATA.empty:
    log_action(EXTRACT_SERVICE_NAME, "No FAOSTAT data available")
elif DATA.shape[0] == LATEST_INDEX:
    log_action(EXTRACT_SERVICE_NAME, "No new FAOSTAT data available")
elif DATA.shape[0] < LATEST_INDEX:
    log_action(EXTRACT_SERVICE_NAME, "FAOSTAT data is missing")
elif DATA.shape[0] > LATEST_INDEX:
    log_action(EXTRACT_SERVICE_NAME, "New FAOSTAT data available")
    TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
    DATA.to_csv(f'{ROOT_DIR}/{PREFIX}_{TIMESTAMP}.csv', index=False)
    save_faostat_end_year(DATA['Year'].max())
    payload = json.dumps(
        {'file_name': f'{ROOT_DIR}/{PREFIX}_{TIMESTAMP}.csv', 'dataset': 'QCL'})
    wake_up_service(
        message=payload, service_name_from=EXTRACT_SERVICE_NAME, service_name_to=TRANSFORM_SERVICE_NAME, queue_name=TRANSFORM_QUEUE)
    # DONE: find last date of weather data
    last_year_on_db = int(db.get_max_feature(engine, 'Weather', 'Year'))
    log_action(EXTRACT_SERVICE_NAME, f"Last year on db: {last_year_on_db}")
    # DONE: find last date of the new FAOSTAT data
    last_year_on_data = int(DATA['Year'].max())
    log_action(EXTRACT_SERVICE_NAME,
               f"Last year on incoming data: {last_year_on_data}")
    # DONE: fetch open-meteo data for these dates
    if last_year_on_db < last_year_on_data:
        start_date, end_date = transform_date(
            last_year_on_db, last_year_on_data)
        log_action(EXTRACT_SERVICE_NAME,
                   f"New weather data available - fetching data for [{start_date} - {end_date}] from Open-Meteo")
        daily = get_daily_data(start_date, end_date)
        hourly = get_hourly_data(start_date, end_date)
        # DONE: wake up transform service
        daily.to_csv(f'{ROOT_DIR}/daily_weather_{TIMESTAMP}.csv', index=False)
        hourly.to_csv(
            f'{ROOT_DIR}/hourly_weather_{TIMESTAMP}.csv', index=False)
        payload = json.dumps(
            {'file_name_daily': f'{ROOT_DIR}/daily_weather_{TIMESTAMP}.csv',
             'file_name_hourly': f'{ROOT_DIR}/hourly_weather_{TIMESTAMP}.csv',
             'dataset': 'Weather'})
        wake_up_service(message=payload,
                        service_name_from=EXTRACT_SERVICE_NAME, service_name_to=TRANSFORM_SERVICE_NAME, queue_name=TRANSFORM_QUEUE)


else:
    log_action(EXTRACT_SERVICE_NAME,
               "Something went wrong with FAOSTAT data extraction")
