import faostat
import pandas as pd
from datetime import datetime
import os
from sqlalchemy import create_engine, text, inspect, Table
from communication.communicate import wake_up_service

# These variables are used to fetch data
CODE = 'QCL'
NETHERLANDS = 'Netherlands (Kingdom of the)'
PREFIX = 'netherlands_qcl_data'
ROOT_DIR = "/data"
PRODUCTION_YIELD = 'Production Quantity'
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


def table_exists(engine, table_name):
    '''
    This function checks if a table exists in a database.

    Parameters:
    - engine: The database engine
    - table_name: The name of the table to check

    Returns:
    - True if the table exists, False otherwise
    '''
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def get_table_row_count(engine, table_name):
    '''
    Get the number of rows in a table.

    Parameters:
    - engine: The database engine
    - table_name: The name of the table to count rows

    Returns:
    - The number of rows in the table
    '''
    with engine.connect() as connection:
        result = connection.execute(
            text(f'SELECT COUNT(*) FROM "{table_name}"'))
        row_count = result.scalar()
    return row_count


latest = get_latest_dataset()
engine = create_engine(
    "postgresql://student:infomdss@database:5432/dashboard")

if table_exists(engine, 'QCL') == False and latest is not None:
    print(f" [{INDICATOR}] Table does not exist, but data is available locally")
    wake_up_service(
        message="/data/" + latest, service_name_from=EXTRACT_SERVICE_NAME, service_name_to=TRANSFORM_SERVICE_NAME, queue_name=TRANSFORM_QUEUE)
    exit(1)

# Define the details you want to fetch
AREA_TUPLES = faostat.get_par(CODE, 'area')
ELEMENT_TUPLES = faostat.get_par(CODE, 'element')
ITEM_TUPLES = faostat.get_par(CODE, 'item')

MY_PARAMS = {'area': AREA_TUPLES[NETHERLANDS], 'element': [
    ELEMENT_TUPLES[PRODUCTION_YIELD]], 'item': ITEM_TUPLES[CROPS_PRIMARY]}
DATA = faostat.get_data_df(CODE, pars=MY_PARAMS, strval=False)

LATEST_INDEX = get_table_row_count(engine, 'QCL')

# Check if new data is available
if DATA is None or DATA.empty:
    print(f" [{INDICATOR}] No FAOSTAT data available")
elif DATA.shape[0] == LATEST_INDEX:
    print(f" [{INDICATOR}] No new FAOSTAT data available")
elif DATA.shape[0] < LATEST_INDEX:
    print(f" [{INDICATOR}] FAOSTAT data is missing")
elif DATA.shape[0] > LATEST_INDEX:
    print(f" [{INDICATOR}] New FAOSTAT data available")
    TIMESTAMP = datetime.now().strftime('%Y_%m_%d')
    DATA.to_csv(f'{ROOT_DIR}/{PREFIX}_{TIMESTAMP}.csv', index=False)
    wake_up_service(
        message=f'{ROOT_DIR}/{PREFIX}_{TIMESTAMP}.csv', service_name_from=EXTRACT_SERVICE_NAME, service_name_to=TRANSFORM_SERVICE_NAME, queue_name=TRANSFORM_QUEUE)
else:
    print(f" [{INDICATOR}] Something went wrong with FAOSTAT data extraction")

engine.dispose()
