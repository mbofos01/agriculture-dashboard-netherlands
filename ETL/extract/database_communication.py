from sqlalchemy import create_engine, text, inspect, Table


def get_max_feature(engine, table_name, feature):
    '''
    This function returns the maximum value of a feature in a table.

    Parameters:
    - engine: The database engine
    - table_name: The table name
    - feature: The feature to get the maximum value of

    Returns:
    - The maximum value of the feature
    '''
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(f'SELECT MAX("{feature}") FROM "{table_name}" as w'))
            return result.scalar()
    except:
        return -1


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
