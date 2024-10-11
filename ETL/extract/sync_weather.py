import pandas as pd
from sqlalchemy import create_engine, text, inspect, Table


def _load_data_to_db(filename):
    print("-----------------------------------------------------------------")

    engine = create_engine(
        "postgresql://student:infomdss@database:5432/dashboard")

    print("Establishing connection to DB")
    with engine.connect() as conn:
        result = conn.execute(text("DROP TABLE IF EXISTS Weather CASCADE;"))

    print("Loading weather data to DB")
    filename = "/data/" + filename
    data_frame = pd.read_csv(filename, delimiter=",")

    data_frame.to_sql("Weather", engine, if_exists="replace", index=True)
    print("Data weather loaded to DB")
    print("-----------------------------------------------------------------")
    engine.dispose()


_load_data_to_db("merged_data.csv")
