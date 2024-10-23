import datetime as dt
import os
import glob
from geopy.geocoders import Nominatim
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import time
from sqlalchemy import create_engine, text
import pretty_errors

GOAL_FILE = '../data/LEGACY_final_yearly_merged_data.csv'

engine = create_engine(
    "postgresql://student:infomdss@database:5432/dashboard")


def load_data_to_database(active_file_name):
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
        print(f"Loading data...")
        with engine.connect() as conn:
            result = conn.execute(
                text('DROP TABLE IF EXISTS "Weather" CASCADE;'))

        print("Connecting with PostgreSQL...")
        data_frame = pd.read_csv(active_file_name, delimiter=",")

        data_frame.to_sql("Weather", engine, if_exists="replace", index=True)

        print("Data loaded successfully!")

    except Exception as e:
        print(f"Something went wrong: {e}")


if os.path.exists(GOAL_FILE):
    load_data_to_database(GOAL_FILE)
    exit(1)


if os.path.exists('../data/yearly_average_merged_data.csv') == False:
    # folder_path = '/data/weatherhis'
    folder_path = '../data/weatherhis'

    # List all files in the directory
    files = os.listdir(folder_path)
    file_dict = {
        'cld.txt': 'Cloud cover',
        'dtr.txt': 'Diurnal Temperature Range',
        'frs.txt': 'Frost days',
        'pet.txt': 'potential evapo-transpiration',
        'pre.txt': 'Precipitation rate',
        'tmn.txt': 'Minimum 2m temperature',
        'tmp.txt': 'Mean 2m temperature',
        'tmx.txt': 'Maximum 2m temperature',
        'vap.txt': 'Vapour pressure',
        'wet.txt': 'Wet days',
    }
    data_frames = []
    for file in files:

        # FutureWarning: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\s+'`` instead
        df = pd.read_csv(folder_path+'/'+file, sep='\s+')

        # Drop the last 5 columns
        df = df.iloc[:, :-5]

        # Melt the DataFrame to reshape
        melted_df = df.melt(
            id_vars=['YEAR'], var_name='Month', value_name=file_dict[file])

        # Create the 'Month-Year' column
        melted_df['Month-Year'] = melted_df['Month'] + \
            '-' + melted_df['YEAR'].astype(str)

        # Convert Months to a categorical attribute to be used for sorting later. If i had theme like 01,02,03 etc this step is not needed'
        melted_df['Month'] = pd.Categorical(melted_df['Month'],
                                            categories=['JAN', 'FEB', 'MAR', 'APR', 'MAY',
                                                        'JUN', 'JUL', 'AUG', 'SEP', 'OCT',
                                                        'NOV', 'DEC'],
                                            ordered=True)
        # Sort the DataFrame
        final_df = melted_df.sort_values(by=['YEAR', 'Month'])

        # Select relevant columns
        final_df_clc = final_df[['Month-Year', file_dict[file]]]

        # Reset the index if needed
        final_df_clc.reset_index(drop=True, inplace=True)
        data_frames.append(final_df_clc)

    # -------------------------------------------------------------------------------------
    merged_df = data_frames[0]
    for df in data_frames[1:]:
        # 'outer' to keep all dates, adjust to 'inner' if you only want common dates
        merged_df = pd.merge(merged_df, df, on='Month-Year', how='inner')

    # FILENAME = '/data/merged_data.csv'
    FILENAME = '../data/merged_data_dirty.csv'

    merged_df = merged_df.drop(columns="Diurnal Temperature Range")
    merged_df.to_csv(FILENAME, index=False)
    print(merged_df.columns)
    # -------------------------------------------------------------------------------------
    # FILTER FROM 1961 AND LATES AND CALCULATE THE AVERAGE
    # Extract the year from the 'Month-Year' column
    merged_df['Year'] = merged_df['Month-Year'].str.extract(
        r'(\d{4})').astype(int)

    # Filter the data to keep only records from 1961 and onwards
    df_filtered = merged_df[(merged_df['Year'] >= 1961)
                            & (merged_df['Year'] <= 2022)]

    # Drop the 'Month-Year' column as it's no longer needed
    df_filtered = df_filtered.drop(columns=['Month-Year'])

    # Define the aggregation functions for each column
    agg_funcs = {col: 'mean' for col in df_filtered.columns if col not in [
        df_filtered.columns[2], df_filtered.columns[9]]}  # Mean for all except 2 and 9
    agg_funcs[df_filtered.columns[2]] = 'sum'  # Sum for column 2
    agg_funcs[df_filtered.columns[9]] = 'sum'  # Sum for column 9

    # Group by 'Year' and apply the aggregation functions
    df_yearly_avg = df_filtered.groupby('Year').agg(agg_funcs)
    df_yearly_avg = df_yearly_avg.drop(columns=['Year'])
    # df_yearly_avg = df_yearly_avg[['Year'] + [col for col in df_yearly_avg.columns if col != 'Year']]
    print(df_yearly_avg.head())

    print(df_filtered['Year'].unique())
    # Display the first few rows of the final DataFrame

    df_yearly_avg.to_csv('../data/yearly_average_merged_data.csv', index=True)

else:
    df_yearly_avg = pd.read_csv('../data/yearly_average_merged_data.csv')

# Load the yearly_average_merged_data.csv
yearly_data = pd.read_csv('../data/yearly_average_merged_data.csv')

# Path to the folder containing the .txt files
folder_path = r'../data/weatherhis'


if os.path.exists('../data/yearly_average_merged_data_with_all_txt.csv') == False:
    # Get all .txt files in the folder
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

    # Iterate over each .txt file
    for txt_file in txt_files:
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(txt_file))[0]

        # Load the .txt file into a DataFrame
        wet_data = pd.read_csv(txt_file, sep='\s+')  # delim_whitespace=True)

        # Select the columns 'YEAR', 'MAM', 'JJA', 'SON', and 'DJF'
        if {'YEAR', 'MAM', 'JJA', 'SON', 'DJF'}.issubset(wet_data.columns):
            wet_columns = wet_data[['YEAR', 'MAM', 'JJA', 'SON', 'DJF']]
            print(yearly_data.columns)
            print(wet_columns.columns)
            # Merge with the yearly_data on 'Year'
            yearly_data = pd.merge(
                yearly_data, wet_columns, how='left', left_on='Year', right_on='YEAR')

            # Rename the new columns to include the filename (e.g., 'wet_MAM')
            yearly_data = yearly_data.rename(columns={
                'MAM': f'{filename}_MAM',
                'JJA': f'{filename}_JJA',
                'SON': f'{filename}_SON',
                'DJF': f'{filename}_DJF'
            })

            # Drop the 'YEAR' column as it's now redundant
            yearly_data = yearly_data.drop(columns=['YEAR'])
        else:
            print(f"Skipping {txt_file}, required columns not found.")

    # Save the updated yearly data to a new CSV file
    output_file = folder_path+'/yearly_average_merged_data_with_all_txt.csv'
    yearly_data.to_csv(output_file, index=False)

    print(
        f"All .txt files processed and merged. Output saved to {output_file}")

else:
    yearly_data = pd.read_csv(
        '../data/yearly_average_merged_data_with_all_txt.csv')

# --------------------------------------------------------------------------------------

if os.path.exists('../data/weather_data_with_Allcities.csv') == False:

    # Step 1: Create a geolocator object
    geolocator = Nominatim(user_agent="my_geocoder_app", timeout=10)

    # Step 2: Define the list of cities in the Netherlands
    cities = [
        "Amsterdam", "Rotterdam", "The Hague", "Utrecht", "Eindhoven",
        "Groningen", "Tilburg", "Almere", "Breda", "Nijmegen",
        "Enschede", "Haarlem", "Haarlemmermeer", "Arnhem", "Zaanstad",
        "Amersfoort", "Apeldoorn", "Dordrecht", "Leiden", "Zoetermeer",
        "Emmen", "Venlo", "Leeuwarden", "Zwolle", "Helmond"
    ]

    # Function to get weather data for a specific city

    def get_weather_data(city_name, start_date="1961-01-01", end_date="2022-12-31"):
        # Get the location
        location = geolocator.geocode(f"{city_name}, Netherlands")

        if location:
            latitude = location.latitude
            longitude = location.longitude
            print(f"{city_name}: Latitude: {latitude}, Longitude: {longitude}")

            # Setup the Open-Meteo API client with cache and retry on error
            cache_session = requests_cache.CachedSession(
                '.cache', expire_after=-1)
            retry_session = retry(
                cache_session, retries=10, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)

            # Define API parameters
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ["daylight_duration", "wind_speed_10m_max"]
            }

            # Fetch weather data
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]

            # Process daily data
            daily = response.Daily()
            daily_daylight_duration = daily.Variables(0).ValuesAsNumpy()
            daily_wind_speed_10m_max = daily.Variables(1).ValuesAsNumpy()

            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "daylight_duration": daily_daylight_duration,
                "wind_speed_10m_max": daily_wind_speed_10m_max
            }

            # Create DataFrame for the city
            daily_dataframe = pd.DataFrame(data=daily_data)
            daily_dataframe['city'] = city_name  # Add city name as a column
            return daily_dataframe
        else:
            print(f"Location for {city_name} not found.")
            return pd.DataFrame()  # Return an empty DataFrame

    # Step 3: Collect data for each city and merge DataFrames
    all_dataframes = []  # Initialize a list to hold DataFrames
    for city in cities:
        time.sleep(65)  # Add a 1-second delay between requests
        city_data = get_weather_data(city)
        if not city_data.empty:  # Check if the DataFrame is not empty
            # Append the DataFrame to the list
            all_dataframes.append(city_data)

    # Merge all DataFrames by appending rows
    merged_data = pd.concat(all_dataframes, ignore_index=True)
    merged_data.to_csv('../data/weather_data_with_Allcities.csv', index=False)
    # Display the merged DataFrame
    print(merged_data.head(5))

else:
    merged_data = pd.read_csv('../data/weather_data_with_Allcities.csv')

# --------------------------------------------------------------------------------------

merged_data['date'] = pd.to_datetime(merged_data['date'], errors='coerce')

# Drop the last column ('city')
merged_data_Y = merged_data.drop(columns=merged_data.columns[-1])

# Extract the year from the 'date' column and create a new 'Year' column
merged_data_Y['Year'] = merged_data_Y['date'].dt.year

# Drop the original 'date' column
merged_data_Y = merged_data_Y.drop(columns=['date'])

# Group by 'Year' and calculate the mean for all other columns
yearly_averages = merged_data_Y.groupby('Year').mean().reset_index()

# Display the result
# print(yearly_averages.head())

# yearly_averages.to_csv('/data/yearly_averages_open_meteo.csv', index=False)

# --------------------------------------------------------------------------------------


merged_data_norm = yearly_averages
merged_data_norm = merged_data_norm.drop(columns="daylight_duration")
# print(merged_data_norm.to_string())
print(yearly_data.columns)
print(yearly_data.head())
# merged_data_norm.to_csv('C:/Users/stili/OneDrive/Desktop/yearly_averages_open_meteo_norm.csv', index=False)
final_merged = yearly_data.merge(merged_data_norm, on='Year', how='inner')
print(final_merged.to_string())
# exit(1)
final_merged.to_csv(GOAL_FILE, index=False)


# load_data_to_database(GOAL_FILE)
# --------------------------------------------------------------------------------------
