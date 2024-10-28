from datetime import datetime
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import time
from geopy.geocoders import Nominatim


##########################################################################################
# Extract the weather data
##########################################################################################

ROOT_DIR = "data"

# Step 1: Create a geolocator object
geolocator = Nominatim(user_agent="my_geocoder_app", timeout=10)

# Step 2: Define the list of cities in the Netherlands
# cities = ["Utrecht", "Rotterdam"]
cities = ["Amsterdam", "Rotterdam", "The Hague", "Utrecht", "Eindhoven",
          "Groningen", "Tilburg", "Almere", "Breda", "Nijmegen",
          "Enschede", "Haarlem", "Haarlemmermeer", "Arnhem", "Zaanstad",
          "Amersfoort", "Apeldoorn", "Dordrecht", "Leiden", "Zoetermeer",
          "Emmen", "Venlo", "Leeuwarden", "Zwolle", "Helmond"
          ]


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
            "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean", "sunrise", "sunset", "daylight_duration",
                      "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
        }

        # Fetch weather data
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Process daily data
        daily = response.Daily()
        # daily_daylight_duration = daily.Variables(0).ValuesAsNumpy()
        # daily_wind_speed_10m_max = daily.Variables(1).ValuesAsNumpy()

        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            # "daylight_duration": daily_daylight_duration,
            # "wind_speed_10m_max": daily_wind_speed_10m_max
        }

        # Iterate over the fields and extract their values
        for i, field in enumerate(params["daily"]):
            daily_data[field] = daily.Variables(i).ValuesAsNumpy()

        # Convert the dictionary to a DataFrame
        daily_df = pd.DataFrame(daily_data)

        # print(daily_df.head())

        # Create DataFrame for the city
        daily_dataframe = pd.DataFrame(data=daily_data)
        daily_dataframe['city'] = city_name  # Add city name as a column
        return daily_dataframe
    else:
        print(f"Location for {city_name} not found.")
        return pd.DataFrame()  # Return an empty DataFrame


def set_date_range(start_year="1961", end_year="2022"):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    return start_date, end_date


def sync_weather_data(start_year="2022", end_year="2023"):
    start_date, end_date = set_date_range(start_year, end_year)
    # Step 3: Collect data for each city and merge DataFrames
    all_dataframes = []  # Initialize a list to hold DataFrames
    for city in cities:
        city_data = get_weather_data(
            city, start_date=start_date, end_date=end_date)
        # print(city_data)
        if not city_data.empty:  # Check if the DataFrame is not empty
            # Append the DataFrame to the list
            all_dataframes.append(city_data)
        time.sleep(5)  # Add a 5-second delay between requests

    merged_data = pd.concat(all_dataframes, ignore_index=True)
    TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f'{ROOT_DIR}/weather_data_{TIMESTAMP}.csv'
    merged_data.to_csv(filename, index=False)

    return filename

new_weater_data = sync_weather_data(start_year="2022", end_year="2023")