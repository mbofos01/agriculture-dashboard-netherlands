from geopy.geocoders import Nominatim
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import time

def transform_date(START_YEAR, END_YEAR):
    # Define the start and end dates for the data extraction
    START_DATE = f"{START_YEAR}-01-01"
    END_DATE = f"{END_YEAR}-12-31"
    return START_DATE, END_DATE


def get_daily_data(START_DATE,END_DATE):
    # Step 1: Create a geolocator object to take lantitude , lontitude
    geolocator = Nominatim(user_agent="my_geocoder_app", timeout=100)

    # Step 2: Define the list of cities in the Netherlands
    cities = [
        "Amsterdam", "Rotterdam", "The Hague", "Utrecht", "Eindhoven",
        "Groningen", "Tilburg", "Almere", "Breda", "Nijmegen",
        "Enschede", "Haarlem", "Haarlemmermeer", "Arnhem", "Zaanstad",
        "Amersfoort", "Apeldoorn", "Dordrecht", "Leiden", "Zoetermeer",
        "Emmen", "Venlo", "Leeuwarden", "Zwolle", "Helmond"
    ]

    #Get weather data from weather api for a specific city and time period
    def get_weather_data(city_name, start_date=START_DATE, end_date=END_DATE):
        # Get the location
        location = geolocator.geocode(f"{city_name}, Netherlands")

        if location:
            latitude = location.latitude
            longitude = location.longitude
            print(f" [E] Daily: {city_name}: Latitude: {latitude}, Longitude: {longitude}")

            # Setup the Open-Meteo API client with cache and retry on error
            cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
            retry_session = retry(cache_session, retries=10, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)

            # Define API parameters
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ["snowfall_sum", "precipitation_sum", "temperature_2m_min",
                          "temperature_2m_mean", "temperature_2m_max", "rain_sum",
                          "wind_speed_10m_max","et0_fao_evapotranspiration"]
            }

            # Fetch weather data in smaller chunks (year by year)
            daily_dataframes = []  # List to store DataFrames for each year

            for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
                # Update start and end dates for the current year
                current_start_date = f"{year}-01-01"
                current_end_date = f"{year}-12-31"
                params["start_date"] = current_start_date
                params["end_date"] = current_end_date

                responses = openmeteo.weather_api(url, params=params)
                response = responses[0]
                daily = response.Daily()

                # Extract weather variables
                snowfall_sum = daily.Variables(0).ValuesAsNumpy()
                precipitation_sum = daily.Variables(1).ValuesAsNumpy()
                temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
                temperature_2m_mean = daily.Variables(3).ValuesAsNumpy()
                temperature_2m_max = daily.Variables(4).ValuesAsNumpy()
                rain_sum = daily.Variables(5).ValuesAsNumpy()
                wind_speed_10m_max = daily.Variables(6).ValuesAsNumpy()
                potential_evapo_transpiration = daily.Variables(7).ValuesAsNumpy()

                # Create DataFrame for the current year
                daily_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=daily.Interval()),
                        inclusive="left"
                    ),
                    "Frost days": snowfall_sum,
                    "Precipitation rate": precipitation_sum,
                    "Minimum 2m temperature": temperature_2m_min,
                    "Mean 2m temperature": temperature_2m_mean,
                    "Maximum 2m temperature": temperature_2m_max,
                    "Wet days": rain_sum,
                    "wind_speed_10m_max": wind_speed_10m_max,
                    "potential evapo-transpiration": potential_evapo_transpiration
                }



            # Create DataFrame for the city
            daily_dataframe = pd.DataFrame(data=daily_data)
            daily_dataframe['city'] = city_name  # Add city name as a column
            return daily_dataframe
        else:
            print(f" [E] Daily: Location for {city_name} not found.")
            return pd.DataFrame()  # Return an empty DataFrame

    # Step 3: Collect dayly data for each city and merge DataFrames
    all_dataframes = []
    for city in cities:

        city_data = get_weather_data(city)
        if not city_data.empty:  # Check if the DataFrame is not empty
            all_dataframes.append(city_data)  # Append the DataFrame to the list
        time.sleep(20)

    # Merge all DataFrames by appending rows
    merged_data = pd.concat(all_dataframes, ignore_index=True)
    
    return merged_data


def get_hourly_data(START_DATE,END_DATE):
    geolocator = Nominatim(user_agent="my_geocoder_app", timeout=10)


    cities = [
        "Amsterdam", "Rotterdam", "The Hague", "Utrecht", "Eindhoven",
        "Groningen", "Tilburg", "Almere", "Breda", "Nijmegen",
        "Enschede", "Haarlem", "Haarlemmermeer", "Arnhem", "Zaanstad",
        "Amersfoort", "Apeldoorn", "Dordrecht", "Leiden", "Zoetermeer",
        "Emmen", "Venlo", "Leeuwarden", "Zwolle", "Helmond"
    ]


    def get_weather_data(city_name, start_date=START_DATE, end_date=END_DATE):
        # Get the location
        location = geolocator.geocode(f"{city_name}, Netherlands")

        if location:
            latitude = location.latitude
            longitude = location.longitude
            print(f" [E] Hourly: {city_name}: Latitude: {latitude}, Longitude: {longitude}")

            cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
            retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
            openmeteo = openmeteo_requests.Client(session = retry_session)

            # Make sure all required weather variables are listed here
            # The order of variables in hourly or daily is important to assign them correctly below
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
              "latitude": latitude,
              "longitude": longitude,
              "start_date": start_date,
              "end_date": end_date,
              "hourly": ["cloud_cover", "vapour_pressure_deficit"]
            }
            responses = openmeteo.weather_api(url, params=params)

            # Process first location. Add a for-loop for multiple locations or weather models
            response = responses[0]

            # Process hourly data. The order of variables needs to be the same as requested.
            hourly = response.Hourly()
            hourly_cloud_cover = hourly.Variables(0).ValuesAsNumpy()
            hourly_vapour_pressure_deficit = hourly.Variables(1).ValuesAsNumpy()

            hourly_data = {"date": pd.date_range(
              start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
              end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
              freq = pd.Timedelta(seconds = hourly.Interval()),
              inclusive = "left"
            )}

            hourly_data["Cloud cover"] = hourly_cloud_cover
            hourly_data["Vapour pressure"] = hourly_vapour_pressure_deficit

            hourly_dataframe = pd.DataFrame(data = hourly_data)
            hourly_dataframe['city'] = city_name  # Add city name as a column
            return hourly_dataframe
        else:
            print(f" [E] Hourly: Location for {city_name} not found.")
            return pd.DataFrame()  # Return an empty DataFrame

    # Step 3: Collect hourly data for each city and merge DataFrames
    all_dataframes = []
    for city in cities:
        city_data = get_weather_data(city)
        if not city_data.empty:  # Check if the DataFrame is not empty
            all_dataframes.append(city_data)  # Append the DataFrame to the list

        # time.sleep(65)  # Add a 1-second delay between requests


    # Merge all DataFrames by appending rows
    merged_data = pd.concat(all_dataframes, ignore_index=True)
    
    return merged_data