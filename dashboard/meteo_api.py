import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
dutch_provinces = {
    "Drenthe": {"lat": 52.7913, "lon": 6.4869},
    "Flevoland": {"lat": 52.5255, "lon": 5.5274},
    "Fryslân": {"lat": 53.2081, "lon": 5.7459},  # Fryslân
    "Gelderland": {"lat": 52.0580, "lon": 5.2685},
    "Groningen": {"lat": 53.2194, "lon": 6.5665},
    "Limburg": {"lat": 51.1104, "lon": 5.8942},
    "Noord-Brabant": {"lat": 51.5859, "lon": 5.0983},
    "Noord-Holland": {"lat": 52.3792, "lon": 4.8998},
    "Overijssel": {"lat": 52.4334, "lon": 6.3396},
    "Utrecht": {"lat": 52.0907, "lon": 5.1214},
    "Zeeland": {"lat": 51.5050, "lon": 3.5833},
    "Zuid-Holland": {"lat": 51.9184, "lon": 4.4792},
}

def get_data(year,province):
	def create_dates(YEAR):
		return f"{YEAR}-01-01", f"{YEAR}-12-31", 

	def get_lat_lon(province):
		return dutch_provinces[province]['lat'], dutch_provinces[province]['lon']

	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": get_lat_lon(province)[0],
		"longitude": get_lat_lon(province)[1],
		"start_date": create_dates(year)[0],
		"end_date": create_dates(year)[1],
		"daily": ["temperature_2m_min", "temperature_2m_mean", "rain_sum", "wind_speed_10m_max", "wind_gusts_10m_max"]
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	# print(f"Elevation {response.Elevation()} m asl")
	# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_temperature_2m_min = daily.Variables(0).ValuesAsNumpy()
	daily_temperature_2m_mean = daily.Variables(1).ValuesAsNumpy()
	daily_rain_sum = daily.Variables(2).ValuesAsNumpy()
	daily_wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()
	daily_wind_gusts_10m_max = daily.Variables(4).ValuesAsNumpy()

	daily_data = {"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)}
	daily_data["temperature_2m_min"] = daily_temperature_2m_min
	daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
	daily_data["rain_sum"] = daily_rain_sum
	daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
	daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max

	return pd.DataFrame(data = daily_data)


