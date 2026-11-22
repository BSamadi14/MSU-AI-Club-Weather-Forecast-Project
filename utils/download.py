"""Download historical hourly weather for East Lansing, MI from the Open-Meteo archive.

Run from the project root:
    python -m utils.download

Open-Meteo's free tier limits how many "weighted" calls you can make per minute/hour/day
(a request with many variables and a long date range counts as several calls). To stay
within the limits we:
  * download one year at a time,
  * pause between requests,
  * cache every response in .cache.sqlite, so re-running the script costs nothing.
"""
import time
from datetime import date, timedelta
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

LATITUDE = 42.7234
LONGITUDE = -84.4814
START_YEAR = 2016
# The archive API lags real time by a few days
END_DATE = date.today() - timedelta(days=6)
PAUSE_SECONDS = 5

# Keep this list in sync with src/autoregressive.py (the app requests the same variables)
HOURLY_VARIABLES = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "cloud_cover", "cloud_cover_low", "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "wind_gusts_10m", "pressure_msl", "shortwave_radiation",
]

DATA_PATH = Path("data/weather_data.csv")
URL = "https://archive-api.open-meteo.com/v1/archive"


def make_client():
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=1.0)
    return openmeteo_requests.Client(session=retry_session)


def download_range(client, start_date, end_date):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARIABLES,
    }
    response = client.weather_api(URL, params=params)[0]
    hourly = response.Hourly()

    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    # The order of variables in the response matches the order requested
    for i, name in enumerate(HOURLY_VARIABLES):
        data[name] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data)


def main():
    client = make_client()
    frames = []

    for year in range(START_YEAR, END_DATE.year + 1):
        start = f"{year}-01-01"
        end = min(date(year, 12, 31), END_DATE).isoformat()
        print(f"Downloading {start} to {end}...")
        frames.append(download_range(client, start, end))
        if year != END_DATE.year:
            time.sleep(PAUSE_SECONDS)

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="date").sort_values("date")
    print(f"\nDownloaded {len(df):,} hourly rows from {df['date'].min()} to {df['date'].max()}")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved to {DATA_PATH}")


if __name__ == "__main__":
    main()
