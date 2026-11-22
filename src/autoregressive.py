import datetime

import numpy as np
import pandas as pd

from src.features import WINDOW, clip_values, prepare_base, window_features

LATITUDE = 42.7234
LONGITUDE = -84.4814

# Keep in sync with utils/download.py
HOURLY_VARIABLES = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "cloud_cover", "cloud_cover_low", "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "wind_gusts_10m", "pressure_msl", "shortwave_radiation",
]


def forecast_windows(bundle, windows, start_times, steps):
    """Autoregressively forecast many starting points at once.

    bundle:      dict saved by src.model (model, base_vars, features)
    windows:     array (n, WINDOW, len(base_vars)) of observed history, oldest first
    start_times: DatetimeIndex (n,), time of the last observed hour of each window
    Returns an array (n, steps, len(base_vars)); [:, h] is the forecast for h + 1 hours ahead.
    """
    model, base_vars, features = bundle["model"], bundle["base_vars"], bundle["features"]
    window = np.array(windows, dtype=np.float32, copy=True)
    out = np.empty((len(window), steps, len(base_vars)), dtype=np.float32)

    for step in range(steps):
        times = start_times + pd.Timedelta(hours=step)
        X = window_features(window, times, base_vars)[features]
        # The model predicts the change over the next hour, so add it to the latest value
        next_values = clip_values(window[:, -1, :] + model.predict(X), base_vars)
        out[:, step] = next_values
        # Slide the window forward: drop the oldest hour, append the prediction
        window = np.concatenate([window[:, 1:], next_values[:, None, :]], axis=1)

    return out


def predict_autoregressive(bundle, history, steps=72):
    """Forecast from a DataFrame of the last WINDOW (or more) observed hours.

    Returns a DataFrame with one row per forecast hour, a "time" column and one column per
    forecast variable (temperature_2m, relative_humidity_2m, ...).

    An ensemble bundle ({"members": [bundle, ...]}) runs each member's own 72-hour forecast
    and averages them; only variables every member forecasts are kept.
    """
    if "members" in bundle:
        forecasts = [predict_autoregressive(m, history, steps) for m in bundle["members"]]
        shared = [c for c in forecasts[0].columns if c != "time" and all(c in f for f in forecasts)]
        averaged = forecasts[0][["time"]].copy()
        for column in shared:
            averaged[column] = np.mean([f[column].to_numpy() for f in forecasts], axis=0)
        return averaged

    history = prepare_base(history).sort_values("date").tail(WINDOW)
    if len(history) < WINDOW:
        raise ValueError(f"Need {WINDOW} hours of history, got {len(history)}")

    base_vars = bundle["base_vars"]
    window = history[base_vars].to_numpy(dtype=np.float32)[None]
    start_time = pd.DatetimeIndex([history["date"].iloc[-1]])

    preds = forecast_windows(bundle, window, start_time, steps)[0]
    forecast = pd.DataFrame(preds, columns=base_vars)
    forecast.insert(0, "time", [start_time[0] + pd.Timedelta(hours=h + 1) for h in range(steps)])
    return forecast


def get_current_weather(hours=WINDOW):
    """Download the most recent `hours` complete hourly observations for East Lansing."""
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    today = datetime.date.today()
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        # A few days of padding: the newest hours in the archive are sometimes still empty
        "start_date": (today - datetime.timedelta(days=4)).isoformat(),
        "end_date": (today - datetime.timedelta(days=1)).isoformat(),
        "hourly": HOURLY_VARIABLES,
    }

    # Cache for an hour only: an "expire never" cache would serve stale observations forever
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)
    response = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]

    hourly = response.Hourly()
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, name in enumerate(HOURLY_VARIABLES):
        data[name] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    # Keep everything up to the last hour where every variable is present
    complete = df.notna().all(axis=1)
    if not complete.any():
        raise ValueError("Open-Meteo returned no complete hourly observations")
    df = df.loc[: complete[complete].index[-1]].ffill()

    return df.tail(hours).reset_index(drop=True)
