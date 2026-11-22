"""Feature engineering shared by training, evaluation and the app.

The model never sees a single row on its own. It sees a *window* of the last WINDOW hours
of observations, shaped (n_samples, WINDOW, n_base_vars), oldest hour first. From that
window we build features such as the current value, how much a variable changed in the
last 1 / 3 / 24 hours and its 24-hour average.

Using one function (window_features) everywhere guarantees that the features the model
was trained on are computed exactly the same way when we forecast autoregressively.
"""
import numpy as np
import pandas as pd

# Current hour + the 24 hours before it
WINDOW = 25

# Every variable the model can use as an input *and* has to forecast for the next hour.
# Wind direction (0-360°) is circular (359° is next to 1°), which trees handle badly,
# so it is replaced with east-west (u) and north-south (v) wind components.
# temperature_2m must stay first: it is the main forecast target.
ALL_BASE_VARS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "cloud_cover", "cloud_cover_low", "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm", "surface_pressure", "wind_speed_10m", "wind_u", "wind_v",
    "wind_gusts_10m", "pressure_msl", "shortwave_radiation",
]

# Physical limits applied to every autoregressive prediction
CLIP_RANGES = {
    "relative_humidity_2m": (0, 100),
    "cloud_cover": (0, 100),
    "cloud_cover_low": (0, 100),
    "precipitation": (0, None),
    "shortwave_radiation": (0, None),
    "wind_speed_10m": (0, None),
    "wind_gusts_10m": (0, None),
    "soil_moisture_0_to_7cm": (0, None),
}


def prepare_base(df):
    """Parse dates and add the derived base variables (wind components)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    direction = np.deg2rad(df["wind_direction_10m"])
    # Meteorological convention: direction is where the wind blows *from*
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(direction)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(direction)
    return df


def window_features(windows, times, base_vars):
    """Build the model's input features.

    windows: array (n_samples, WINDOW, len(base_vars)), oldest hour first
    times:   DatetimeIndex (n_samples,), the time of the newest hour in each window
    """
    feats = {}
    for j, var in enumerate(base_vars):
        x = windows[:, :, j]
        feats[var] = x[:, -1]
        feats[f"{var}_diff1"] = x[:, -1] - x[:, -2]
        feats[f"{var}_diff3"] = x[:, -1] - x[:, -4]
        feats[f"{var}_diff24"] = x[:, -1] - x[:, 0]
        feats[f"{var}_mean24"] = x[:, 1:].mean(axis=1)

    # Cyclical time features so that 23:00 is close to 00:00 and Dec 31 to Jan 1
    hour = np.asarray(times.hour)
    day = np.asarray(times.dayofyear)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["day_sin"] = np.sin(2 * np.pi * day / 365.25)
    feats["day_cos"] = np.cos(2 * np.pi * day / 365.25)

    return pd.DataFrame(feats)


def clip_values(values, base_vars):
    """Clip an array (..., len(base_vars)) to physically possible ranges, in place."""
    for j, var in enumerate(base_vars):
        if var in CLIP_RANGES:
            low, high = CLIP_RANGES[var]
            values[..., j] = np.clip(values[..., j], low, high)
    return values
