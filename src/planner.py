"""Campus outdoor planner: turn the raw forecast into decisions.

* add_uncertainty()   adds 80% ranges for temperature / feels-like and a calibrated chance of
                      rain, using the model's measured past errors (see src.evaluate)
* comfort_score()     0-100 score for how pleasant an hour is for walking, running or biking
* what_to_wear()      clothing advice from feels-like temperature, rain chance and wind
* best_windows()      the best time slot on each day for an activity
* is_daylight()       whether the sun is up (running or biking in the dark costs points)

Everything works in °C and km/h; the app converts units for display.
"""
import numpy as np
import pandas as pd

from src.evaluate import season_index

# Comfortable feels-like range (°C), and how much wind / rain bothers each activity.
# Running warms you up, so its comfortable range is cooler; biking makes its own wind.
ACTIVITIES = {
    "Walk to class": {"ideal": (12, 24), "wind_start": 20, "wind_penalty": 1.5, "rain_penalty": 0.8, "dark_penalty": 5},
    "Run": {"ideal": (5, 18), "wind_start": 20, "wind_penalty": 1.5, "rain_penalty": 0.7, "dark_penalty": 25},
    "Bike": {"ideal": (12, 25), "wind_start": 15, "wind_penalty": 2.5, "rain_penalty": 1.0, "dark_penalty": 25},
}
TEMP_PENALTY_PER_DEGREE = 4  # points lost per °C outside the comfortable range


def add_uncertainty(forecast, uncertainty):
    """Add *_low / *_high columns and rain_chance (0-1) to a forecast with one row per hour ahead.

    forecast must be in forecast-hour order: row 0 is 1 hour ahead. The error statistics for
    the season the forecast starts in are used.
    """
    forecast = forecast.copy()
    n = len(forecast)
    season = int(season_index(forecast["time"].iloc[0].month))

    for var, column in [("temperature_2m", "temperature_c"), ("apparent_temperature", "apparent_temperature")]:
        band = uncertainty["bands"][var]
        forecast[f"{column}_low"] = forecast[column] + np.asarray(band["lower_offset"][season][:n])
        forecast[f"{column}_high"] = forecast[column] + np.asarray(band["upper_offset"][season][:n])

    rain = uncertainty["rain_chance"]
    horizon = np.arange(1, n + 1)
    h_bin = np.digitize(horizon, rain["horizon_edges"][1:-1], right=True)
    p_bin = np.digitize(forecast["precipitation"].to_numpy(), rain["precip_edges"][1:-1], right=False)
    forecast["rain_chance"] = np.asarray(rain["table"])[season, h_bin, p_bin]

    return forecast


def is_daylight(times_utc, latitude, longitude):
    """True when the sun is above the horizon, using NOAA's approximate solar position formulas.

    times_utc: timezone-aware timestamps (Series or DatetimeIndex) in any timezone.
    """
    t = pd.DatetimeIndex(times_utc).tz_convert("UTC")
    hour = np.asarray(t.hour + t.minute / 60)
    gamma = 2 * np.pi / 365 * (np.asarray(t.dayofyear) - 1 + (hour - 12) / 24)
    declination = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
                   - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
                   - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    eq_time = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    solar_minutes = hour * 60 + eq_time + 4 * longitude
    hour_angle = np.deg2rad(solar_minutes / 4 - 180)
    lat = np.deg2rad(latitude)
    cos_zenith = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    elevation = 90 - np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))
    return elevation > -0.833  # the standard sunrise/sunset definition


def comfort_score(feels_like_c, wind_kmh, rain_chance, activity, daylight=True):
    """0 (miserable) to 100 (perfect). Works on single numbers or whole columns."""
    a = ACTIVITIES[activity]
    low, high = a["ideal"]
    feels_like_c = np.asarray(feels_like_c, dtype=float)
    degrees_outside = np.maximum(low - feels_like_c, 0) + np.maximum(feels_like_c - high, 0)

    penalty = (
        TEMP_PENALTY_PER_DEGREE * degrees_outside
        + a["wind_penalty"] * np.maximum(np.asarray(wind_kmh, dtype=float) - a["wind_start"], 0)
        + a["rain_penalty"] * 100 * np.asarray(rain_chance, dtype=float)
        + a["dark_penalty"] * ~np.asarray(daylight, dtype=bool)
    )
    return np.clip(100 - penalty, 0, 100)


def score_label(score):
    if score >= 80:
        return "🟢 Great"
    if score >= 60:
        return "🟡 Good"
    if score >= 40:
        return "🟠 Fair"
    return "🔴 Poor"


CONFIDENCE_WIDTHS = (7, 13)  # °C width of the 80% feels-like range for High / Medium


def confidence_label(low, high):
    """How much to trust an hour, from the width of its 80% feels-like range (°C).

    Roughly: next-day summer/fall forecasts are High, next-day winter/spring and day-3
    summer/fall forecasts are Medium, and day-3 winter/spring forecasts are Low.
    """
    width = high - low
    if width < CONFIDENCE_WIDTHS[0]:
        return "High"
    if width < CONFIDENCE_WIDTHS[1]:
        return "Medium"
    return "Low"


def what_to_wear(feels_like_c, rain_chance, wind_kmh, dark=False):
    """Short clothing advice for one hour."""
    if feels_like_c < -10:
        items = ["heavy winter coat", "hat", "gloves", "scarf"]
    elif feels_like_c < 0:
        items = ["winter coat", "hat", "gloves"]
    elif feels_like_c < 8:
        items = ["warm jacket"]
    elif feels_like_c < 15:
        items = ["light jacket or sweater"]
    elif feels_like_c < 22:
        items = ["long sleeves or a t-shirt"]
    elif feels_like_c < 27:
        items = ["t-shirt and shorts"]
    else:
        items = ["light clothes", "water bottle"]

    if rain_chance >= 0.4:
        items.append("snow boots" if feels_like_c < 0 else "rain jacket or umbrella")
    elif rain_chance >= 0.2:
        items.append("maybe an umbrella")
    if wind_kmh >= 30 and feels_like_c >= 8:
        items.append("windbreaker")
    if dark:
        items.append("reflective gear or a light")

    return ", ".join(items).capitalize()


def best_windows(forecast, activity, duration_hours=1, earliest_hour=7, latest_hour=21):
    """The best start time on each day for an activity lasting `duration_hours`.

    A window's score is the average of its hourly scores. Only windows that start at or after
    earliest_hour and end by latest_hour (local time) count. Returns one row per day in date
    order, with is_best marking the best window overall.
    """
    daylight = forecast["daylight"] if "daylight" in forecast else True
    scores = pd.Series(
        comfort_score(forecast["apparent_temperature"], forecast["wind_speed_10m"],
                      forecast["rain_chance"], activity, daylight),
        index=forecast.index,
    )
    hours = forecast["time"].dt.hour

    candidates = []
    for i in range(len(forecast) - duration_hours + 1):
        window = forecast.iloc[i:i + duration_hours]
        end_hour = hours.iloc[i] + duration_hours
        same_day = window["time"].dt.date.nunique() == 1
        if hours.iloc[i] >= earliest_hour and end_hour <= latest_hour and same_day:
            candidates.append((i, scores.iloc[i:i + duration_hours].mean()))

    best_per_day = {}
    for i, score in candidates:
        day = forecast["time"].iloc[i].date()
        if day not in best_per_day or score > best_per_day[day][1]:
            best_per_day[day] = (i, score)
    chosen = [best_per_day[day] for day in sorted(best_per_day)]

    rows = []
    for i, score in chosen:
        window = forecast.iloc[i:i + duration_hours]
        rows.append({
            "start": window["time"].iloc[0],
            "end": window["time"].iloc[-1] + pd.Timedelta(hours=1),
            "score": score,
            "feels_like_c": window["apparent_temperature"].mean(),
            "rain_chance": window["rain_chance"].max(),
            "wind_kmh": window["wind_speed_10m"].max(),
            "dark": bool(("daylight" in window) and not window["daylight"].all()),
            "confidence": confidence_label(window["apparent_temperature_low"].mean(),
                                           window["apparent_temperature_high"].mean()),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["is_best"] = result["score"] == result["score"].max()
    return result
