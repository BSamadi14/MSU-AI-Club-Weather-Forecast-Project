"""Competition runner: forecast temperature from any Open-Meteo API call and score it.

Examples
--------
Use the API call you're given. The script adds the weather variables the model needs, forces
default units (°C, km/h, mm) and UTC time, and leaves the location and dates alone:

    python compete.py --url "https://archive-api.open-meteo.com/v1/archive?latitude=42.7234&longitude=-84.4814&start_date=2026-11-10&end_date=2026-11-17&hourly=temperature_2m"

Or just give dates (archive API, East Lansing):

    python compete.py --start 2026-11-10 --end 2026-11-17

Or a CSV with a "date" column (UTC) plus the weather variables:

    python compete.py --csv some_data.csv

Choosing what to forecast
-------------------------
By default the first 25 hours are used as history and every remaining hour is forecast.
    --forecast-from "2026-11-15 00:00"   last observed hour (UTC) to forecast from
    --hours 24                          how many hours to forecast
    --rolling                           also start a forecast from every hour and report the
                                        average error (a fairer score than one single run)

If the data doesn't contain some variables (the forecast API returns no soil data, for
example), a model that doesn't need them is picked automatically.
"""
import argparse
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import joblib
import numpy as np
import pandas as pd
import requests

from src.autoregressive import HOURLY_VARIABLES, LATITUDE, LONGITUDE, forecast_windows, predict_autoregressive
from src.data_processor import make_windows
from src.features import WINDOW, prepare_base

# Tried in order; the first one whose inputs are all in the data is used
MODEL_PATHS = [Path("models/weather_model.pkl"), Path("models/weather_model_nosoil.pkl")]
UNIT_PARAMS = ["temperature_unit", "wind_speed_unit", "windspeed_unit", "precipitation_unit"]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def build_url(url):
    """Ask for every variable the models can use, in default units and UTC."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or "open-meteo.com" not in parts.netloc:
        raise SystemExit(
            f"--url needs a full Open-Meteo API link starting with https://, got: {url}\n"
            "Example: --url \"https://archive-api.open-meteo.com/v1/archive?latitude=42.7234"
            "&longitude=-84.4814&start_date=2026-11-10&end_date=2026-11-20&hourly=temperature_2m\""
        )
    query = {k: v[0] for k, v in parse_qs(parts.query).items()}
    for p in UNIT_PARAMS + ["timeformat", "format"]:
        query.pop(p, None)
    query["hourly"] = ",".join(HOURLY_VARIABLES)
    query["timezone"] = "GMT"
    return urlunparse(parts._replace(query=urlencode(query, safe=",")))


def fetch(url):
    url = build_url(url)
    print(f"Requesting:\n  {url}\n")
    response = requests.get(url, timeout=60)
    data = response.json()
    if response.status_code != 200 or data.get("error"):
        raise SystemExit(f"Open-Meteo error: {data.get('reason', response.text[:300])}")
    hourly = data["hourly"]
    df = pd.DataFrame({k: v for k, v in hourly.items() if k != "time"}, dtype=float)
    df.insert(0, "date", pd.to_datetime(hourly["time"]).tz_localize("UTC"))
    print(f"Got {len(df)} hours at {data['latitude']:.3f}, {data['longitude']:.3f}: "
          f"{df['date'].iloc[0]:%Y-%m-%d %H:%M} to {df['date'].iloc[-1]:%Y-%m-%d %H:%M} UTC")
    return df


def load_csv(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    df["date"] = dates.dt.tz_localize("UTC") if dates.dt.tz is None else dates.dt.tz_convert("UTC")
    return df


def required_columns(bundle):
    """Raw API columns a model needs (wind_u / wind_v are built from speed and direction)."""
    cols = set()
    for var in bundle["base_vars"]:
        cols |= {"wind_speed_10m", "wind_direction_10m"} if var in ("wind_u", "wind_v") else {var}
    return cols


def choose_model(df):
    for path in MODEL_PATHS:
        if not path.exists():
            continue
        bundle = joblib.load(path)
        missing = [c for c in required_columns(bundle) if c not in df or df[c].isna().all()]
        if not missing:
            print(f"Using {path.name} ({len(bundle['base_vars'])} variables)")
            return bundle
        print(f"Skipping {path.name}: data has no {', '.join(sorted(missing))}")
    raise SystemExit("No saved model can run on this data. Run `python main.py --retrain` first.")


def clean(df, bundle):
    """Keep the needed columns, fill short gaps, and make sure hours are consecutive."""
    cols = sorted(required_columns(bundle))
    df = df[["date"] + cols].drop_duplicates("date").sort_values("date").set_index("date")
    full = pd.date_range(df.index[0], df.index[-1], freq="h")
    df = df.reindex(full)
    gaps = df.isna().any(axis=1).sum()
    if gaps:
        print(f"Filling {gaps} hours with missing values by interpolation")
        df = df.interpolate(limit_direction="both")
    return df.rename_axis("date").reset_index()


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def single_forecast(bundle, df, forecast_from, hours):
    if forecast_from is None:
        end = WINDOW - 1
    else:
        ts = pd.Timestamp(forecast_from)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        matches = np.where(df["date"] == ts)[0]
        if not len(matches):
            raise SystemExit(f"{ts} is not in the data")
        end = int(matches[0])
    if end < WINDOW - 1:
        raise SystemExit(f"Need {WINDOW} hours of history before the forecast start")
    available = len(df) - end - 1
    hours = hours or available
    if hours <= 0:
        raise SystemExit("No hours left after the forecast start")

    history = df.iloc[end - WINDOW + 1: end + 1]
    forecast = predict_autoregressive(bundle, history, steps=hours)
    actual = df["temperature_2m"].iloc[end + 1: end + 1 + hours].to_numpy()
    result = pd.DataFrame({
        "time_utc": forecast["time"],
        "hours_ahead": np.arange(1, hours + 1),
        "predicted_c": forecast["temperature_2m"].round(2),
    })
    result["actual_c"] = np.concatenate([actual, np.full(hours - len(actual), np.nan)])
    result["error_c"] = (result["predicted_c"] - result["actual_c"]).round(2)

    print(f"\nForecast from {df['date'].iloc[end]:%Y-%m-%d %H:%M} UTC for {hours} hours")
    print(result.iloc[np.r_[0:min(6, hours), 6:hours:6]].to_string(index=False))
    scored = result.dropna()
    if len(scored):
        e = rmse(scored["predicted_c"], scored["actual_c"])
        print(f"\nTemperature RMSE over {len(scored)} hours: {e:.3f} °C ({e * 1.8:.3f} °F)")
        for h in [1, 6, 12, 24, 48, 72]:
            if h <= len(scored):
                print(f"  first {h:2d} h: {rmse(scored['predicted_c'][:h], scored['actual_c'][:h]):.3f} °C")
    return result


def rolling_score(bundle, df, hours):
    """Start a forecast from every hour that has `hours` of truth after it; average the errors."""
    hours = hours or 24
    clean_df = prepare_base(df)
    windows, times = make_windows(clean_df, bundle["base_vars"])
    n = len(df) - (WINDOW - 1) - hours
    if n <= 0:
        print(f"\nNot enough data for a rolling {hours}-hour evaluation")
        return
    preds = forecast_windows(bundle, windows[:n], times[:n], hours)[:, :, bundle["base_vars"].index("temperature_2m")]
    temps = df["temperature_2m"].to_numpy()
    truth = np.stack([temps[i + WINDOW: i + WINDOW + hours] for i in range(n)])
    by_h = np.sqrt(((preds - truth) ** 2).mean(axis=0))
    print(f"\nRolling evaluation: {n} forecasts of {hours} hours each")
    print(f"  average RMSE {np.sqrt(((preds - truth) ** 2).mean()):.3f} °C")
    print("  by hours ahead: " + ", ".join(f"{h}h {by_h[h - 1]:.2f}" for h in [1, 6, 12, 24, 48, 72] if h <= hours))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Open-Meteo API call (any endpoint)")
    source.add_argument("--start", help="start date YYYY-MM-DD (archive API, East Lansing)")
    source.add_argument("--csv", help="CSV with a date column and weather variables")
    parser.add_argument("--end", help="end date YYYY-MM-DD (with --start)")
    parser.add_argument("--forecast-from", help="last observed hour to forecast from, UTC, e.g. '2026-11-15 00:00'")
    parser.add_argument("--hours", type=int, help="hours to forecast (default: all remaining)")
    parser.add_argument("--rolling", action="store_true", help="also score forecasts started from every hour")
    parser.add_argument("--out", default="reports/competition_forecast.csv", help="where to save the forecast")
    args = parser.parse_args()

    if args.url:
        df = fetch(args.url)
    elif args.start:
        if not args.end:
            parser.error("--start needs --end")
        df = fetch(f"{ARCHIVE_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
                   f"&start_date={args.start}&end_date={args.end}")
    else:
        df = load_csv(args.csv)

    bundle = choose_model(df)
    # Drop trailing hours the API hasn't filled in yet
    needed = sorted(required_columns(bundle))
    complete = df[needed].notna().all(axis=1)
    df = df.loc[: complete[complete].index[-1]] if complete.any() else df
    df = clean(df, bundle)

    result = single_forecast(bundle, df, args.forecast_from, args.hours)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"\nSaved forecast to {args.out}")

    if args.rolling:
        rolling_score(bundle, df, args.hours)


if __name__ == "__main__":
    main()
