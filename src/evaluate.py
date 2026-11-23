"""Evaluate models the way they are actually used: 72-hour autoregressive forecasts.

One-step RMSE (predicting only the next hour) looks great for almost any model, because the
next hour is usually close to the current one. What matters for the app is how quickly
errors pile up when predictions are fed back in, so we start a forecast from many points in
the evaluation period and compare each forecast hour with what was actually observed.
"""
import numpy as np
import pandas as pd

from src.autoregressive import forecast_windows
from src.data_processor import make_windows
from src.features import WINDOW

HORIZONS = [1, 3, 6, 12, 24, 48, 72]


def _start_indices(df, year, steps, every_hours):
    """Indices of windows whose last observed hour is in `year` and that have `steps` future hours."""
    in_year = np.where(df["date"].dt.year.to_numpy() == year)[0]
    # The window ending at row r is window number r - (WINDOW - 1)
    rows = in_year[(in_year >= WINDOW - 1) & (in_year + steps < len(df))]
    rows = rows[::every_hours]
    return rows, rows - (WINDOW - 1)


def forecasts_and_truth(bundle, df, year, steps=72, every_hours=12, start_hours=None):
    """Run forecasts from many start points in `year`.

    Returns (preds, truth, rows): preds and truth are (n_starts, steps, len(base_vars)).
    start_hours optionally keeps only starts at those UTC hours.
    """
    base_vars = bundle["base_vars"]
    windows, times = make_windows(df, base_vars)
    rows, idx = _start_indices(df, year, steps, every_hours)
    if start_hours is not None:
        keep = np.isin(df["date"].dt.hour.to_numpy()[rows], start_hours)
        rows, idx = rows[keep], idx[keep]

    preds = forecast_windows(bundle, windows[idx], times[idx], steps)
    values = df[base_vars].to_numpy(dtype=np.float32)
    truth = np.stack([values[r + 1: r + 1 + steps] for r in rows])
    return preds, truth, rows


def autoregressive_errors(bundle, df, year, steps=72, every_hours=12, var="temperature_2m"):
    """Return the squared-error array (n_starts, steps) for `var` and the start rows."""
    preds, truth, rows = forecasts_and_truth(bundle, df, year, steps, every_hours)
    j = bundle["base_vars"].index(var)
    return (preds[:, :, j] - truth[:, :, j]) ** 2, rows


# A wet hour = at least this much precipitation (mm)
RAIN_THRESHOLD_MM = 0.1
# Buckets for the "chance of rain" lookup table
RAIN_HORIZON_EDGES = [0, 6, 24, 48, 72]
RAIN_PRECIP_EDGES = [0, 0.02, 0.05, 0.1, 0.3, np.inf]
# The model's rain amount only beats the seasonal average this many hours ahead (measured on
# held-out 2026 weeks); after that the chance of rain is the season's typical rate
RAIN_SKILL_HOURS = 6
# Bump when the uncertainty format changes so main.py recomputes it
UNCERTAINTY_VERSION = 2
SEASONS = ["Winter (Dec-Feb)", "Spring (Mar-May)", "Summer (Jun-Aug)", "Fall (Sep-Nov)"]


def season_index(month):
    """0 = Dec-Feb, 1 = Mar-May, 2 = Jun-Aug, 3 = Sep-Nov. Works on numbers or arrays."""
    return (np.asarray(month) % 12) // 3


def forecast_uncertainty(bundle, df, year, steps=72, every_hours=1, quantiles=(0.1, 0.9)):
    """Measure how wrong the model really was, to turn single-number forecasts into ranges.

    Starts a forecast every hour of `year` (so every time of day is represented) and passes
    the forecasts and what really happened to uncertainty_from_forecasts().
    """
    preds, truth, rows = forecasts_and_truth(bundle, df, year, steps, every_hours)
    start_months = df["date"].dt.month.to_numpy()[rows]
    result = uncertainty_from_forecasts(preds, truth, start_months, bundle["base_vars"], quantiles)
    result.update({"year": year, "n_forecasts": len(rows)})
    return result


def uncertainty_from_forecasts(preds, truth, start_months, base_vars, quantiles=(0.1, 0.9)):
    """Error statistics per season, used for the app's ranges and chance of rain.

    Forecasts are harder in winter and spring than in summer and fall, so everything is
    measured separately for each season (of the forecast's start time).

    Temperature bands: for each season and forecast hour h, the 10th and 90th percentile of
    (forecast - observed). A new forecast's 80% range is then
    [forecast - p90 error, forecast - p10 error]. This also corrects any systematic bias.

    Chance of rain: the model forecasts an amount of precipitation, not a probability. We
    group past forecasts by season, how far ahead they were and how much rain they predicted,
    and record how often at least RAIN_THRESHOLD_MM actually fell. That observed frequency
    is the chance of rain. Beyond RAIN_SKILL_HOURS the predicted amount is ignored and the
    chance is simply how often it rained at that season and forecast horizon.
    """
    steps = preds.shape[1]
    seasons = season_index(start_months)
    result = {"quantiles": list(quantiles), "seasons": SEASONS, "bands": {}}

    for var in ["temperature_2m", "apparent_temperature"]:
        j = base_vars.index(var)
        errors = preds[:, :, j] - truth[:, :, j]
        lower, upper = [], []
        for season in range(len(SEASONS)):
            # A season with no data (e.g. a partial year) falls back to all seasons together
            e = errors[seasons == season] if (seasons == season).any() else errors
            low_q, high_q = np.quantile(e, quantiles, axis=0)
            # forecast - high_q = lower edge, forecast - low_q = upper edge
            lower.append((-high_q).tolist())
            upper.append((-low_q).tolist())
        result["bands"][var] = {"lower_offset": lower, "upper_offset": upper}

    j = base_vars.index("precipitation")
    horizon = np.broadcast_to(np.arange(1, steps + 1), preds[:, :, j].shape)
    h_bin = np.digitize(horizon, RAIN_HORIZON_EDGES[1:-1], right=True)
    p_bin = np.digitize(preds[:, :, j], RAIN_PRECIP_EDGES[1:-1], right=False)
    s_bin = np.broadcast_to(seasons[:, None], h_bin.shape)
    wet = truth[:, :, j] >= RAIN_THRESHOLD_MM

    table = np.full((len(SEASONS), len(RAIN_HORIZON_EDGES) - 1, len(RAIN_PRECIP_EDGES) - 1), np.nan)
    for sb in range(table.shape[0]):
        in_season = s_bin == sb if (s_bin == sb).any() else np.ones_like(wet)
        for hb in range(table.shape[1]):
            base = in_season & (h_bin == hb)
            if RAIN_HORIZON_EDGES[hb] >= RAIN_SKILL_HOURS:
                table[sb, hb] = wet[base].mean()
                continue
            for pb in range(table.shape[2]):
                mask = base & (p_bin == pb)
                if mask.sum() >= 30:  # too few cases -> fall back to the overall rate below
                    table[sb, hb, pb] = wet[mask].mean()
            table[sb, hb] = np.where(np.isnan(table[sb, hb]), wet[base].mean(), table[sb, hb])
            # More predicted rain should never mean a lower chance of rain
            table[sb, hb] = np.maximum.accumulate(table[sb, hb])

    result["version"] = UNCERTAINTY_VERSION
    result["rain_chance"] = {
        "skill_hours": RAIN_SKILL_HOURS,
        "threshold_mm": RAIN_THRESHOLD_MM,
        "horizon_edges": RAIN_HORIZON_EDGES,
        "precip_edges": [float(e) for e in RAIN_PRECIP_EDGES],
        "table": table.tolist(),
    }
    return result


def baseline_errors(df, rows, steps=72, var="temperature_2m"):
    """Squared errors of two simple baselines every model should beat.

    persistence: the temperature stays exactly what it is now
    yesterday:   tomorrow looks like the last 24 hours (repeat them)
    """
    values = df[var].to_numpy(dtype=np.float32)
    truth = np.stack([values[r + 1: r + 1 + steps] for r in rows])
    persistence = np.repeat(values[rows][:, None], steps, axis=1)
    h = np.arange(steps)
    yesterday = np.stack([values[r - 23 + (h % 24)] for r in rows])
    return {
        "Persistence": (persistence - truth) ** 2,
        "Yesterday repeats": (yesterday - truth) ** 2,
    }


def rmse_by_horizon(squared_errors):
    """RMSE at each horizon in HORIZONS plus the average over all 72 hours."""
    rmse = np.sqrt(squared_errors.mean(axis=0))
    row = {f"{h}h": rmse[h - 1] for h in HORIZONS if h <= len(rmse)}
    row["mean 1-72h"] = np.sqrt(squared_errors.mean())
    return row


def score(bundle, df, year, steps=72, every_hours=12):
    """Single number used for model selection: RMSE over all 72 forecast hours."""
    squared, _ = autoregressive_errors(bundle, df, year, steps, every_hours)
    return float(np.sqrt(squared.mean()))


def comparison_table(bundles, df, year, steps=72, every_hours=12):
    """Table of RMSE by horizon for each model and the baselines."""
    rows = {}
    for name, bundle in bundles.items():
        squared, starts = autoregressive_errors(bundle, df, year, steps, every_hours)
        rows[name] = rmse_by_horizon(squared)
    for name, squared in baseline_errors(df, starts, steps).items():
        rows[name] = rmse_by_horizon(squared)
    table = pd.DataFrame(rows).T
    table.index.name = f"Temperature RMSE (°C), {year}, {len(starts)} forecasts"
    return table.round(2)
