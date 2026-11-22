import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.features import WINDOW, prepare_base, window_features


def load_data(path):
    """Load the downloaded CSV and check it is a gap-free hourly series."""
    print("Loading and processing weather data...")
    df = prepare_base(pd.read_csv(path)).sort_values("date").reset_index(drop=True)

    gaps = df["date"].diff().dropna() != pd.Timedelta(hours=1)
    if gaps.any():
        raise ValueError(f"Data has {gaps.sum()} gaps; the windowing code assumes consistent hourly rows")
    if df.isna().any().any():
        raise ValueError("Data has missing values; fill or drop them before training")

    return df


def make_windows(df, base_vars):
    """Return every WINDOW-hour window in the data as (n, WINDOW, n_vars) plus the newest time of each."""
    values = df[base_vars].to_numpy(dtype=np.float32)
    windows = sliding_window_view(values, WINDOW, axis=0).transpose(0, 2, 1)
    times = pd.DatetimeIndex(df["date"].iloc[WINDOW - 1:])
    return windows, times


def make_supervised(df, base_vars):
    """Build (X, Y) for one-step-ahead training.

    X = features from the window ending at hour t
    Y = change in each base variable from hour t to hour t + 1
    Also returns the time of hour t + 1 (used to split by year).
    """
    windows, times = make_windows(df, base_vars)
    values = df[base_vars].to_numpy(dtype=np.float32)

    # The last window has no "next hour", so drop it
    X = window_features(windows[:-1], times[:-1], base_vars)
    Y = pd.DataFrame(values[WINDOW:] - values[WINDOW - 1:-1], columns=base_vars)
    target_times = pd.DatetimeIndex(df["date"].iloc[WINDOW:])

    return X, Y, target_times


def split_by_year(X, Y, target_times, train_end_year=2024, val_year=2025):
    """Chronological split: train <= train_end_year, validation = val_year, test = everything after.

    Splitting by time (not randomly) matters for weather: neighbouring hours are nearly
    identical, so a random split would leak the answers into training.
    """
    years = target_times.year
    train = years <= train_end_year
    val = years == val_year
    test = years > val_year
    return (X[train], Y[train]), (X[val], Y[val]), (X[test], Y[test])
