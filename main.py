"""Weather forecaster pipeline.

    python -m utils.download        # (optional) refresh the data to include the latest days
    python main.py --retrain        # test on 2026, then train the final models on all data
    python main.py --retrain --select   # also rerun the slow feature selection + tuning experiments
    python main.py                  # show the saved test results and a live 72-hour forecast

How the data is used (always split by time, never randomly):
    2016-2024  training
    2025       validation: chose the features and hyperparameters, tested rollout training
    2026       test: the honest score below (models trained on 2016-2025 only)
    final      once everything is chosen, the saved models are refit on ALL data, so they know
               the most recent weeks (the competition uses the newest data)
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.autoregressive import get_current_weather, predict_autoregressive
from src.data_processor import load_data, make_supervised
from src.evaluate import comparison_table, forecast_uncertainty
from src.features import ALL_BASE_VARS
from src.model import (
    all_features, plot_feature_importances, prune_features, save_bundle,
    select_base_vars, train_with_rollouts, tune_hyperparameters,
)

DATA_PATH = Path("data/weather_data.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
VAL_YEAR = 2025
TEST_YEAR = 2026

# The soil variables only exist in Open-Meteo's archive API (the forecast API returns them
# empty), so a backup model without them is trained too. compete.py picks automatically.
NO_SOIL_VARS = [v for v in ALL_BASE_VARS if not v.startswith("soil_")]
MODELS = {
    "weather_model.pkl": ("All 16 variables", ALL_BASE_VARS),
    "weather_model_nosoil.pkl": ("No soil variables (backup)", NO_SOIL_VARS),
}
# Rounds of training on the model's own forecasts (see src/model.py: rollout_training_data).
# Tried and turned off: on 2025 validation, 72-hour RMSE went from 4.17 °C (normal) to
# 6.31 °C with 1 round and 4.39 °C with 2 rounds for the 16-variable model.
ROLLOUT_ROUNDS = 0


def fit(df, base_vars, last_year):
    """Train on every hour up to the end of last_year (plus rollout training if turned on)."""
    X, Y, target_times = make_supervised(df, base_vars)
    keep = target_times.year <= last_year
    return train_with_rollouts(X[keep], Y[keep], df, base_vars, all_features(base_vars),
                               last_year, rounds=ROLLOUT_ROUNDS)


def run_selection_experiments(df):
    """Week 6 experiments: feature selection and hyperparameter tuning (validation year 2025).

    Result: every smaller set of variables/features and every tuned setting did worse on
    72-hour forecasts than all 84 features with the default settings, so those are used.
    """
    print("\n=== Feature selection, step 1: which weather variables to use ===")
    base_vars, var_results, _ = select_base_vars(df, ALL_BASE_VARS, VAL_YEAR)
    print("\n=== Feature selection, step 2: prune engineered features ===")
    features, feat_results = prune_features(df, base_vars, VAL_YEAR)
    print("\n=== Hyperparameter tuning ===")
    _, tune_results = tune_hyperparameters(df, base_vars, features, VAL_YEAR)

    var_results.drop(columns="vars").to_csv(REPORT_DIR / "selection_variables.csv", index=False)
    feat_results.drop(columns="features").to_csv(REPORT_DIR / "selection_features.csv", index=False)
    tune_results.assign(params=tune_results["params"].map(json.dumps)).to_csv(
        REPORT_DIR / "tuning.csv", index=False)


def retrain(df, select=False):
    REPORT_DIR.mkdir(exist_ok=True)
    if select:
        run_selection_experiments(df)

    print(f"\n=== Test: train on 2016-{TEST_YEAR - 1}, forecast every 12 hours of {TEST_YEAR} ===")
    test_models, uncertainty = {}, {}
    for file, (name, base_vars) in MODELS.items():
        print(f"Training {name}...")
        test_models[name] = fit(df, base_vars, TEST_YEAR - 1)
        # Confidence ranges / chance of rain come from these honest out-of-sample errors
        uncertainty[file] = forecast_uncertainty(test_models[name], df, TEST_YEAR)
    table = comparison_table(test_models, df, TEST_YEAR)
    with pd.option_context("display.width", 200):
        print(table)
    table.to_csv(REPORT_DIR / "test_results.csv")

    last_year = df["date"].dt.year.max()
    print(f"\n=== Final models: train on all data through {df['date'].max():%Y-%m-%d} ===")
    for file, (name, base_vars) in MODELS.items():
        print(f"Training {name}...")
        bundle = fit(df, base_vars, last_year)
        bundle["uncertainty"] = uncertainty[file]
        bundle["trained_through"] = str(df["date"].max())
        bundle["test_rmse"] = table.loc[name].to_dict()
        save_bundle(bundle, MODEL_DIR / file)
    plot_feature_importances(joblib.load(MODEL_DIR / "weather_model.pkl"),
                             REPORT_DIR / "feature_importances.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrain", action="store_true", help="test on 2026 and train the final models")
    parser.add_argument("--select", action="store_true", help="also rerun feature selection + tuning (slow)")
    args = parser.parse_args()

    df = load_data(DATA_PATH)
    if args.retrain or not all((MODEL_DIR / f).exists() for f in MODELS):
        retrain(df, select=args.select)

    print(f"\n=== Test results: models trained on 2016-{TEST_YEAR - 1}, temperature RMSE (°C) on {TEST_YEAR} ===")
    with pd.option_context("display.width", 200):
        print(pd.read_csv(REPORT_DIR / "test_results.csv", index_col=0))

    bundle = joblib.load(MODEL_DIR / "weather_model.pkl")
    print(f"\n=== Live 72-hour forecast (model trained through {bundle.get('trained_through', '?')[:10]}) ===")
    forecast = predict_autoregressive(bundle, get_current_weather(), steps=72)
    print(forecast[["time", "temperature_2m"]].iloc[::6].round({"temperature_2m": 1}).to_string(index=False))


if __name__ == "__main__":
    main()
