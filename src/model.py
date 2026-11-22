"""Training, feature selection and hyperparameter tuning.

A trained model is saved as a "bundle" dict so the app always knows how to build its inputs:
    {"model": MultiOutputRegressor, "base_vars": [...], "features": [...], "params": {...}}
"""
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.data_processor import make_supervised, make_windows, split_by_year
from src.evaluate import score
from src.features import WINDOW, clip_values, window_features

DEFAULT_PARAMS = {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6}
FEATURE_SUFFIXES = ["_diff1", "_diff3", "_diff24", "_mean24"]


def train_model(X, Y, base_vars, features, params=None):
    """Fit one XGBoost model per forecast variable and return a bundle."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    base = XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1, **params)
    model = MultiOutputRegressor(base).fit(X[features], Y[base_vars])
    return {"model": model, "base_vars": list(base_vars), "features": list(features), "params": params}


# Forecast steps at which rollout windows are kept as extra training rows
ROLLOUT_STEPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 71)


def rollout_training_data(bundle, df, last_year, every_hours=12, keep_steps=ROLLOUT_STEPS):
    """Extra training rows made from the model's *own* forecasts (the DAgger idea).

    A model trained only on real history never sees the slightly-wrong windows it creates
    when forecasting autoregressively, so its errors snowball. Here we let the model forecast
    from many start points in the training years, and at several steps save its window
    (which now contains its own predictions) with the target "real value next hour minus the
    model's current value". Training on these rows teaches the model to steer back toward
    reality instead of drifting.

    Only uses start points whose whole rollout ends in or before `last_year`, so no
    validation or test data leaks in.
    """
    base_vars, features, model = bundle["base_vars"], bundle["features"], bundle["model"]
    windows, times = make_windows(df, base_vars)
    values = df[base_vars].to_numpy(dtype=np.float32)
    max_step = max(keep_steps)
    keep = set(keep_steps)

    idx = np.arange(0, len(windows) - max_step - 1, every_hours)
    idx = idx[(times[idx] + pd.Timedelta(hours=max_step + 1)).year <= last_year]
    start_times = times[idx]
    w = windows[idx].copy()

    X_parts, Y_parts = [], []
    for step in range(max_step + 1):
        X_all = window_features(w, start_times + pd.Timedelta(hours=step), base_vars)
        if step in keep:
            # Window number i ends at data row i + WINDOW - 1; after `step` steps the
            # newest (predicted) row stands for row i + WINDOW - 1 + step
            truth_next = values[idx + WINDOW + step]
            X_parts.append(X_all)
            Y_parts.append(pd.DataFrame(truth_next - w[:, -1, :], columns=base_vars))
        if step < max_step:
            next_values = clip_values(w[:, -1, :] + model.predict(X_all[features]), base_vars)
            w = np.concatenate([w[:, 1:], next_values[:, None, :]], axis=1)

    return pd.concat(X_parts, ignore_index=True), pd.concat(Y_parts, ignore_index=True)


def train_with_rollouts(X, Y, df, base_vars, features, last_year, params=None, rounds=1):
    """Train normally, then retrain `rounds` times adding rows from the latest model's rollouts."""
    bundle = train_model(X, Y, base_vars, features, params)
    X_extra, Y_extra = [], []
    for _ in range(rounds):
        X_r, Y_r = rollout_training_data(bundle, df, last_year)
        X_extra.append(X_r)
        Y_extra.append(Y_r)
        bundle = train_model(pd.concat([X, *X_extra], ignore_index=True),
                             pd.concat([Y, *Y_extra], ignore_index=True), base_vars, features, params)
    bundle["rollout_rounds"] = rounds
    return bundle


def save_bundle(bundle, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def temperature_importances(bundle):
    """Gain importance of each input feature for the temperature model."""
    estimator = bundle["model"].estimators_[bundle["base_vars"].index("temperature_2m")]
    return pd.Series(estimator.feature_importances_, index=bundle["features"]).sort_values(ascending=False)


def feature_to_base_var(feature, base_vars):
    """Map a feature name like 'pressure_msl_diff3' back to its base variable."""
    for suffix in FEATURE_SUFFIXES:
        if feature.endswith(suffix) and feature[: -len(suffix)] in base_vars:
            return feature[: -len(suffix)]
    return feature if feature in base_vars else None  # time features map to None


def all_features(base_vars):
    """Every feature name window_features() builds for these base variables."""
    names = []
    for var in base_vars:
        names += [var] + [var + s for s in FEATURE_SUFFIXES]
    return names + ["hour_sin", "hour_cos", "day_sin", "day_cos"]


def select_base_vars(df, candidate_vars, val_year, subset_sizes=(1, 2, 3, 5, 8, 12, 16)):
    """Step 1 of feature selection: which weather variables should the model use at all?

    Every variable the model uses as input must also be forecast for the next hour, so extra
    variables cost accuracy whenever they're hard to predict. We rank variables by how much the
    temperature model relies on them, then score the top-k for several k on the validation year
    using the real 72-hour autoregressive forecast.
    """
    X, Y, t = make_supervised(df, candidate_vars)
    (X_tr, Y_tr), _, _ = split_by_year(X, Y, t, train_end_year=val_year - 1, val_year=val_year)

    full = train_model(X_tr, Y_tr, candidate_vars, all_features(candidate_vars))
    imp = temperature_importances(full)
    by_var = imp.groupby(lambda f: feature_to_base_var(f, candidate_vars)).sum()
    by_var = by_var.drop(index="temperature_2m").sort_values(ascending=False)
    ranking = ["temperature_2m"] + list(by_var.index)
    print("Base variables ranked by importance to temperature:\n ", ranking)

    results = []
    for k in subset_sizes:
        k = min(k, len(ranking))
        vars_k = ranking[:k]
        bundle = full if k == len(candidate_vars) else train_model(X_tr, Y_tr, vars_k, all_features(vars_k))
        rmse = score(bundle, df, val_year)
        print(f"  top {k:2d} variables -> validation 72h RMSE {rmse:.3f}")
        results.append({"n_vars": k, "rmse": rmse, "vars": vars_k})

    results = pd.DataFrame(results)
    best = results.loc[results["rmse"].idxmin()]
    return list(best["vars"]), results, imp


def prune_features(df, base_vars, val_year, keep_fractions=(1.0, 0.75, 0.5, 0.3)):
    """Step 2 of feature selection: drop the least important engineered columns."""
    X, Y, t = make_supervised(df, base_vars)
    (X_tr, Y_tr), _, _ = split_by_year(X, Y, t, train_end_year=val_year - 1, val_year=val_year)

    full = train_model(X_tr, Y_tr, base_vars, all_features(base_vars))
    ranked = list(temperature_importances(full).index)

    results = []
    for frac in keep_fractions:
        n = max(4, round(len(ranked) * frac))
        bundle = full if n == len(ranked) else train_model(X_tr, Y_tr, base_vars, ranked[:n])
        rmse = score(bundle, df, val_year)
        print(f"  keep {n:3d}/{len(ranked)} features -> validation 72h RMSE {rmse:.3f}")
        results.append({"n_features": n, "rmse": rmse, "features": ranked[:n]})

    results = pd.DataFrame(results)
    best = results.loc[results["rmse"].idxmin()]
    return list(best["features"]), results


def tune_hyperparameters(df, base_vars, features, val_year, n_iter=30, finalists=5):
    """Randomized search over a wide grid, then re-rank the best candidates by 72h forecast skill.

    RandomizedSearchCV (with TimeSeriesSplit so folds never train on the future) scores the
    temperature model's one-hour-ahead error. The top `finalists` are then trained for every
    variable and compared on the validation year's autoregressive forecasts, which is what the
    app actually does.
    """
    X, Y, t = make_supervised(df, base_vars)
    (X_tr, Y_tr), _, _ = split_by_year(X, Y, t, train_end_year=val_year - 1, val_year=val_year)

    param_distributions = {
        "n_estimators": randint(100, 1000),
        "learning_rate": loguniform(0.01, 0.3),
        "max_depth": randint(3, 11),
        "min_child_weight": randint(1, 20),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.4, 0.6),
        "reg_lambda": loguniform(0.1, 20),
    }
    search = RandomizedSearchCV(
        XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1),
        param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=3),
        random_state=42,
        verbose=1,
    )
    search.fit(X_tr[features], Y_tr["temperature_2m"])

    cv = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(finalists)
    results = [{"params": DEFAULT_PARAMS, "cv_rmse": np.nan,
                "rmse": score(train_model(X_tr, Y_tr, base_vars, features), df, val_year)}]
    print(f"  default params -> validation 72h RMSE {results[0]['rmse']:.3f}")
    for _, row in cv.iterrows():
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in row["params"].items()}
        rmse = score(train_model(X_tr, Y_tr, base_vars, features, params), df, val_year)
        print(f"  {params} -> one-step CV RMSE {-row['mean_test_score']:.4f}, validation 72h RMSE {rmse:.3f}")
        results.append({"params": params, "cv_rmse": -row["mean_test_score"], "rmse": rmse})

    results = pd.DataFrame(results)
    return results.loc[results["rmse"].idxmin(), "params"], results


def plot_feature_importances(bundle, path=None, top_n=25):
    """Bar chart of the temperature model's most important features."""
    imp = temperature_importances(bundle).head(top_n).sort_values()
    plt.figure(figsize=(10, 8))
    plt.barh(imp.index, imp.values)
    plt.title("Feature importances for the temperature model")
    plt.xlabel("Relative importance (gain)")
    plt.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=120)
        plt.close()
    else:
        plt.show()
