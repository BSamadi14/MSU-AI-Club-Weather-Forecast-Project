# AI Weather Forecaster: East Lansing, MI

A machine learning model that forecasts the next 72 hours of weather for East Lansing, plus a
Streamlit app that turns the forecast into an **outdoor planner**: when to walk to class, run or
bike, what to wear, and how much to trust each hour.

## Quick start

```bash
source .venv/bin/activate          # the prompt should now show (.venv)
python -m utils.download           # optional: refresh the data (2016 to ~6 days ago)
python main.py --retrain           # test on 2026, then train the final models (~5-10 min)
streamlit run app/main.py          # open the app
```

**Official competition script** (forecasts Nov 15-17 from Nov 14 and prints the RMSE):

```bash
python competition.py
```

It uses `models/weather_model_competition.pkl`, which was trained only on data **before
2026-11-14**, so it has never seen the starting day or the three scored days.

Practice on any other dates or API call with:

```bash
python compete.py --url "https://...open-meteo.com/..." --rolling
python compete.py --url "https://..." --forecast-from "2026-12-05 00:00" --hours 24   # times in UTC
```

`compete.py` adds the variables the model needs, switches to °C / km/h / UTC, picks a model that
works with the variables the API returned, forecasts autoregressively and prints the RMSE against
what actually happened. `--rolling` also starts a forecast from every hour and averages the error.

> After editing anything in `src/`, restart Streamlit (Ctrl+C, then run it again). It only
> reloads files in `app/` automatically.

## How the model works

1. **Data:** 95,000 hourly observations (Jan 2016 to Nov 2026) of 15 weather variables from the
   Open-Meteo archive API. Wind direction is converted to east-west / north-south components.
2. **Features (84):** for each of 16 variables, the current value, its change over the last 1, 3
   and 23 hours, and its 24-hour average, plus the hour of day and day of year as sine/cosine.
3. **Model:** one XGBoost regressor per variable predicts how much it changes in the next hour.
4. **Autoregressive forecast:** the prediction is appended to the 24-hour window, the oldest
   hour is dropped, and the model predicts again, 72 times.

The **backup model** skips the two soil variables, because Open-Meteo's *forecast* API returns them
empty. `compete.py` switches to it automatically.

## Results

### Competition score: **3.8791 °C** RMSE

From `python competition.py`: a 72-hour autoregressive forecast for Nov 15-17, 2026, starting from
the 24 hours of Nov 14, scored against the real temperatures. By day: 1.88 °C (Nov 15),
3.65 °C (Nov 16), 5.32 °C (Nov 17). The model ran too cold and missed a warm-up.

The competition model is the **average of the 16-variable and no-soil models**, each trained only
on data before 2026-11-14. Averaging was chosen on the 2025 validation year (4.33 °C vs 4.49 and
4.50 °C for the single models), never on the competition days.

### Test year

Temperature RMSE (°C) for 634 forecasts started every 12 hours in 2026, data the models never
trained on (trained on 2016-2025):

| Model | 1 h | 6 h | 24 h | 48 h | 72 h | Avg over 72 h |
|---|---|---|---|---|---|---|
| Week 5 model (2 weeks of Berlin data, 3 features) | 7.51 | 9.46 | 11.67 | 12.24 | 12.21 | 11.23 |
| Week 5 features, 2016-2025 East Lansing data | 0.95 | 3.34 | 4.32 | 5.17 | 5.22 | 4.82 |
| Main model (16 variables, 84 features) | 0.63 | 2.23 | 3.79 | 4.93 | 5.78 | 4.54 |
| Backup model (no soil variables) | 0.61 | 2.25 | 3.69 | 4.89 | 5.57 | 4.42 |
| **Average of both (competition model)** | **0.61** | **2.22** | **3.65** | **4.73** | **5.32** | **4.33** |
| Baseline: temperature stays the same | 1.46 | 6.73 | 4.62 | 5.85 | 6.07 | 6.48 |
| Baseline: yesterday repeats | 4.55 | 4.93 | 4.62 | 5.85 | 6.07 | 5.73 |

### What helped and what didn't

| Change | Effect on 72-hour RMSE | Kept? |
|---|---|---|
| Right location (East Lansing instead of Berlin) and 11 years instead of 2 weeks | 11.2 → 4.8 °C | ✅ |
| All 15 variables + lag / change / average features | 4.8 → 4.5 °C | ✅ |
| Averaging the 16-variable and no-soil models | 4.5 → 4.3 °C | ✅ |
| 24-hour history instead of 25 (the competition allows one day) | 4.3 → 4.5 °C for the single model | required |
| Fixing the validation leak (train and validation used to be the same rows) | an honest score | ✅ |
| Feature selection (fewer variables or features) | every smaller set did worse | ❌ |
| Hyperparameter tuning (30-setting random search) | best one-hour settings did *worse* at 72 hours | ❌ |
| Training on the model's own forecasts (DAgger) | 4.17 → 6.31 °C on 2025 | ❌ |

The tuning result is a lesson about autoregressive models: settings that predict the next hour best
aren't the ones that hold up best when predictions are fed back in 72 times.

Splits are always by time, never random: 2016-2024 train, 2025 validation (all choices), 2026
test. After choosing, the saved models were refit on everything through the latest data.

## The novel idea: an outdoor planner that's honest about uncertainty

- **80% ranges:** the model was run from every hour of 2026 and compared with what happened. The
  band on the chart is built from those errors, separately per season (winter and spring are
  much harder). On held-out weeks it contained the real temperature 77-80% of the time.
- **Calibrated chance of rain:** how often it really rained when the model predicted a similar
  amount. The model's rain forecast only beats the seasonal average for ~6 hours, so after that
  the app shows the typical chance for the season instead of pretending to know.
- **My schedule:** pick the times you're outside. For each day: feels-like temperature with its
  range, rain, wind, comfort rating, daylight, what to wear, forecast confidence.
- **Find the best time:** best slot each day to walk, run or bike, scored on feels-like
  temperature, wind, rain chance and darkness (sunrise/sunset is calculated from the date and
  location).

## Project layout

| Path | What it does |
|---|---|
| `utils/download.py` | downloads the data, a year per request, cached |
| `src/features.py` | the 84 features, built the same way for training and forecasting |
| `src/data_processor.py` | windows, training targets, split by year |
| `src/model.py` | training, feature selection, tuning, rollout (DAgger) experiment |
| `src/autoregressive.py` | 72-hour forecasting, live data download |
| `src/evaluate.py` | RMSE by hour ahead, baselines, error ranges and chance of rain |
| `src/planner.py` | comfort scores, clothing advice, daylight, best time slots |
| `main.py` | full pipeline |
| `competition.py` | official competition script (TODOs filled in) |
| `compete.py` | practice runner for any dates or API call |
| `app/main.py` | Streamlit app |
| `reports/` | test results, selection and tuning results, feature importance chart |
