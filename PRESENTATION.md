# Competition presentation notes (about 3-4 minutes)

## 1. The problem (20 s)
- Anyone can look up a forecast. What's missing is *what it means for my day* and *how much to trust it*.
- Weather apps show one number for 3 days out as if it were as certain as the next hour.

## 2. The idea (30 s)
**An outdoor planner for campus that's honest about uncertainty.**
- Tell it when you're outside (walking to class at 8, 12 and 5). For each day it shows the
  feels-like temperature *with a likely range*, rain chance, wind, a comfort rating and what to wear.
- Ask for the best time to run, walk or bike. It picks the best slot each day, avoiding
  darkness, wind and rain.

## 3. Demo (60-90 s)
Start the app beforehand: `source .venv/bin/activate`, then `streamlit run app/main.py`.
1. **Forecast tab:** point at the shaded 80% band and the "Now" line. "It gets wider because
   errors add up, and it's wider in spring than in summer."
2. **Outdoor planner → My schedule:** add or remove a time; read one row out loud
   (feels like, range, what to wear, confidence).
3. **Find the best time:** switch Run → Bike, move "How long" to 2 hours, and point out that the ⭐
   slot avoids darkness ("sunset is 5:10 PM in late November").
4. Open **How the scores work** if asked how scoring works.

## 4. How it works (45 s)
- 11 years of hourly East Lansing data, 15 variables, 84 features from the last 24 hours (current
  values, 1/3/23-hour changes, 24-hour averages, time of day and year).
- XGBoost predicts the next hour's *change* for every variable, then feeds its prediction back in,
  72 times (autoregressive). Two versions (with and without soil data) forecast separately
  and their forecasts are averaged.
- The ranges come from the model's *real* mistakes: it was run from every hour of 2026 (never
  trained on) and compared with what happened, per season. On held-out weeks the "80%" range held
  the real temperature 77-80% of the time.

## 5. Results (30 s)
- **Competition score: 3.8791 °C** (Nov 15-17). The model was trained only on data before Nov 14.
- 2026 test, average temperature error over 72 hours: **4.33 °C**, down from **11.2 °C** for my
  Week 5 model. First hour: **0.61 °C**. Next day: **3.65 °C**.
- Beats "repeat yesterday" (5.7 °C) and "nothing changes" (6.5 °C).

## 6. What I learned (20 s)
- **Data beat everything:** the right location and more years cut the error by more than half (11.2 → 4.8 °C). Model
  tweaks did less.
- **Tuning for one hour ≠ tuning for 72 hours.** Settings that predicted the next hour best did
  worse over 3 days.
- **Know when your model can't help.** Its rain forecast only beats the seasonal average for
  ~6 hours, so after that the app says so instead of guessing.
- Feature selection and training on its own forecasts (DAgger) both made it worse, so both
  were left out.

## Likely questions
- **Why not just tune more?** Tried a 30-setting random search; nothing beat the defaults over 72 hours.
- **What if the API has no soil data?** A backup model without soil variables (4.71 °C) is
  picked automatically.
- **Overfitting?** Every choice was made on 2025 data; the scores above are on 2026, which no choice used.
- **Why do forecasts start "yesterday"?** The archive API lags about a day, so the app shows the
  forecast from the last real observation and marks "Now".

## Competition-day checklist
- [ ] `source .venv/bin/activate`: the prompt shows `(.venv)`
- [ ] Practice: `python compete.py --start 2026-11-10 --end 2026-11-20 --rolling --hours 24`
- [ ] Paste their call: `python compete.py --url "..."`, plus `--forecast-from` / `--hours` if they give a start and length
- [ ] Note which model it says it's using (main or backup)
- [ ] App already running in a browser tab for the demo
