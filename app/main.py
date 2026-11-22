import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# app/main.py lives one folder below the project root, so add the root to the
# system path to be able to import from src/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.autoregressive import LATITUDE, LONGITUDE, get_current_weather, predict_autoregressive
from src.planner import (
    ACTIVITIES, CONFIDENCE_WIDTHS, add_uncertainty, best_windows, comfort_score, confidence_label,
    is_daylight, score_label, what_to_wear,
)


LOCATION_NAME = "East Lansing, MI"
LOCAL_TZ = "America/Detroit"
MAX_FORECAST_HOURS = 72
MODEL_FILES = {
    "Main model (16 variables)": "weather_model.pkl",
    "Backup model (no soil data)": "weather_model_nosoil.pkl",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def model_path_for(model_file):
    local_path = Path("models") / model_file
    return local_path if local_path.exists() else ROOT_DIR / "models" / model_file


@st.cache_resource
def _load_model(model_path, modified_time):
    """Load a trained model once and keep it in memory between reruns.

    modified_time is part of the cache key, so retraining (which rewrites the file)
    makes the app load the new model without a restart.
    """
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise RuntimeError(
            f"{model_path.name} was saved by an older version of this project. "
            "Run `python main.py --retrain` to create new models."
        )
    return bundle


def get_model(model_file="weather_model.pkl"):
    model_path = model_path_for(model_file)
    return _load_model(model_path, model_path.stat().st_mtime)


@st.cache_data(ttl=3600, show_spinner=False)
def load_current_weather():
    """Download the last 24 hours of observations from Open-Meteo (cached for 1 hour)."""
    return get_current_weather()


@st.cache_data(ttl=3600, show_spinner=False)
def make_forecast(model_file, current_weather, model_version):
    """Run the autoregressive model for the max horizon and return a DataFrame.

    We always predict the full 72 hours and slice it later, so moving the
    slider in the sidebar doesn't re-run the model. model_version (the model file's
    modified time) is only there so the cache resets when the model is retrained.
    """
    model = get_model(model_file)

    predictions = predict_autoregressive(model, current_weather, steps=MAX_FORECAST_HOURS)

    forecast = predictions.rename(columns={"temperature_2m": "temperature_c"})
    forecast["daylight"] = is_daylight(forecast["time"], LATITUDE, LONGITUDE)
    # Show times in local time (and drop the timezone so charts don't shift it)
    forecast["time"] = forecast["time"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    # 80% ranges and chance of rain, from the model's measured errors (see src/evaluate.py)
    if "uncertainty" in model:
        forecast = add_uncertainty(forecast, model["uncertainty"])

    return forecast


def has_uncertainty(forecast):
    return "rain_chance" in forecast.columns


def local_now():
    return pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)


def convert_temp(temp_c, unit):
    """Convert a Celsius temperature (or Series) to the chosen unit."""
    return temp_c * 9 / 5 + 32 if unit == "°F" else temp_c


def convert_temp_change(change_c, unit):
    """Convert a temperature *difference* (no +32 offset)."""
    return change_c * 9 / 5 if unit == "°F" else change_c


def format_value(value, suffix="", decimals=1):
    """Format a number for display, showing a dash if the value is missing."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}{suffix}"


def wind_direction_to_compass(degrees):
    if pd.isna(degrees):
        return ""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[int((degrees + 22.5) // 45) % 8]


def describe_conditions(row):
    """Turn the raw numbers into a simple text description with an emoji."""
    precipitation = row.get("precipitation")
    cloud_cover = row.get("cloud_cover")
    temperature = row.get("temperature_2m")

    if pd.notna(precipitation) and precipitation > 0:
        if pd.notna(temperature) and temperature <= 0:
            return "🌨️ Snow"
        return "🌧️ Rain"
    if pd.isna(cloud_cover):
        return "🌡️ Conditions unavailable"
    if cloud_cover >= 80:
        return "☁️ Cloudy"
    if cloud_cover >= 30:
        return "⛅ Partly cloudy"
    return "☀️ Clear"


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------

def show_sidebar():
    st.sidebar.header("⚙️ Settings")

    model_name = st.sidebar.radio(
        "Model", list(MODEL_FILES.keys()),
        help="The backup model skips the soil variables, which Open-Meteo's forecast API doesn't provide.",
    )
    hours = st.sidebar.slider(
        "Forecast length (hours)", min_value=6, max_value=MAX_FORECAST_HOURS, value=MAX_FORECAST_HOURS, step=6
    )
    unit = st.sidebar.radio("Temperature unit", ["°F", "°C"], horizontal=True)

    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh weather data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.caption("Weather data from the [Open-Meteo](https://open-meteo.com/) archive API.")

    return MODEL_FILES[model_name], hours, unit


def show_current_conditions(row, unit):
    obs_time = pd.to_datetime(row["date"]).tz_convert(LOCAL_TZ)

    st.subheader("Latest observed conditions")
    st.caption(f"Observed {obs_time:%A, %B %d at %I:%M %p} (local time)")
    st.markdown(f"### {describe_conditions(row)}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", format_value(convert_temp(row.get("temperature_2m"), unit), f" {unit}"))
    col2.metric("Feels like", format_value(convert_temp(row.get("apparent_temperature"), unit), f" {unit}"))
    col3.metric("Humidity", format_value(row.get("relative_humidity_2m"), "%", 0))
    col4.metric(
        "Wind",
        format_value(row.get("wind_speed_10m"), " km/h"),
        help=f"Gusts up to {format_value(row.get('wind_gusts_10m'), ' km/h')}",
    )
    col4.caption(f"From the {wind_direction_to_compass(row.get('wind_direction_10m'))}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Cloud cover", format_value(row.get("cloud_cover"), "%", 0))
    col6.metric("Precipitation", format_value(row.get("precipitation"), " mm"))
    col7.metric("Pressure", format_value(row.get("pressure_msl"), " hPa", 0))
    col8.metric("Dew point", format_value(convert_temp(row.get("dew_point_2m"), unit), f" {unit}"))


def show_forecast_summary(forecast, current_temp_c, unit):
    temps = forecast["temperature_c"]
    high_idx = temps.idxmax()
    low_idx = temps.idxmin()

    next_hour_c = temps.iloc[0]
    change_c = next_hour_c - current_temp_c if pd.notna(current_temp_c) else None

    st.subheader(f"Next {len(forecast)} hours")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Next hour",
        format_value(convert_temp(next_hour_c, unit), f" {unit}"),
        delta=None if change_c is None else f"{convert_temp_change(change_c, unit):+.1f} {unit}",
        help="Change compared to the latest observed temperature",
    )
    col2.metric("High", format_value(convert_temp(temps[high_idx], unit), f" {unit}"))
    col2.caption(f"{forecast.loc[high_idx, 'time']:%a %I %p}")
    col3.metric("Low", format_value(convert_temp(temps[low_idx], unit), f" {unit}"))
    col3.caption(f"{forecast.loc[low_idx, 'time']:%a %I %p}")
    col4.metric("Average", format_value(convert_temp(temps.mean(), unit), f" {unit}"))


def show_forecast_chart(forecast, unit):
    temp_label = f"Temperature ({unit})"
    chart_df = pd.DataFrame({
        "Time": forecast["time"],
        "Forecast": convert_temp(forecast["temperature_c"], unit).round(1),
    })

    st.subheader("Temperature forecast")
    base = alt.Chart(chart_df).encode(x=alt.X("Time:T", title="Time (local)"))
    layers = []

    if has_uncertainty(forecast):
        chart_df["Low"] = convert_temp(forecast["temperature_c_low"], unit).round(1)
        chart_df["High"] = convert_temp(forecast["temperature_c_high"], unit).round(1)
        base = alt.Chart(chart_df).encode(x=alt.X("Time:T", title="Time (local)"))
        layers.append(base.mark_area(opacity=0.25).encode(y=alt.Y("Low:Q", title=temp_label), y2="High:Q"))

    tooltip = [alt.Tooltip("Time:T", format="%a %I %p")] + [
        alt.Tooltip(f"{c}:Q", title=f"{c} ({unit})") for c in ["Forecast", "Low", "High"] if c in chart_df
    ]
    layers.append(base.mark_line().encode(
        y=alt.Y("Forecast:Q", title=temp_label, scale=alt.Scale(zero=False)), tooltip=tooltip))

    now = local_now()
    if chart_df["Time"].min() < now < chart_df["Time"].max():
        now_df = pd.DataFrame({"Time": [now], "label": ["Now"]})
        rule = alt.Chart(now_df).encode(x="Time:T")
        layers += [rule.mark_rule(strokeDash=[4, 4], color="#9aa0a6"),
                   rule.mark_text(align="left", dx=4, color="#9aa0a6").encode(text="label", y=alt.value(8))]

    st.altair_chart(alt.layer(*layers), width="stretch")
    if has_uncertainty(forecast):
        st.caption(
            "Shaded area: 80% range. In past forecasts made at this time of year, the real temperature "
            "landed inside it about 8 times out of 10. It widens with time because errors add up."
        )


def show_daily_summary(forecast, unit):
    daily = forecast.copy()
    daily["temp"] = convert_temp(daily["temperature_c"], unit)
    daily["Day"] = daily["time"].dt.strftime("%a, %b %d")
    daily["date"] = daily["time"].dt.date

    summary = (
        daily.groupby(["date", "Day"], sort=True)["temp"]
        .agg(High="max", Low="min", Average="mean", Hours="count")
        .reset_index()
        .drop(columns="date")
    )

    st.subheader("Daily summary")
    temp_format = f"%.1f {unit}"
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "High": st.column_config.NumberColumn(format=temp_format),
            "Low": st.column_config.NumberColumn(format=temp_format),
            "Average": st.column_config.NumberColumn(format=temp_format),
            "Hours": st.column_config.NumberColumn(help="Forecast hours that fall on this day"),
        },
    )


def show_hourly_table(forecast, unit):
    hourly = pd.DataFrame({
        "Time": forecast["time"].dt.strftime("%a %b %d, %I:%M %p"),
        f"Temperature ({unit})": convert_temp(forecast["temperature_c"], unit).round(1),
    })
    if has_uncertainty(forecast):
        hourly[f"80% range ({unit})"] = [
            f"{convert_temp(lo, unit):.0f} to {convert_temp(hi, unit):.0f}"
            for lo, hi in zip(forecast["temperature_c_low"], forecast["temperature_c_high"])
        ]
        hourly[f"Feels like ({unit})"] = convert_temp(forecast["apparent_temperature"], unit).round(1)
        hourly["Chance of rain"] = (forecast["rain_chance"] * 100).round().astype(int).astype(str) + "%"
        hourly["Wind (km/h)"] = forecast["wind_speed_10m"].round(0)

    with st.expander("Hourly forecast table"):
        st.dataframe(hourly, hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Download forecast as CSV",
            data=hourly.to_csv(index=False),
            file_name="weather_forecast.csv",
            mime="text/csv",
        )


def show_about():
    with st.expander("How does this work?"):
        st.markdown(
            """
            1. The app downloads the last 24 hours of observations from the Open-Meteo archive API.
            2. From that window it builds features: the current value of each weather variable, how much it
               changed over the last 1, 3 and 23 hours, its 24-hour average, and the time of day and year.
            3. XGBoost models (one per variable, wrapped in a `MultiOutputRegressor`) predict how much each
               variable will change in the next hour. They were trained on hourly East Lansing data
               from 2016 to the most recent week.
            4. The prediction is appended to the window and the oldest hour dropped, then the model predicts
               the following hour, and so on for 72 hours. This is called **autoregressive** forecasting.

            Because each step builds on the previous prediction, small errors add up, so forecasts further
            in the future are less reliable.

            **Ranges and chance of rain.** The model was run from every hour of 2026 (data it never trained
            on) and compared with what really happened. The shaded 80% range is built from those errors,
            separately for each season, because winter and spring are harder to forecast. The chance of rain
            is how often it actually rained when the model predicted a similar amount. Beyond about 6 hours
            the model's rain forecast was no better than the seasonal average, so after that the app shows
            the typical chance for this time of year.
            """
        )


# ---------------------------------------------------------------------------
# Outdoor planner tab
# ---------------------------------------------------------------------------

SCHEDULE_TIMES = [f"{h % 12 or 12}:00 {'AM' if h < 12 else 'PM'}" for h in range(6, 24)]


def hour_from_label(label):
    hour = int(label.split(":")[0]) % 12
    return hour + 12 if label.endswith("PM") else hour


def temp_range_text(low_c, high_c, unit):
    return f"{convert_temp(low_c, unit):.0f}-{convert_temp(high_c, unit):.0f} {unit}"


def show_schedule(upcoming, unit):
    st.subheader("🎒 My schedule")
    st.caption("Pick the times you're usually outside (walking to class, heading to work...) to see "
               "what to expect and what to wear each day.")

    col1, col2 = st.columns([3, 1])
    times = col1.multiselect("Times I'm outside", SCHEDULE_TIMES, default=["8:00 AM", "12:00 PM", "5:00 PM"])
    mode = col2.radio("Getting around by", ["Walk to class", "Bike"], horizontal=False)
    if not times:
        st.info("Pick at least one time above.")
        return

    hours = {hour_from_label(t) for t in times}
    rows = upcoming[upcoming["time"].dt.hour.isin(hours)]
    if rows.empty:
        st.info("None of those times are in the forecast window. Try other times.")
        return

    scores = comfort_score(rows["apparent_temperature"], rows["wind_speed_10m"], rows["rain_chance"], mode,
                           rows["daylight"])
    table = pd.DataFrame({
        "When": rows["time"].dt.strftime("%a %b %d, %I %p"),
        "Feels like": [f"{convert_temp(v, unit):.0f} {unit}" for v in rows["apparent_temperature"]],
        "Likely range": [temp_range_text(lo, hi, unit) for lo, hi in
                         zip(rows["apparent_temperature_low"], rows["apparent_temperature_high"])],
        "Rain": (rows["rain_chance"] * 100).round().astype(int).astype(str) + "%",
        "Wind": rows["wind_speed_10m"].round().astype(int).astype(str) + " km/h",
        "Conditions": [score_label(sc) for sc in scores],
        "Daylight": ["☀️" if d else "🌙" for d in rows["daylight"]],
        "What to wear": [what_to_wear(f, r, w, dark=not d) for f, r, w, d in
                         zip(rows["apparent_temperature"], rows["rain_chance"], rows["wind_speed_10m"],
                             rows["daylight"])],
        "Confidence": [confidence_label(lo, hi) for lo, hi in
                       zip(rows["apparent_temperature_low"], rows["apparent_temperature_high"])],
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Likely range": st.column_config.TextColumn(help="80% range for the feels-like temperature"),
            "Confidence": st.column_config.TextColumn(
                help=f"High: the 80% range is under {CONFIDENCE_WIDTHS[0]} °C wide. "
                     f"Medium: under {CONFIDENCE_WIDTHS[1]} °C. Low: wider."),
        },
    )


def show_best_times(upcoming, unit):
    st.subheader("🏃 Find the best time")
    col1, col2, col3 = st.columns(3)
    activity = col1.selectbox("Activity", list(ACTIVITIES), index=1)
    duration = col2.slider("How long (hours)", 1, 3, 1)
    earliest, latest = col3.slider("Between", 5, 23, (7, 21), format="%d:00")

    windows = best_windows(upcoming, activity, duration, earliest, latest)
    if windows.empty:
        st.info("No time slots fit those settings in the forecast window.")
        return

    st.caption("Best time slot on each day. ⭐ = best overall.")
    cols = st.columns(len(windows))
    for col, (_, w) in zip(cols, windows.iterrows()):
        with col.container(border=True):
            star = "⭐ " if w["is_best"] else ""
            st.markdown(f"**{star}{w['start']:%A}, {w['start']:%I %p} – {w['end']:%I %p}**")
            st.metric("Comfort score", f"{w['score']:.0f}/100", help="See 'How the scores work' below")
            st.caption(score_label(w["score"]))
            st.write(f"Feels like {convert_temp(w['feels_like_c'], unit):.0f} {unit} · "
                     f"rain {w['rain_chance'] * 100:.0f}% · wind up to {w['wind_kmh']:.0f} km/h")
            if w["dark"]:
                st.write("🌙 Partly or fully after dark")
            st.write(f"👕 {what_to_wear(w['feels_like_c'], w['rain_chance'], w['wind_kmh'], w['dark'])}")
            st.caption(f"Forecast confidence: {w['confidence']}")

    chart_df = pd.DataFrame({
        "Time": upcoming["time"],
        "Score": comfort_score(upcoming["apparent_temperature"], upcoming["wind_speed_10m"],
                               upcoming["rain_chance"], activity, upcoming["daylight"]).round(0),
    })
    chart_df["Conditions"] = [score_label(sc).split(" ", 1)[1] for sc in chart_df["Score"]]
    in_hours = chart_df["Time"].dt.hour.between(earliest, latest - 1)
    colors = alt.Scale(domain=["Great", "Good", "Fair", "Poor"],
                       range=["#2e9e5b", "#e0b422", "#ec8a2c", "#d64545"])
    chart = alt.Chart(chart_df[in_hours]).mark_bar().encode(
        x=alt.X("Time:T", title="Time (local)"),
        y=alt.Y("Score:Q", title=f"{activity} score", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Conditions:N", scale=colors),
        tooltip=[alt.Tooltip("Time:T", format="%a %I %p"), "Score:Q", "Conditions:N"],
    )
    st.altair_chart(chart, width="stretch")


def show_score_explainer():
    with st.expander("How the scores work"):
        rows = [
            f"| {name} | {a['ideal'][0]} to {a['ideal'][1]} °C | above {a['wind_start']} km/h, "
            f"-{a['wind_penalty']} per km/h | -{a['rain_penalty']} per % chance | -{a['dark_penalty']} |"
            for name, a in ACTIVITIES.items()
        ]
        st.markdown(
            "Each hour starts at 100 points and loses points for being uncomfortable:\n\n"
            "| Activity | Comfortable feels-like | Wind penalty | Rain penalty | After dark |\n"
            "|---|---|---|---|---|\n"
            + "\n".join(rows)
            + "\n\nOutside the comfortable range it loses 4 points per °C. Running has a cooler range "
            "because you warm up; biking minds wind more. 80+ is Great, 60+ Good, 40+ Fair.\n\n"
            "**Confidence** comes from how wide the 80% feels-like range is at that hour. "
            "**Chance of rain** is only better than the seasonal average for the first ~6 hours of the "
            "forecast, so treat later values as \"typical for this time of year\"."
        )


def show_planner(forecast, unit):
    if not has_uncertainty(forecast):
        st.warning("This model file has no measured forecast errors yet. Run `python main.py` once, "
                   "then refresh.")
        return

    now = local_now().floor("h")
    upcoming = forecast[forecast["time"] > now].reset_index(drop=True)
    if upcoming.empty:
        st.info("The forecast doesn't reach past the current time. Try refreshing the weather data.")
        return

    st.caption(f"Planning with the forecast from now until {upcoming['time'].iloc[-1]:%A %I %p}. "
               "The forecast starts at the latest archived observation, which is usually a day old, so "
               "earlier hours are already behind us.")
    show_schedule(upcoming, unit)
    st.divider()
    show_best_times(upcoming, unit)
    show_score_explainer()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Weather Forecaster", page_icon="🌤️", layout="wide")

    st.title("🌤️ AI Weather Forecaster")
    st.caption(f"Hourly forecast and outdoor planner for {LOCATION_NAME}, powered by a machine learning model")

    model_file, hours, unit = show_sidebar()

    with st.spinner("Downloading the latest weather data..."):
        try:
            current_weather = load_current_weather()
        except Exception as e:
            st.error(f"Couldn't download weather data from Open-Meteo: {e}")
            st.stop()

    # The newest observation is the last row
    row = current_weather.iloc[-1]
    current_temp_c = row.get("temperature_2m")

    if pd.isna(current_temp_c):
        st.warning(
            "The latest observed temperature is missing from the API response, "
            "so the forecast below may not be reliable. Try refreshing later."
        )

    with st.spinner("Running the model..."):
        try:
            model_version = model_path_for(model_file).stat().st_mtime
            full_forecast = make_forecast(model_file, current_weather, model_version)
        except Exception as e:
            st.error(f"Something went wrong while making the forecast: {e}")
            st.stop()
    forecast = full_forecast.head(hours)

    forecast_tab, planner_tab = st.tabs(["📈 Forecast", "🎒 Outdoor planner"])
    with forecast_tab:
        show_current_conditions(row, unit)
        st.divider()
        show_forecast_summary(forecast, current_temp_c, unit)
        show_forecast_chart(forecast, unit)
        show_daily_summary(forecast, unit)
        show_hourly_table(forecast, unit)
        show_about()
    with planner_tab:
        show_planner(full_forecast, unit)


if __name__ == "__main__":
    main()
