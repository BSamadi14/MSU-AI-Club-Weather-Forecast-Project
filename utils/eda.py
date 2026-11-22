import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


data_path = Path("data/weather_data.csv")
df = pd.read_csv(data_path)


# Look at the first day of temperature_2m
df_day1 = df[df["date"] < "2026-09-23"]

print(len(df_day1))

plt.plot(df_day1["date"], df_day1["temperature_2m"])
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Temperature 2m for the first day")
plt.show()


# Look at the first week of temperature_2m
df_week1 = df[df["date"] < "2016-01-08"]

print(len(df_week1))

plt.plot(df_week1["date"], df_week1["temperature_2m"])
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Temperature 2m for the first week")
plt.show()


# Look at the first month of temperature_2m
df_month1 = df[df["date"] < "2016-02-01"]

print(len(df_month1))

plt.plot(df_month1["date"], df_month1["temperature_2m"])
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Temperature 2m for the first month")
plt.show()