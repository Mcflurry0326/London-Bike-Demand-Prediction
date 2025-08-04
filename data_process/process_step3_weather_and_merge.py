import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEATHER_PATH = os.path.join(BASE_DIR, "data", "weather_data", "London_weather_data_hourly.csv")
WEATHER_CLEAN_PATH = os.path.join(BASE_DIR, "processed_data", "cleaned_weather_hourly.csv")

PICKUP_PATH = os.path.join(BASE_DIR, "processed_data", "hourly_pickup_count.csv")
DROPOFF_PATH = os.path.join(BASE_DIR, "processed_data", "hourly_dropoff_count.csv")

PICKUP_OUTPUT = os.path.join(BASE_DIR, "processed_data", "hourly_pickup_with_weather.csv")
DROPOFF_OUTPUT = os.path.join(BASE_DIR, "processed_data", "hourly_dropoff_with_weather.csv")

weather_df = pd.read_csv(WEATHER_PATH)


weather_df = weather_df[[
    "datetime", "temp", "humidity", "precip", "windspeed", "cloudcover", "conditions"
]].copy()
weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])


weather_df.to_csv(WEATHER_CLEAN_PATH, index=False)
print(f"已保存清洗天气数据至: {WEATHER_CLEAN_PATH}")


pickup_df = pd.read_csv(PICKUP_PATH)
dropoff_df = pd.read_csv(DROPOFF_PATH)

pickup_df["start_hour"] = pd.to_datetime(pickup_df["start_hour"])
dropoff_df["end_hour"] = pd.to_datetime(dropoff_df["end_hour"])


pickup_merged = pickup_df.merge(weather_df, how="left", left_on="start_hour", right_on="datetime")
dropoff_merged = dropoff_df.merge(weather_df, how="left", left_on="end_hour", right_on="datetime")


pickup_merged = pickup_merged.drop(columns=["datetime"])
dropoff_merged = dropoff_merged.drop(columns=["datetime"])


pickup_merged.to_csv(PICKUP_OUTPUT, index=False)
dropoff_merged.to_csv(DROPOFF_OUTPUT, index=False)
print(f"出发数据合并完成: {PICKUP_OUTPUT}")
print(f"到达数据合并完成: {DROPOFF_OUTPUT}")
