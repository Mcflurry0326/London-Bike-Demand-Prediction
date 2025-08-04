import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKUP_PATH = os.path.join(BASE_DIR, "processed_data", "hourly_pickup_with_weather.csv")
DROPOFF_PATH = os.path.join(BASE_DIR, "processed_data", "hourly_dropoff_with_weather.csv")
COORD_PATH = os.path.join(BASE_DIR, "processed_data", "station_coordinates_google.csv")

PICKUP_OUTPUT = os.path.join(BASE_DIR, "processed_data", "pickup_weather_coords.csv")
DROPOFF_OUTPUT = os.path.join(BASE_DIR, "processed_data", "dropoff_weather_coords.csv")


pickup_df = pd.read_csv(PICKUP_PATH)
dropoff_df = pd.read_csv(DROPOFF_PATH)
coord_df = pd.read_csv(COORD_PATH)


pickup_merged = pickup_df.merge(coord_df, how="left", left_on="Start station", right_on="station")
dropoff_merged = dropoff_df.merge(coord_df, how="left", left_on="End station", right_on="station")

pickup_merged = pickup_merged.drop(columns=["station"])
dropoff_merged = dropoff_merged.drop(columns=["station"])

pickup_merged.to_csv(PICKUP_OUTPUT, index=False)
dropoff_merged.to_csv(DROPOFF_OUTPUT, index=False)

print(f"出发数据 + 天气 + 经纬度 已保存至: {PICKUP_OUTPUT}")
print(f"到达数据 + 天气 + 经纬度 已保存至: {DROPOFF_OUTPUT}")
