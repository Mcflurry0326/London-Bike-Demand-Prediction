import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MERGED_PATH = os.path.join(BASE_DIR, "processed_data", "merged_tfl_data.csv")
CLEANED_PATH = os.path.join(BASE_DIR, "processed_data", "cleaned_tfl_data.csv")
PICKUP_PATH = os.path.join(BASE_DIR, "processed_data", "hourly_pickup_count.csv")
DROPOFF_PATH = os.path.join(BASE_DIR, "processed_data", "hourly_dropoff_count.csv")


df = pd.read_csv(MERGED_PATH)


cols_to_drop = [
    "Number", "Start station number", "End station number",
    "Bike number", "Bike model", "Total duration", "Total duration (ms)"
]
df = df.drop(columns=cols_to_drop, errors="ignore")


df["Start date"] = pd.to_datetime(df["Start date"], errors="coerce")
df["End date"] = pd.to_datetime(df["End date"], errors="coerce")


df["start_hour"] = df["Start date"].dt.floor("H")
df["end_hour"] = df["End date"].dt.floor("H")


df.to_csv(CLEANED_PATH, index=False)
print(f"已保存清洗数据至: {CLEANED_PATH}")


pickup_df = (
    df.groupby(["Start station", "start_hour"])
    .size()
    .reset_index(name="pickup_count")
)
pickup_df.to_csv(PICKUP_PATH, index=False)
print(f"已保存出发统计至: {PICKUP_PATH}")


dropoff_df = (
    df.groupby(["End station", "end_hour"])
    .size()
    .reset_index(name="dropoff_count")
)
dropoff_df.to_csv(DROPOFF_PATH, index=False)
print(f"已保存到达统计至: {DROPOFF_PATH}")
