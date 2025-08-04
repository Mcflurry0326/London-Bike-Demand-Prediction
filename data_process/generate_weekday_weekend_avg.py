import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

PICKUP_FILE = os.path.join(PROCESSED_DIR, "hourly_pickup_count.csv")
DROPOFF_FILE = os.path.join(PROCESSED_DIR, "hourly_dropoff_count.csv")

PICKUP_WEEKDAY_OUT = os.path.join(PROCESSED_DIR, "pickup_avg_weekday.csv")
PICKUP_WEEKEND_OUT = os.path.join(PROCESSED_DIR, "pickup_avg_weekend.csv")
DROPOFF_WEEKDAY_OUT = os.path.join(PROCESSED_DIR, "dropoff_avg_weekday.csv")
DROPOFF_WEEKEND_OUT = os.path.join(PROCESSED_DIR, "dropoff_avg_weekend.csv")


def generate_weektype_avg(file_path, station_col, time_col, count_col, output_prefix):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    df[time_col] = pd.to_datetime(df[time_col])  
    df["hour"] = df[time_col].dt.hour
    df["is_weekend"] = df[time_col].dt.weekday >= 5
    df.rename(columns={station_col: "station", count_col: "count"}, inplace=True)

   
    weekday_avg = (
        df[df["is_weekend"] == False]
        .groupby(["station", "hour"])["count"]
        .mean()
        .reset_index()
        .rename(columns={"count": f"{output_prefix}_count"})
    )
    weekday_avg.to_csv(os.path.join(PROCESSED_DIR, f"{output_prefix}_avg_weekday.csv"), index=False)

    weekend_avg = (
        df[df["is_weekend"] == True]
        .groupby(["station", "hour"])["count"]
        .mean()
        .reset_index()
        .rename(columns={"count": f"{output_prefix}_count"})
    )
    weekend_avg.to_csv(os.path.join(PROCESSED_DIR, f"{output_prefix}_avg_weekend.csv"), index=False)

    print(f"{output_prefix} 工作日与周末平均小时数据已保存至 processed_data/ 文件夹")


if __name__ == "__main__":
    generate_weektype_avg(
        file_path=PICKUP_FILE,
        station_col="Start station",
        time_col="start_hour",
        count_col="pickup_count",
        output_prefix="pickup"
    )

    generate_weektype_avg(
        file_path=DROPOFF_FILE,
        station_col="End station",
        time_col="end_hour",
        count_col="dropoff_count",
        output_prefix="dropoff"
    )
