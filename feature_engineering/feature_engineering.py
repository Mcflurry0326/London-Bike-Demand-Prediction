import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROC_DIR = os.path.join(BASE_DIR, "processed_data")
OUT_DIR = os.path.join(BASE_DIR, "data_for_model")

pickup_file = os.path.join(PROC_DIR, "pickup_weather_coords.csv")
dropoff_file = os.path.join(PROC_DIR, "dropoff_weather_coords.csv")
cluster_file = os.path.join(PROC_DIR, "station_with_clusters.csv")


pickup_df = pd.read_csv(pickup_file)
dropoff_df = pd.read_csv(dropoff_file)
clusters = pd.read_csv(cluster_file)


def build_features(df, time_col, station_col, count_col, clusters, direction="pickup"):
    df = df.copy()
    
   
    df["timestamp"] = pd.to_datetime(df[time_col])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].apply(lambda x: 1 if 7 <= x <= 9 or 17 <= x <= 19 else 0)
    df["date"] = df["timestamp"].dt.date

    df = df.merge(clusters[["station", "cluster_id"]], left_on=station_col, right_on="station", how="left")

   
    cluster_hourly_avg = df.groupby(["cluster_id", "timestamp"])[count_col].mean().reset_index()
    cluster_hourly_avg.rename(columns={count_col: f"cluster_hourly_avg_{direction}"}, inplace=True)
    df = df.merge(cluster_hourly_avg, on=["cluster_id", "timestamp"], how="left")

    
    df = df.sort_values([station_col, "timestamp"])
    df[f"{count_col}_lag_1h"] = df.groupby(station_col)[count_col].shift(1)
    df[f"{count_col}_rolling_3h_mean"] = (
        df.groupby(station_col)[count_col].rolling(3).mean().reset_index(level=0, drop=True)
    )
    df[f"{count_col}_rolling_6h_std"] = (
        df.groupby(station_col)[count_col].rolling(6).std().reset_index(level=0, drop=True)
    )
    df[f"{count_col}_cumsum_day"] = df.groupby([station_col, "date"])[count_col].cumsum()

    return df


pickup_features = build_features(
    pickup_df,
    time_col="start_hour",
    station_col="Start station",
    count_col="pickup_count",
    clusters=clusters,
    direction="pickup"
)


dropoff_features = build_features(
    dropoff_df,
    time_col="end_hour",
    station_col="End station",
    count_col="dropoff_count",
    clusters=clusters,
    direction="dropoff"
)

os.makedirs(OUT_DIR, exist_ok=True)
pickup_features.to_csv(os.path.join(OUT_DIR, "pickup_features.csv"), index=False)
dropoff_features.to_csv(os.path.join(OUT_DIR, "dropoff_features.csv"), index=False)

print("特征工程完成，特征文件已保存至 data_for_model 文件夹。")
