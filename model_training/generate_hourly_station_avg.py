
import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_for_model")
OUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)


pickup_df = pd.read_csv(os.path.join(DATA_DIR, "pickup_features_2.csv"))
dropoff_df = pd.read_csv(os.path.join(DATA_DIR, "dropoff_features_2.csv"))


if "hour" not in pickup_df.columns or "hour" not in dropoff_df.columns:
    raise ValueError("数据中缺少 'hour' 字段，请确认已完成特征工程。")

pickup_avg = pickup_df.groupby(["station", "hour"])["pickup_count"].mean().reset_index()
pickup_avg.rename(columns={"pickup_count": "avg_pickup_count"}, inplace=True)

dropoff_avg = dropoff_df.groupby(["station", "hour"])["dropoff_count"].mean().reset_index()
dropoff_avg.rename(columns={"dropoff_count": "avg_dropoff_count"}, inplace=True)


pickup_avg.to_csv(os.path.join(OUT_DIR, "avg_hourly_pickup_by_station.csv"), index=False)
dropoff_avg.to_csv(os.path.join(OUT_DIR, "avg_hourly_dropoff_by_station.csv"), index=False)

print("每站点每小时平均值已生成并保存至 models 文件夹。")
