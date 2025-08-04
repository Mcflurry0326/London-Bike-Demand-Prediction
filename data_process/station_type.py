import os
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

cluster_path = os.path.join(PROCESSED_DIR, "station_with_clusters.csv")
type_path = os.path.join(PROCESSED_DIR, "station_weektype_label.csv")


cluster_df = pd.read_csv(cluster_path)
type_df = pd.read_csv(type_path)


merged_df = cluster_df.merge(type_df, on="station", how="left")

output_path = os.path.join(PROCESSED_DIR, "station_with_clusters.csv")  # 覆盖原文件
merged_df.to_csv(output_path, index=False)
print(f"已更新 station_with_clusters.csv，加入 type 字段")
