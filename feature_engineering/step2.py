import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_for_model")


pickup_path = os.path.join(DATA_DIR, "pickup_features.csv")
dropoff_path = os.path.join(DATA_DIR, "dropoff_features.csv")


pickup_df = pd.read_csv(pickup_path)
dropoff_df = pd.read_csv(dropoff_path)


pickup_cols_to_drop = [
    "start_hour", "conditions", "Start station", "timestamp", "date"
]
dropoff_cols_to_drop = [
    "end_hour", "conditions", "End station", "timestamp", "date"
]


pickup_df.drop(columns=pickup_cols_to_drop, inplace=True, errors='ignore')
dropoff_df.drop(columns=dropoff_cols_to_drop, inplace=True, errors='ignore')


pickup_df.fillna(0, inplace=True)
dropoff_df.fillna(0, inplace=True)


pickup_df.to_csv(os.path.join(DATA_DIR, "pickup_features_2.csv"), index=False)
dropoff_df.to_csv(os.path.join(DATA_DIR, "dropoff_features_2.csv"), index=False)

print("特征清洗完成，文件已保存为 pickup_features_2.csv 和 dropoff_features_2.csv。")
