import os
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")


weekday_df = pd.read_csv(os.path.join(PROCESSED_DIR, "pickup_avg_weekday.csv"))
weekend_df = pd.read_csv(os.path.join(PROCESSED_DIR, "pickup_avg_weekend.csv"))


weekday_sum = weekday_df.groupby("station")["pickup_count"].sum().reset_index(name="weekday_sum")
weekend_sum = weekend_df.groupby("station")["pickup_count"].sum().reset_index(name="weekend_sum")


merged = pd.merge(weekday_sum, weekend_sum, on="station", how="outer").fillna(0)


merged["type"] = merged.apply(
    lambda row: "weekday" if row["weekday_sum"] > row["weekend_sum"] else "weekend", axis=1
)

output_path = os.path.join(PROCESSED_DIR, "station_weektype_label.csv")
merged[["station", "type"]].to_csv(output_path, index=False)

print(f"每个站点的工作日/周末类型已保存到：{output_path}")
