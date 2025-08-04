import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Arial"


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "processed_data", "pickup_weather_coords.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(INPUT_PATH)
df["start_hour"] = pd.to_datetime(df["start_hour"], errors="coerce")


df["hour"] = df["start_hour"].dt.hour
df["weekday"] = df["start_hour"].dt.dayofweek  # Monday=0, Sunday=6
df["month"] = df["start_hour"].dt.month
df["date"] = df["start_hour"].dt.date
df["is_weekend"] = df["weekday"].isin([5, 6])


hourly_avg = df.groupby("hour")["pickup_count"].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(data=hourly_avg, x="hour", y="pickup_count", marker="o")
plt.title("Average Bike Pickups by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Pickups")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hourly_pattern.png"))
plt.close()


heatmap_data = df.pivot_table(index="weekday", columns="hour", values="pickup_count", aggfunc="mean")
plt.figure(figsize=(12, 5))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False)
plt.title("Heatmap: Weekday vs Hour")
plt.xlabel("Hour")
plt.ylabel("Weekday (0=Mon)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "weekday_hour_heatmap.png"))
plt.close()


daily_trend = df.groupby("date")["pickup_count"].sum().reset_index()
plt.figure(figsize=(14, 5))
sns.lineplot(data=daily_trend, x="date", y="pickup_count")
plt.title("Daily Total Bike Pickups")
plt.xlabel("Date")
plt.ylabel("Total Pickups")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "daily_demand_trend.png"))
plt.close()


plt.figure(figsize=(10, 6))
sns.boxplot(x="month", y="pickup_count", data=df)
plt.title("Monthly Distribution of Bike Pickups")
plt.xlabel("Month")
plt.ylabel("Pickup Count per Hour")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_boxplot.png"))
plt.close()


weekend_stats = df.groupby("is_weekend")["pickup_count"].mean().reset_index()
weekend_stats["Label"] = weekend_stats["is_weekend"].map({True: "Weekend", False: "Weekday"})
plt.figure(figsize=(6, 5))
sns.barplot(data=weekend_stats, x="Label", y="pickup_count", palette="Set2")
plt.title("Average Pickups: Weekday vs Weekend")
plt.ylabel("Average Pickups per Hour")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "weekday_vs_weekend.png"))
plt.close()

print("时间维度 EDA 图表全部已生成并保存在 eda_output 文件夹。")
