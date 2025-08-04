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


df = df.dropna(subset=["pickup_count", "temp", "humidity", "precip", "windspeed", "conditions"])


def draw_scatter(x_col, title, filename, color):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y="pickup_count", alpha=0.2, s=10, color=color)
    plt.title(title)
    plt.xlabel(x_col.capitalize())
    plt.ylabel("Pickup Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

draw_scatter("temp", "Temperature vs Pickup Count", "scatter_temp.png", "orange")
draw_scatter("humidity", "Humidity vs Pickup Count", "scatter_humidity.png", "teal")
draw_scatter("precip", "Precipitation vs Pickup Count", "scatter_precip.png", "blue")
draw_scatter("windspeed", "Wind Speed vs Pickup Count", "scatter_windspeed.png", "purple")


top_conditions = df["conditions"].value_counts().head(10).index
df_top = df[df["conditions"].isin(top_conditions)]
cond_avg = df_top.groupby("conditions")["pickup_count"].mean().sort_values()

plt.figure(figsize=(10, 5))
sns.barplot(x=cond_avg.values, y=cond_avg.index, palette="viridis")
plt.title("Average Pickups by Weather Condition")
plt.xlabel("Avg Pickups")
plt.ylabel("Condition")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "barplot_by_condition.png"))
plt.close()

df["temp_bin"] = pd.cut(df["temp"], bins=[-5, 5, 10, 15, 20, 25, 30, 40])
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="temp_bin", y="pickup_count")
plt.title("Pickup Count by Temperature Range")
plt.xlabel("Temperature Range (°C)")
plt.ylabel("Pickup Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_temp_bins.png"))
plt.close()

df["precip_bin"] = pd.cut(df["precip"], bins=[-0.1, 0.1, 1, 3, 5, 10, 100])
heatmap_data = df.pivot_table(index="hour", columns="precip_bin", values="pickup_count", aggfunc="mean")

plt.figure(figsize=(10, 5))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False)
plt.title("Avg Pickups by Hour and Precipitation Level")
plt.xlabel("Precipitation Bin")
plt.ylabel("Hour of Day")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_precip_hour.png"))
plt.close()

print("天气维度 EDA 图表已全部生成，位于 eda_output 文件夹。")
