import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


PICKUP_PATH = os.path.join(DATA_DIR, "pickup_weather_coords.csv")
DROPOFF_PATH = os.path.join(DATA_DIR, "dropoff_weather_coords.csv")
COORD_PATH = os.path.join(DATA_DIR, "station_coordinates_google.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "station_level_summary.csv")


pickup = pd.read_csv(PICKUP_PATH)
dropoff = pd.read_csv(DROPOFF_PATH)
coords = pd.read_csv(COORD_PATH)


for df in [pickup, dropoff, coords]:
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('"', '')


pickup_total = pickup.groupby("Start station")["pickup_count"].sum().reset_index()
pickup_total.columns = ["station", "total_pickup"]

dropoff_total = dropoff.groupby("End station")["dropoff_count"].sum().reset_index()
dropoff_total.columns = ["station", "total_dropoff"]


counts = pd.merge(pickup_total, dropoff_total, on="station", how="outer")
counts["total_pickup"] = counts["total_pickup"].fillna(0)
counts["total_dropoff"] = counts["total_dropoff"].fillna(0)
counts["net_flow"] = counts["total_pickup"] - counts["total_dropoff"]


summary = pd.merge(counts, coords, on="station", how="left")
summary = summary.dropna(subset=["latitude", "longitude"])
summary.to_csv(SUMMARY_PATH, index=False)



sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Arial"


top_pickup = summary.nlargest(20, "total_pickup")
plt.figure(figsize=(10, 6))
sns.barplot(data=top_pickup, y="station", x="total_pickup", palette="Blues_r")
plt.title("Top 20 Stations by Total Pickups")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top20_pickup_barplot.png"))
plt.close()


top_dropoff = summary.nlargest(20, "total_dropoff")
plt.figure(figsize=(10, 6))
sns.barplot(data=top_dropoff, y="station", x="total_dropoff", palette="Greens_r")
plt.title("Top 20 Stations by Total Dropoffs")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top20_dropoff_barplot.png"))
plt.close()


top_netflow = pd.concat([
    summary.nlargest(10, "net_flow"),
    summary.nsmallest(10, "net_flow")
])
plt.figure(figsize=(10, 7))
sns.barplot(data=top_netflow, y="station", x="net_flow", palette="coolwarm")
plt.axvline(0, color="gray", linestyle="--")
plt.title("Top ±10 Stations by Net Flow")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top20_netflow_barplot.png"))
plt.close()

plt.figure(figsize=(10, 8))
sns.scatterplot(data=summary, x="longitude", y="latitude", size="total_pickup",
                hue="net_flow", palette="coolwarm", alpha=0.7, legend=False)
plt.title("Station Scatter: Size=Pickup, Color=Net Flow")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "station_scatter_netflow.png"))
plt.close()


plt.figure(figsize=(10, 8))
sns.kdeplot(
    data=summary,
    x="longitude", y="latitude",
    weights=summary["total_pickup"],
    cmap="Reds", fill=True, thresh=0.01, levels=100
)
plt.scatter(summary["longitude"], summary["latitude"], s=5, alpha=0.2, color="black")
plt.title("Pickup Volume Density (KDE Heatmap)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "station_pickup_heatmap.png"))
plt.close()


plt.figure(figsize=(10, 8))
plt.hexbin(summary["longitude"], summary["latitude"],
           C=summary["total_pickup"], gridsize=30, cmap="Oranges", bins="log")
plt.colorbar(label="Log(Total Pickup)")
plt.title("Pickup Volume Hexbin Map")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pickup_2d_grid_heatmap.png"))
plt.close()


m = folium.Map(location=[51.5074, -0.1278], zoom_start=12)
heat_data = summary[["latitude", "longitude", "total_pickup"]].values.tolist()
HeatMap(heat_data, radius=12, max_zoom=13).add_to(m)
m.save(os.path.join(OUTPUT_DIR, "pickup_heatmap.html"))

print("完整空间 EDA（含 summary 构建与图表）已生成！")
