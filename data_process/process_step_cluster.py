import os
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(BASE_DIR, "processed_data", "station_level_summary.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "station_with_clusters.csv")
EDA_OUTPUT = os.path.join(BASE_DIR, "eda_output")
os.makedirs(EDA_OUTPUT, exist_ok=True)

df = pd.read_csv(SUMMARY_PATH)
df = df.dropna(subset=["latitude", "longitude"])
coords = df[["latitude", "longitude"]]


inertias = []
K_range = range(2, 15)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(coords)
    inertias.append(km.inertia_)


kneedle = KneeLocator(K_range, inertias, curve="convex", direction="decreasing")
optimal_k = kneedle.elbow if kneedle.elbow is not None else 6  # fallback


plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker="o", label="Inertia")
if kneedle.elbow:
    plt.axvline(optimal_k, color="red", linestyle="--", label=f"Elbow: k={optimal_k}")
plt.title("Elbow Method with Kneedle Detection")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(K_range)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EDA_OUTPUT, "elbow_curve_kmeans_auto.png"))
plt.close()

print(f"自动识别的最佳聚类数：k = {optimal_k}")


kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["cluster_id"] = kmeans.fit_predict(coords)


df.to_csv(OUTPUT_PATH, index=False)
print(f"聚类已完成，共 {optimal_k} 个 cluster，保存至：{OUTPUT_PATH}")


plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x="longitude", y="latitude", hue="cluster_id", palette="tab10", alpha=0.8)
plt.title(f"Station Clusters by Location (k={optimal_k})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(os.path.join(EDA_OUTPUT, "station_location_clusters_auto.png"))
plt.close()
print("聚类分布图已保存：station_location_clusters_auto.png")
