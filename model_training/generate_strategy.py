
import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STRATEGY_PATH = os.path.join(MODEL_DIR, "strategy_by_station.json")
CLUSTER_MAP_PATH = os.path.join(BASE_DIR, "processed_data", "station_with_clusters.csv")


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)

global_metrics = load_metrics(os.path.join(MODEL_DIR, "global", "metrics.json"))
cluster_metrics = load_metrics(os.path.join(MODEL_DIR, "cluster", "metrics.json"))
station_metrics = load_metrics(os.path.join(MODEL_DIR, "station", "metrics.json"))

cluster_map = pd.read_csv(CLUSTER_MAP_PATH)[["station", "cluster_id"]]
station_to_cluster = dict(zip(cluster_map["station"], cluster_map["cluster_id"]))

pickup_avg = pd.read_csv(os.path.join(MODEL_DIR, "avg_hourly_pickup_by_station.csv"))
dropoff_avg = pd.read_csv(os.path.join(MODEL_DIR, "avg_hourly_dropoff_by_station.csv"))
stations = sorted(set(pickup_avg["station"].unique()) | set(dropoff_avg["station"].unique()))


def is_good(metric):
    return metric["MAE"] < 2 and metric["R2"] > 0.6


strategy = {}

for station in stations:
    result = {}

    for task in ["pickup", "dropoff"]:
        candidates = {}

        
        if task in global_metrics:
            best_name = global_metrics[task].get("best_model")
            if best_name:
                candidates["global"] = global_metrics[task].get(best_name)

       
        cluster_id = station_to_cluster.get(station)
        if cluster_id is not None:
            cluster_entry = cluster_metrics.get(task, {}).get(str(cluster_id))
            if cluster_entry:
                best_cluster_model = cluster_entry.get(cluster_entry.get("best_model"))
                if best_cluster_model:
                    candidates["cluster"] = best_cluster_model

       
        if task in station_metrics and station in station_metrics[task]:
            station_entry = station_metrics[task][station]
            best_station_model = station_entry.get(station_entry.get("best_model"))
            if best_station_model:
                candidates["station"] = best_station_model

      
        best_model = None
        best_mae = float("inf")
        for model_name, metric in candidates.items():
            if "MAE" in metric and metric["MAE"] < best_mae:
                best_model = model_name
                best_mae = metric["MAE"]

        
        if best_model and is_good(candidates[best_model]):
            result[task] = best_model
        else:
            result[task] = "average"

    strategy[station] = result


with open(STRATEGY_PATH, "w") as f:
    json.dump(strategy, f, indent=4, ensure_ascii=False)

print(f"策略文件已生成：{STRATEGY_PATH}")
