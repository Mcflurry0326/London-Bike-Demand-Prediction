
import os
import json
import pandas as pd
import joblib
from datetime import datetime
from dateutil import parser
import requests
import math


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data_for_model")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
STRATEGY_PATH = os.path.join(MODEL_DIR, "strategy_by_station.json")
CLUSTER_MAP_PATH = os.path.join(PROCESSED_DIR, "station_with_clusters.csv")
PICKUP_AVG_PATH = os.path.join(MODEL_DIR, "avg_hourly_pickup_by_station.csv")
DROPOFF_AVG_PATH = os.path.join(MODEL_DIR, "avg_hourly_dropoff_by_station.csv")
WEEKTYPE_PATH = os.path.join(PROCESSED_DIR, "station_weektype_label.csv")


with open(STRATEGY_PATH, "r") as f:
    strategy = json.load(f)

cluster_map_df = pd.read_csv(CLUSTER_MAP_PATH)
station_to_cluster = dict(zip(cluster_map_df["station"], cluster_map_df["cluster_id"]))

pickup_avg_df = pd.read_csv(PICKUP_AVG_PATH)
dropoff_avg_df = pd.read_csv(DROPOFF_AVG_PATH)

weektype_df = pd.read_csv(WEEKTYPE_PATH)
station_weektype = dict(zip(weektype_df["station"], weektype_df["type"]))


WEEKDAY_FACTOR = {
    0: 0.1, 1: 0.1, 2: 0.05, 3: 0.05, 4: 0.05,
    5: 0.1, 6: 0.2,
    7: 0.6, 8: 0.7, 9: 0.6,
    10: 0.3, 11: 0.3, 12: 0.3, 13: 0.25, 14: 0.25,
    15: 0.3, 16: 0.4,
    17: 0.7, 18: 0.7, 19: 0.6,
    20: 0.2, 21: 0.15, 22: 0.1, 23: 0.1
}

WEEKDAY_ON_WEEKEND_FACTOR = {h: v * 0.5 for h, v in WEEKDAY_FACTOR.items()}
WEEKEND_FACTOR = {
    0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05,
    5: 0.1, 6: 0.15,
    7: 0.2, 8: 0.25, 9: 0.3,
    10: 0.4, 11: 0.5, 12: 0.55, 13: 0.55, 14: 0.6,
    15: 0.65, 16: 0.7,
    17: 0.8, 18: 0.8, 19: 0.6,
    20: 0.3, 21: 0.2, 22: 0.1, 23: 0.1
}

WEEKEND_ON_WEEKDAY_FACTOR = {h: v * 0.5 for h, v in WEEKEND_FACTOR.items()}


def get_weather_features(dt_str):
    API_KEY = "ARP24KS4QMBWNTE8SSSUMV4WU"
    dt_obj = parser.parse(dt_str)
    date_param = dt_obj.strftime("%Y-%m-%dT%H:00:00")

    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/london/{date_param}"
        f"?unitGroup=metric&key={API_KEY}&include=hours"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        hour_data = data["days"][0]["hours"][dt_obj.hour]
        return {
            "temp": hour_data.get("temp", 12.0),
            "humidity": hour_data.get("humidity", 70.0),
            "precip": hour_data.get("precip", 0.0),
            "windspeed": hour_data.get("windspeed", 10.0),
            "cloudcover": hour_data.get("cloudcover", 80.0)
        }
    except:
        return {
            "temp": 12.0,
            "humidity": 70.0,
            "precip": 0.0,
            "windspeed": 10.0,
            "cloudcover": 80.0
        }


def build_feature_row(station_name, dt, task="pickup"):
    dt = parser.parse(dt) if isinstance(dt, str) else dt
    cluster_id = station_to_cluster.get(station_name, -1)
    hour = dt.hour
    dayofweek = dt.weekday()
    is_weekend = int(dayofweek >= 5)
    is_peak_hour = int(hour in [7, 8, 9, 17, 18, 19])
    month = dt.month

    weektype = station_weektype.get(station_name, "weekday")
    if weektype == "weekday" and is_weekend == 0:
        factor = WEEKDAY_FACTOR.get(hour, 1)
    elif weektype == "weekday" and is_weekend == 1:
        factor = WEEKDAY_ON_WEEKEND_FACTOR.get(hour, 1)
    elif weektype == "weekend" and is_weekend == 0:
        factor = WEEKEND_ON_WEEKDAY_FACTOR.get(hour, 1)
    else:
        factor = WEEKEND_FACTOR.get(hour, 1)

    cluster_avg_df = pickup_avg_df if task == "pickup" else dropoff_avg_df
    cluster_avg = cluster_avg_df.query("station == @station_name and hour == @hour")
    cluster_mean = cluster_avg.iloc[0, 2] if not cluster_avg.empty else 0.0

    lag_1h = cluster_mean * factor
    rolling_3h = cluster_mean * (factor + 0.1)
    rolling_6h = cluster_mean * (factor + 0.15)
    cumsum_day = cluster_mean * (hour + 1) * factor

    weather = get_weather_features(dt.strftime("%Y-%m-%d %H:%M"))

    base = {
        "cloudcover": weather["cloudcover"],
        f"cluster_hourly_avg_{task}": cluster_mean,
        "dayofweek": dayofweek,
        "hour": hour,
        "humidity": weather["humidity"],
        "is_peak_hour": is_peak_hour,
        "is_weekend": is_weekend,
        "latitude": cluster_map_df.loc[cluster_map_df["station"] == station_name, "latitude"].values[0],
        "longitude": cluster_map_df.loc[cluster_map_df["station"] == station_name, "longitude"].values[0],
        "month": month,
        f"{task}_count_cumsum_day": cumsum_day,
        f"{task}_count_lag_1h": lag_1h,
        f"{task}_count_rolling_3h_mean": rolling_3h,
        f"{task}_count_rolling_6h_std": rolling_6h,
        "precip": weather["precip"],
        "temp": weather["temp"],
        "windspeed": weather["windspeed"]
    }

    return pd.DataFrame([base])


def predict_bike_usage(station_name, dt_str, task="pickup"):
    if station_name not in station_to_cluster:
        return 0.0, f"未知站点: {station_name}"
    assert task in ["pickup", "dropoff"]

    features = build_feature_row(station_name, dt_str, task)
    hour = features["hour"].values[0]
    strat = strategy.get(station_name, {}).get(task, "average")

    if strat == "average":
        df = pickup_avg_df if task == "pickup" else dropoff_avg_df
        try:
            avg_val = df.query("station == @station_name and hour == @hour").values[0][2]
            return max(math.ceil(avg_val), 0), "average"
        except:
            return 0.0, "average (default)"

    if strat == "global":
        model_path = os.path.join(MODEL_DIR, "global", f"{task}_model.pkl")
        feature_path = os.path.join(MODEL_DIR, "global", f"{task}_features_order.json")
    elif strat == "cluster":
        cluster_id = int(station_to_cluster.get(station_name, -1))
        model_path = os.path.join(MODEL_DIR, "cluster", f"{task}_cluster_{cluster_id}.pkl")
        feature_path = os.path.join(MODEL_DIR, "cluster", f"{task}_features_order.json")
    elif strat == "station":
        safe_name = station_name.replace(" ", "_").replace("/", "_").replace(",", "")
        model_path = os.path.join(MODEL_DIR, "station", f"{task}_station_{safe_name}.pkl")
        feature_path = os.path.join(MODEL_DIR, "station", f"{task}_features_order.json")
    else:
        return 0.0, f"{strat} (invalid strategy)"

    if not os.path.exists(model_path):
        return 0.0, f"{strat} (missing model)"

    try:
        if os.path.exists(feature_path):
            with open(feature_path, "r") as f:
                feature_order = json.load(f)
                features = features[feature_order]

        model = joblib.load(model_path)
        pred = model.predict(features)[0]
        weather = get_weather_features(dt_str)
        return max(math.ceil(pred), 0), strat, weather
    except Exception as e:
        return 0.0, f"{strat} (error: {str(e)})"
