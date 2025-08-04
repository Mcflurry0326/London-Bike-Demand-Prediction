
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_for_model")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_BASE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_BASE_DIR, exist_ok=True)


def save_feature_order(df, path):
    with open(path, "w") as f:
        json.dump(df.columns.tolist(), f, indent=2)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "xgboost": XGBRegressor(random_state=42, verbosity=0),
        "lightgbm": LGBMRegressor(random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "MAE": round(mean_absolute_error(y_test, preds), 4),
            "R2": round(r2_score(y_test, preds), 4)
        }
    best_model_name = min(results, key=lambda k: results[k]["MAE"])
    best_model = models[best_model_name]
    best_model.fit(X, y)
    results["best_model"] = best_model_name
    return best_model, results

def filter_features(df, target_col, drop_cols):
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.select_dtypes(include=["int", "float", "bool"])
    return df.drop(columns=[target_col]), df[target_col]


def train_global():
    model_dir = os.path.join(MODEL_BASE_DIR, "global")
    os.makedirs(model_dir, exist_ok=True)

    pickup_df = pd.read_csv(os.path.join(DATA_DIR, "pickup_features_2.csv"))
    dropoff_df = pd.read_csv(os.path.join(DATA_DIR, "dropoff_features_2.csv"))

    X_pu, y_pu = filter_features(pickup_df, "pickup_count", ["Start station", "cluster_id"])
    X_do, y_do = filter_features(dropoff_df, "dropoff_count", ["End station", "cluster_id"])

    pu_model, pu_metrics = train_model(X_pu, y_pu)
    joblib.dump(pu_model, os.path.join(model_dir, "pickup_model.pkl"))
    save_feature_order(X_pu, os.path.join(model_dir, "pickup_features_order.json"))

    do_model, do_metrics = train_model(X_do, y_do)
    joblib.dump(do_model, os.path.join(model_dir, "dropoff_model.pkl"))
    save_feature_order(X_do, os.path.join(model_dir, "dropoff_features_order.json"))

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({"pickup": pu_metrics, "dropoff": do_metrics}, f, indent=4)


def train_cluster():
    model_dir = os.path.join(MODEL_BASE_DIR, "cluster")
    os.makedirs(model_dir, exist_ok=True)

    pickup_df = pd.read_csv(os.path.join(DATA_DIR, "pickup_features_2.csv"))
    dropoff_df = pd.read_csv(os.path.join(DATA_DIR, "dropoff_features_2.csv"))

    pickup_df["cluster_id"] = pickup_df["cluster_id"].astype(int)
    dropoff_df["cluster_id"] = dropoff_df["cluster_id"].astype(int)

    def train_by_cluster(df, target_col, name):
        results = {}
        for cluster_id, group in df.groupby("cluster_id"):
            if len(group) < 20:
                continue
            X, y = filter_features(group, target_col, ["cluster_id"])
            model, metrics = train_model(X, y)
            model_path = os.path.join(model_dir, f"{name}_cluster_{int(cluster_id)}.pkl")
            joblib.dump(model, model_path)
            results[str(cluster_id)] = metrics
        save_feature_order(X, os.path.join(model_dir, f"{name}_features_order.json"))
        return results

    pu_metrics = train_by_cluster(pickup_df, "pickup_count", "pickup")
    do_metrics = train_by_cluster(dropoff_df, "dropoff_count", "dropoff")

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({"pickup": pu_metrics, "dropoff": do_metrics}, f, indent=4)


def train_station():
    model_dir = os.path.join(MODEL_BASE_DIR, "station")
    os.makedirs(model_dir, exist_ok=True)

    pickup_df = pd.read_csv(os.path.join(DATA_DIR, "pickup_features_2.csv"))
    dropoff_df = pd.read_csv(os.path.join(DATA_DIR, "dropoff_features_2.csv"))

    def train_by_station(df, target_col, station_col, name):
        results = {}
        for station, group in df.groupby(station_col):
            if len(group) < 20:
                continue
            X, y = filter_features(group, target_col, [station_col, "cluster_id"])
            model, metrics = train_model(X, y)
            safe_station = station.replace("/", "_").replace(" ", "_").replace(",", "")
            path = os.path.join(model_dir, f"{name}_station_{safe_station}.pkl")
            joblib.dump(model, path)
            results[station] = metrics
        save_feature_order(X, os.path.join(model_dir, f"{name}_features_order.json"))
        return results

    pu_metrics = train_by_station(pickup_df, "pickup_count", "station", "pickup")
    do_metrics = train_by_station(dropoff_df, "dropoff_count", "station", "dropoff")

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({"pickup": pu_metrics, "dropoff": do_metrics}, f, indent=4)


if __name__ == "__main__":
    print("开始训练 Global 模型...")
    train_global()
    print("Global 完成\n")

    print("开始训练 Cluster 模型...")
    train_cluster()
    print("Cluster 完成\n")

    print("开始训练 Station 模型...")
    train_station()
    print("Station 完成\n")

    print("所有模型训练完成！")
