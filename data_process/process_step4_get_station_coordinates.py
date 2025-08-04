import os
import pandas as pd
import requests
import time


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "processed_data", "hourly_pickup_count.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "station_coordinates_google.csv")


API_KEY = "AIzaSyB7u0xntolbGJO_K7qbY0qQeNZokeHcoGk"


df = pd.read_csv(INPUT_PATH)
stations = df["Start station"].dropna().unique()


def geocode_google(station_name):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": f"{station_name}, London, UK",
        "key": API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("results")
        if results:
            location = results[0]["geometry"]["location"]
            return location["lat"], location["lng"]
    except Exception as e:
        print(f"查询失败：{station_name} → {e}")
    return None, None


records = []
for i, name in enumerate(stations):
    lat, lon = geocode_google(name)
    records.append({"station": name, "latitude": lat, "longitude": lon})
    print(f" [{i+1}/{len(stations)}] {name}: ({lat}, {lon})")
    time.sleep(0.2)  # 防止请求过快触发限制


coord_df = pd.DataFrame(records)
coord_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n所有站点经纬度已保存至: {OUTPUT_PATH}")
