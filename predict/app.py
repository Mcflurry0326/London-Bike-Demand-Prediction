import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime, time
import os
import sys
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from predict_bike_usage import predict_bike_usage
MAPBOX_KEY = "pk.eyJ1IjoibWNmbHVycnkwMzI2IiwiYSI6ImNtZHdwbDZxdzF5OXUybnNiYjRvcDI1c3gifQ._25DNkeuzveF7frWmA88KQ"



st.set_page_config(page_title="🚲 Bike Demand Forecasting", layout="wide")


if "history" not in st.session_state:
    st.session_state.history = []

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLUSTER_CSV_PATH = os.path.join(BASE_DIR, "processed_data", "station_with_clusters.csv")
station_df = pd.read_csv(CLUSTER_CSV_PATH)
station_name_to_coords = dict(zip(station_df["station"], zip(station_df["latitude"], station_df["longitude"])))


@st.cache_data
def load_icon_as_base64(path):
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGBA")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_img}"

icon_base64_url = load_icon_as_base64(os.path.join(BASE_DIR, "predict", "destination.png"))
if icon_base64_url is None:
    st.error("图标文件 destination.png 未找到，请确保它在 predict 文件夹中。")
    st.stop()


col_logo, col_blank = st.columns([1, 5])
with col_logo:
    st.image(os.path.join(BASE_DIR, "predict", "logo.jpg"), width=120)
st.title("🚲 Future Bike Demand Forecasting System")

page = st.sidebar.radio(
    "Navigation",
    ["🚴 Future Forecast", "📊 History Comparison"],
    key="main_nav_radio"
)

def run_prediction_module():
    station_list = sorted(list(station_name_to_coords.keys()))
    station_name = st.selectbox("Select station name", station_list)

    col1, col2 = st.columns(2)
    with col1:
        date_input = st.date_input("Select forecast date", min_value=datetime.today().date())
    with col2:
        time_input = st.time_input("Select forecast hour", value=time(8, 0))
        hour = time_input.hour if time_input.minute < 30 else (time_input.hour + 1) % 24

    target = st.radio(
        "Target",
        options=["start", "end"],
        format_func=lambda x: "Start Volume" if x == "start" else "End Volume",
        key="target_radio"
    )

    if st.button("Predict 🚀"):
        with st.spinner("Fetching weather and predicting..."):
            task = "pickup" if target == "start" else "dropoff"
            dt_str = f"{date_input} {hour:02d}:00"
            prediction, strategy, weather = predict_bike_usage(station_name, dt_str, task)

      
        station_type = station_df.loc[station_df["station"] == station_name, "type"].values
        if len(station_type) > 0:
            st.markdown(f"#### 🏷️ Station Type: `{station_type[0]}`-oriented station (based on weekly usage pattern)")
        else:
            st.markdown("#### 🏷️ Station Type: Unknown")

        st.subheader("🔍 Prediction Result")
        st.success(f"✅ Success: Expected **{target} count is {prediction} bikes** (strategy: {strategy})")
        st.balloons()

        st.markdown("### 🌦️ Weather Conditions")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🌡️ Temperature:** {weather.get('temp', 'N/A')} ℃")
            st.markdown(f"**☔ Precipitation:** {weather.get('precip', 'N/A')} mm")
        with col2:
            st.markdown(f"**💨 Wind Speed:** {weather.get('windspeed', 'N/A')} miles/h")
            st.markdown(f"**💧 Humidity:** {weather.get('humidity', 'N/A')}%")


        
        st.session_state.history.insert(0, {
            "datetime": dt_str,
            "station": station_name,
            "target": "Start Volume" if target == "start" else "End Volume",
            "value": prediction,
            "weather": {}  # 可扩展天气信息
        })
        st.session_state.history = st.session_state.history[:5]

        
        st.markdown("### 🗺️ Map Location and Forecast")
        lat, lon = station_name_to_coords.get(station_name, (51.5074, -0.1278))  # fallback

        map_df = pd.DataFrame([{
            "Station": station_name,
            "Latitude": lat,
            "Longitude": lon,
            "Prediction": f"{target}: {prediction} bikes",
            "icon_data": {
                "url": icon_base64_url,
                "width": 128,
                "height": 128,
                "anchorY": 128
            }
        }])

        icon_layer = pdk.Layer(
            "IconLayer",
            data=map_df,
            get_icon='icon_data',
            get_size=4,
            size_scale=15,
            get_position='[Longitude, Latitude]',
            pickable=True
        )

        text_layer = pdk.Layer(
            "TextLayer",
            data=map_df,
            get_position='[Longitude, Latitude]',
            get_text='Station',
            get_size=20,
            get_color='[20, 20, 60]',
            get_alignment_baseline="'top'",
            get_text_anchor="'middle'"
        )

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/streets-v11',
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=13, pitch=0),
            layers=[icon_layer, text_layer],
            tooltip={"text": "{Station}\n{Prediction}"},
            mapbox_key=MAPBOX_KEY
        ))


def run_history_module():
    st.subheader("📊 Forecast History")
    history = st.session_state.get("history", [])
    if not history:
        st.info("No history available.")
        return

    options = [f"{h['datetime']} | {h['station']} | {h['target']}" for h in history]
    selected = st.multiselect("Select records to compare:", options)
    if not selected:
        st.warning("Please select at least one record.")
        return

    compare_data = [h for h, label in zip(history, options) if label in selected]
    stations = [f"{h['station']} ({h['datetime'][-5:]})" for h in compare_data]
    values = [h["value"] for h in compare_data]

    fig, ax = plt.subplots(figsize=(8, len(stations) * 0.6))  # 图像高度按记录数自动调整    
    bars = ax.barh(stations, values, color="#74C0FC", edgecolor="#1C7ED6")

   
    ax.set_xlabel("Forecast Count (bikes)", fontsize=12)
    ax.set_title("Forecast Comparison", fontsize=14, weight="bold")

   
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)


    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}', va='center', fontsize=10, color="#1C1C1C")

    # 去掉不必要的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)



if page == "🚴 Future Forecast":
    run_prediction_module()
elif page == "📊 History Comparison":
    run_history_module()


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:gray;'>© 2025 Xiang Wang. v1.0.0</div>",
    unsafe_allow_html=True
)
