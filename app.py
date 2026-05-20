import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import io

# Set professional page configurations
st.set_page_config(page_title="Meteorological Precision Dashboard", layout="wide", page_icon="📊")

st.title("📊 Meteorological Metric Precision Dashboard")
st.markdown("### High-Resolution 24-Hour Convective Early Warning Tracking Core")
st.divider()

# 1. Sidebar Configuration Controls
st.sidebar.header("🎯 Location Constraints")
lat = st.sidebar.slider("Latitude (°N)", min_value=22.0, max_value=24.5, value=22.83, step=0.01)
lon = st.sidebar.slider("Longitude (°E)", min_value=86.5, max_value=89.5, value=87.92, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Metrics:** Uses a live Magnus-Tetens thermodynamic framework to monitor instability and predict severe convective squall signatures.")

# 2. Live API Processing Core with Safety Fallback
@st.cache_data(ttl=300)
def fetch_weather_matrix(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,precipitation,precipitation_probability,wind_gusts_10m",
        "timezone": "auto",
        "forecast_days": 2
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        res_json = response.json()
        
        # VALIDATION GATEWAY: If 'hourly' isn't in the keys, Open-Meteo sent an explicit error payload
        if "hourly" not in res_json:
            st.sidebar.error(f"API Warning: {res_json.get('reason', 'Rate limited or network drop')}")
            return None
        return res_json
    except Exception as e:
        st.sidebar.error(f"Network Timeout: {e}")
        return None

raw_data = fetch_weather_matrix(lat, lon)

# Generate a high-precision synthetic weather matrix if the API line drops
if raw_data is None:
    st.warning("⚠️ API connection threshold reached. Displaying mathematically simulated thermodynamic data canvas.")
    base_time = datetime.now()
    times = [base_time + timedelta(hours=i) for i in range(48)]
    
    # Simulate a realistic local diurnal barometric and temperature wave signature
    np.random.seed(42)
    temps = 28 + 6 * np.sin(np.arange(48) * (2 * np.pi / 24) - np.pi/2) + np.random.normal(0, 0.5, 48)
    humidity = 75 - 15 * np.sin(np.arange(48) * (2 * np.pi / 24) - np.pi/2) + np.random.normal(0, 1, 48)
    pressure = 1008 - 2 * np.sin(np.arange(48) * (2 * np.pi / 24)) + np.random.normal(0, 0.2, 48)
    pop = np.random.choice([10, 20, 65, 80], size=48, p=[0.4, 0.3, 0.2, 0.1])
    precip = [0.0 if p < 50 else round(np.random.uniform(0.5, 6.0), 2) for p in pop]
    gusts = [12.0 if p < 50 else round(np.random.uniform(30.0, 55.0), 1) for p in pop]
    
    df_live = pd.DataFrame({
        "Timestamp": pd.to_datetime(times),
        "Temp_C": temps,
        "Humidity_Pct": humidity,
        "Pressure_hPa": pressure,
        "Precip_mm": precip,
        "PoP_Pct": pop,
        "Wind_Gust_kmh": gusts
    })
else:
    hourly_data = raw_data["hourly"]
    df_live = pd.DataFrame({
        "Timestamp": pd.to_datetime(hourly_data["time"]),
        "Temp_C": hourly_data["temperature_2m"],
        "Humidity_Pct": hourly_data["relative_humidity_2m"],
        "Pressure_hPa": hourly_data["surface_pressure"],
        "Precip_mm": hourly_data["precipitation"],
        "PoP_Pct": hourly_data["precipitation_probability"],
        "Wind_Gust_kmh": hourly_data["wind_gusts_10m"]
    })

# 3. Process Structural Mathematical Layers
now = pd.Timestamp(datetime.now())
df_24h = df_live[df_live["Timestamp"] >= now].head(24).copy()

df_24h["Pressure_Delta_3h"] = df_24h["Pressure_hPa"].diff(3).fillna(0)
a, b = 17.27, 237.7
alpha = ((a * df_24h["Temp_C"]) / (b + df_24h["Temp_C"])) + np.log(df_24h["Humidity_Pct"] / 100.0)
df_24h["Dew_Point_C"] = (b * alpha) / (a - alpha)
df_24h["Thermodynamic_Spread"] = df_24h["Temp_C"] - df_24h["Dew_Point_C"]

# Collapse into 3-Hourly Windows
df_3h = df_24h.groupby(df_24h.index // 3).agg({
    "Timestamp": "first",
    "Temp_C": "max",
    "Thermodynamic_Spread": "min",
    "Pressure_Delta_3h": "last",
    "PoP_Pct": "max",
    "Precip_mm": "sum",
    "Wind_Gust_kmh": "max"
}).reset_index(drop=True)

# Apply Risk Logic
def classify_storm_risk(row):
    if row["PoP_Pct"] >= 70 and row["Wind_Gust_kmh"] >= 40 and row["Pressure_Delta_3h"] < -1.5:
        return "HIGH RISK"
    elif row["PoP_Pct"] >= 60 and (row["Precip_mm"] >= 4.0 or row["Thermodynamic_Spread"] < 2.5):
        return "MODERATE RISK"
    elif row["PoP_Pct"] >= 30 or row["Precip_mm"] > 0.1:
        return "LOW RISK"
    return "NIL"

df_3h["Risk_Assessment"] = df_3h.apply(classify_storm_risk, axis=1)
df_3h["Time Window Starting"] = df_3h["Timestamp"].dt.strftime("%Y-%m-%d %I:%M %p")

# 4. Layout: Live Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Peak Target Temp", f"{df_3h['Temp_C'].max():.1f} °C")
with col2:
    st.metric("Max Rain Probability", f"{df_3h['PoP_Pct'].max()}%")
with col3:
    st.metric("Total 24h Vol (Est)", f"{df_3h['Precip_mm'].sum():.2f} mm")
with col4:
    st.metric("Peak Predicted Gust", f"{df_3h['Wind_Gust_kmh'].max():.1f} km/h")

# 5. Interactive Data Visualizations
st.markdown("### 📈 24-Hour Trend Analytics")
chart_df = df_24h.set_index("Timestamp")
st.line_chart(chart_df[["Temp_C", "Dew_Point_C"]])
st.area_chart(chart_df["Wind_Gust_kmh"])

# 6. Formatted Status Data Display Matrix
st.markdown("### 📋 3-Hourly Precision Timeline")
def style_risk_row(val):
    if val == "HIGH RISK": return 'background-color: #FFC7CE; color: #9C0006; font-weight: bold;'
    elif val == "MODERATE RISK": return 'background-color: #FFEB9C; color: #9C6500;'
    elif val == "LOW RISK": return 'background-color: #D9EAD3; color: #274E13;'
    return 'background-color: #F3F3F3; color: #666666;'

df_display = df_3h[["Time Window Starting", "Temp_C", "Thermodynamic_Spread", "Pressure_Delta_3h", "PoP_Pct", "Precip_mm", "Wind_Gust_kmh", "Risk_Assessment"]]
df_display.columns = ["Time Window", "Max Temp (°C)", "Dewpoint Spread (°C)", "3h Pressure Delta (hPa)", "Rain Prob (%)", "Volume (mm)", "Peak Gust (km/h)", "ALERT METRIC"]
st.dataframe(df_display.style.applymap(style_risk_row, subset=["ALERT METRIC"]), use_container_width=True)

# 7. In-Memory Excel Workbook Compilation
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_display.to_excel(writer, sheet_name="Live Forecast", index=False)

st.download_button(
    label="📥 Download Formatted Excel Sheet",
    data=buffer.getvalue(),
    file_name=f"Forecast_{lat}_{lon}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
