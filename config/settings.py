"""
config/settings.py — Central configuration for Pearls AQI Predictor
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────
AQICN_API_KEY        = os.getenv("AQICN_API_KEY", "")
OPENWEATHER_API_KEY  = os.getenv("OPENWEATHER_API_KEY", "")

# ── City ──────────────────────────────────────────────
CITY_NAME       = os.getenv("CITY_NAME", "Karachi")
CITY_LAT        = float(os.getenv("CITY_LAT", 24.8607))
CITY_LON        = float(os.getenv("CITY_LON", 67.0011))
AQICN_CITY_SLUG = os.getenv("AQICN_CITY_SLUG", "karachi")

# ── Hopsworks ─────────────────────────────────────────
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "pearls_aqi")

# ── Forecast ──────────────────────────────────────────
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", 3))
ALERT_THRESHOLD = int(os.getenv("ALERT_AQI_THRESHOLD", 150))

# ── Feature names ─────────────────────────────────────
FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "no2", "o3", "co",
    "temperature", "humidity", "wind_speed", "pressure",
    "hour", "day_of_week", "month", "is_weekend",
    "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_24",
    "aqi_rolling_mean_3", "aqi_rolling_mean_6",
    "aqi_change_rate"
]

TARGET_COLUMN = "aqi"

# ── AQI Categories ────────────────────────────────────
AQI_CATEGORIES = {
    (0,   50):  ("Good",                "#00e400", "😊"),
    (51,  100): ("Moderate",            "#ffff00", "😐"),
    (101, 150): ("Unhealthy for Sensitive", "#ff7e00", "😷"),
    (151, 200): ("Unhealthy",           "#ff0000", "🚨"),
    (201, 300): ("Very Unhealthy",      "#8f3f97", "⚠️"),
    (301, 500): ("Hazardous",           "#7e0023", "☠️"),
}

def get_aqi_category(aqi_value: float) -> tuple:
    for (lo, hi), info in AQI_CATEGORIES.items():
        if lo <= aqi_value <= hi:
            return info
    return ("Hazardous", "#7e0023", "☠️")
