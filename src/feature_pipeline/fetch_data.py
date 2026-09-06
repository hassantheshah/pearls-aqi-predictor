"""
src/feature_pipeline/fetch_data.py
Fetches real-time AQI + weather data from AQICN and OpenWeatherMap APIs.
"""
import requests
import pandas as pd
from datetime import datetime
from loguru import logger
import sys
sys.path.append("../..")
from config.settings import (
    AQICN_API_KEY, OPENWEATHER_API_KEY,
    CITY_NAME, CITY_LAT, CITY_LON, AQICN_CITY_SLUG
)


def fetch_aqicn_data() -> dict:
    """Fetch current AQI and pollutant data from AQICN API."""
    url = f"https://api.waqi.info/feed/{AQICN_CITY_SLUG}/?token={AQICN_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "ok":
            logger.warning(f"AQICN returned non-ok status: {data}")
            return {}

        iaqi = data["data"].get("iaqi", {})
        return {
            "aqi":         data["data"].get("aqi", None),
            "pm25":        iaqi.get("pm25", {}).get("v", None),
            "pm10":        iaqi.get("pm10", {}).get("v", None),
            "no2":         iaqi.get("no2",  {}).get("v", None),
            "o3":          iaqi.get("o3",   {}).get("v", None),
            "co":          iaqi.get("co",   {}).get("v", None),
            "timestamp":   data["data"]["time"]["iso"],
        }
    except Exception as e:
        logger.error(f"AQICN fetch failed: {e}")
        return {}


def fetch_openweather_data() -> dict:
    """Fetch current weather data from OpenWeatherMap API."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={CITY_LAT}&lon={CITY_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity":    data["main"]["humidity"],
            "pressure":    data["main"]["pressure"],
            "wind_speed":  data["wind"]["speed"],
            "weather_desc": data["weather"][0]["description"],
        }
    except Exception as e:
        logger.error(f"OpenWeather fetch failed: {e}")
        return {}


def fetch_openweather_forecast() -> list[dict]:
    """Fetch 5-day / 3-hour weather forecast from OpenWeatherMap."""
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={CITY_LAT}&lon={CITY_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        forecasts = []
        for item in data["list"]:
            forecasts.append({
                "forecast_time": item["dt_txt"],
                "temperature":   item["main"]["temp"],
                "humidity":      item["main"]["humidity"],
                "pressure":      item["main"]["pressure"],
                "wind_speed":    item["wind"]["speed"],
            })
        return forecasts
    except Exception as e:
        logger.error(f"OpenWeather forecast fetch failed: {e}")
        return []


def fetch_combined_current() -> dict:
    """Merge AQI + weather into a single record."""
    aqi_data     = fetch_aqicn_data()
    weather_data = fetch_openweather_data()

    if not aqi_data:
        logger.error("No AQI data available — aborting.")
        return {}

    record = {
        "city":      CITY_NAME,
        "fetched_at": datetime.utcnow().isoformat(),
        **aqi_data,
        **weather_data,
    }
    logger.info(f"Fetched combined record: AQI={record.get('aqi')}")
    return record


if __name__ == "__main__":
    import json
    record = fetch_combined_current()
    print(json.dumps(record, indent=2))
