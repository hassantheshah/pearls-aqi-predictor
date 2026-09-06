"""
src/feature_pipeline/fetch_data.py

Fetches:
- real-time AQI + weather data for live inference
- historical AQI + weather data from Open-Meteo for training/backfill
"""

import sys
from datetime import datetime

import pandas as pd
import requests
from loguru import logger

sys.path.append("../..")

from config.settings import (
    AQICN_API_KEY,
    OPENWEATHER_API_KEY,
    CITY_NAME,
    CITY_LAT,
    CITY_LON,
    AQICN_CITY_SLUG,
)


def fetch_aqicn_data() -> dict:
    """Fetch current AQI and pollutant data from AQICN API."""

    url = (
        f"https://api.waqi.info/feed/"
        f"{AQICN_CITY_SLUG}/?token={AQICN_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "ok":
            logger.warning(
                f"AQICN returned non-ok status: {data}"
            )
            return {}

        iaqi = data["data"].get("iaqi", {})

        return {
            "aqi": data["data"].get("aqi"),
            "pm25": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
            "co": iaqi.get("co", {}).get("v"),
            "timestamp": data["data"]["time"]["iso"],
        }

    except Exception as exc:
        logger.error(f"AQICN fetch failed: {exc}")
        return {}


def fetch_openweather_data() -> dict:
    """Fetch current weather data from OpenWeatherMap API."""

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={CITY_LAT}"
        f"&lon={CITY_LON}"
        f"&appid={OPENWEATHER_API_KEY}"
        "&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "weather_desc": data["weather"][0]["description"],
        }

    except Exception as exc:
        logger.error(f"OpenWeather fetch failed: {exc}")
        return {}


def fetch_openweather_forecast() -> list[dict]:
    """Fetch 5-day / 3-hour weather forecast from OpenWeatherMap."""

    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={CITY_LAT}"
        f"&lon={CITY_LON}"
        f"&appid={OPENWEATHER_API_KEY}"
        "&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        forecasts = []

        for item in data["list"]:
            forecasts.append(
                {
                    "forecast_time": item["dt_txt"],
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "pressure": item["main"]["pressure"],
                    "wind_speed": item["wind"]["speed"],
                }
            )

        return forecasts

    except Exception as exc:
        logger.error(
            f"OpenWeather forecast fetch failed: {exc}"
        )
        return []


def fetch_openmeteo_historical_data(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical hourly AQI and weather data from Open-Meteo.

    AQI and pollutant data come from Open-Meteo's historical
    air-quality/reanalysis data.

    Weather data comes from Open-Meteo historical weather
    reanalysis data.

    Returns a single hourly DataFrame using the project's
    standard column names.
    """

    logger.info(
        f"Fetching Open-Meteo historical data: "
        f"{start_date} -> {end_date}"
    )

    air_quality_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    weather_url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    air_params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "us_aqi,"
            "pm2_5,"
            "pm10,"
            "nitrogen_dioxide,"
            "ozone,"
            "carbon_monoxide"
        ),
        "timezone": "Asia/Karachi",
    }

    weather_params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "pressure_msl"
        ),
        "wind_speed_unit": "ms",
        "timezone": "Asia/Karachi",
    }

    try:
        air_response = requests.get(
            air_quality_url,
            params=air_params,
            timeout=60,
        )
        air_response.raise_for_status()

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=60,
        )
        weather_response.raise_for_status()

        air_data = air_response.json()
        weather_data = weather_response.json()

        if "hourly" not in air_data:
            raise ValueError(
                "Open-Meteo air-quality response has no hourly data."
            )

        if "hourly" not in weather_data:
            raise ValueError(
                "Open-Meteo weather response has no hourly data."
            )

        air_hourly = air_data["hourly"]
        weather_hourly = weather_data["hourly"]

        air_df = pd.DataFrame(air_hourly)
        weather_df = pd.DataFrame(weather_hourly)

        air_df = air_df.rename(
            columns={
                "time": "fetched_at",
                "us_aqi": "aqi",
                "pm2_5": "pm25",
                "nitrogen_dioxide": "no2",
                "ozone": "o3",
                "carbon_monoxide": "co",
            }
        )

        weather_df = weather_df.rename(
            columns={
                "time": "fetched_at",
                "temperature_2m": "temperature",
                "relative_humidity_2m": "humidity",
                "wind_speed_10m": "wind_speed",
                "pressure_msl": "pressure",
            }
        )

        air_df["fetched_at"] = pd.to_datetime(
            air_df["fetched_at"]
        )

        weather_df["fetched_at"] = pd.to_datetime(
            weather_df["fetched_at"]
        )

        merged = pd.merge(
            air_df,
            weather_df,
            on="fetched_at",
            how="inner",
        )

        merged["city"] = CITY_NAME

        required_columns = [
            "city",
            "fetched_at",
            "aqi",
            "pm25",
            "pm10",
            "no2",
            "o3",
            "co",
            "temperature",
            "humidity",
            "wind_speed",
            "pressure",
        ]

        missing = [
            column
            for column in required_columns
            if column not in merged.columns
        ]

        if missing:
            raise ValueError(
                f"Missing historical columns: {missing}"
            )

        merged = merged[required_columns]

        merged = merged.dropna(
            subset=[
                "aqi",
                "temperature",
                "humidity",
                "wind_speed",
                "pressure",
            ]
        )

        merged = (
            merged
            .sort_values("fetched_at")
            .drop_duplicates(
                subset=["fetched_at"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        if merged.empty:
            raise ValueError(
                "Open-Meteo returned no usable historical rows."
            )

        logger.info(
            f"Open-Meteo historical fetch complete: "
            f"{len(merged)} hourly rows."
        )

        logger.info(
            f"Historical range: "
            f"{merged['fetched_at'].min()} -> "
            f"{merged['fetched_at'].max()}"
        )

        return merged

    except Exception as exc:
        logger.error(
            f"Open-Meteo historical fetch failed: {exc}"
        )
        raise


def fetch_combined_current() -> dict:
    """Merge current AQI + weather into one record."""

    aqi_data = fetch_aqicn_data()
    weather_data = fetch_openweather_data()

    if not aqi_data:
        logger.error(
            "No AQI data available — aborting."
        )
        return {}

    record = {
        "city": CITY_NAME,
        "fetched_at": datetime.utcnow().isoformat(),
        **aqi_data,
        **weather_data,
    }

    logger.info(
        f"Fetched combined record: "
        f"AQI={record.get('aqi')}"
    )

    return record


if __name__ == "__main__":
    import json

    record = fetch_combined_current()

    print(
        json.dumps(
            record,
            indent=2,
        )
    )
