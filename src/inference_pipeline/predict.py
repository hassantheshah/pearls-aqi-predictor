"""
src/inference_pipeline/predict.py
Loads trained model and generates AQI forecasts for the next 3 days (72 hours).
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.settings import FEATURE_COLUMNS, TARGET_COLUMN, FORECAST_DAYS, get_aqi_category
from src.feature_pipeline.feature_store import read_latest_features

MODEL_DIR    = "models"
FEATURE_COLS = [c for c in FEATURE_COLUMNS if c != TARGET_COLUMN]


def load_best_model():
    """Load the best registered model from the models directory."""
    try:
        best_name = open(f"{MODEL_DIR}/best_model.txt").read().strip()
        logger.info(f"Loading best model: {best_name}")

        if best_name == "RandomForest":
            model = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
            scaler = None
        elif best_name == "Ridge":
            model  = joblib.load(f"{MODEL_DIR}/ridge.pkl")
            scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        elif best_name == "LSTM":
            import tensorflow as tf
            model  = tf.keras.models.load_model(f"{MODEL_DIR}/lstm_model.keras")
            scaler = None
        else:
            raise ValueError(f"Unknown model: {best_name}")

        feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
        return model, scaler, best_name, feature_cols

    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}. Run training first.")
        raise


def _build_future_row(last_row: pd.Series, step: int,
                      forecast_weather: dict) -> pd.DataFrame:
    """
    Build a synthetic feature row for a future hour.
    Uses the last known AQI and weather forecast values.
    """
    future_time = pd.Timestamp(last_row["fetched_at"]) + timedelta(hours=step)

    row = {
        "fetched_at":   future_time,
        "aqi":          last_row["aqi"],           # will be overwritten by prediction
        "pm25":         last_row.get("pm25", 0),
        "pm10":         last_row.get("pm10", 0),
        "no2":          last_row.get("no2", 0),
        "o3":           last_row.get("o3", 0),
        "co":           last_row.get("co", 0),
        "temperature":  forecast_weather.get("temperature", last_row.get("temperature", 25)),
        "humidity":     forecast_weather.get("humidity",    last_row.get("humidity", 60)),
        "wind_speed":   forecast_weather.get("wind_speed",  last_row.get("wind_speed", 3)),
        "pressure":     forecast_weather.get("pressure",    last_row.get("pressure", 1013)),
        "hour":         future_time.hour,
        "day_of_week":  future_time.dayofweek,
        "month":        future_time.month,
        "is_weekend":   int(future_time.dayofweek >= 5),
        "aqi_lag_1":    last_row.get("aqi", 80),
        "aqi_lag_3":    last_row.get("aqi_lag_1", 80),
        "aqi_lag_6":    last_row.get("aqi_lag_3", 80),
        "aqi_lag_24":   last_row.get("aqi_lag_6", 80),
        "aqi_rolling_mean_3": last_row.get("aqi_rolling_mean_3", 80),
        "aqi_rolling_mean_6": last_row.get("aqi_rolling_mean_6", 80),
        "aqi_change_rate":    last_row.get("aqi_change_rate", 0),
    }
    return pd.DataFrame([row])


def predict_next_hours(n_hours: int = 72,
                       weather_forecast: list = None) -> pd.DataFrame:
    """
    Generate AQI predictions for the next n_hours.

    Args:
        n_hours: Number of hours to forecast (default: 72 = 3 days)
        weather_forecast: List of weather dicts per hour from OpenWeather

    Returns:
        DataFrame with columns: timestamp, predicted_aqi, category, color, emoji, confidence
    """
    model, scaler, model_name, feature_cols = load_best_model()

    # Get recent history for context
    history = read_latest_features(n_rows=48)
    if history.empty:
        logger.error("No recent feature data available.")
        return pd.DataFrame()

    history = history.sort_values("fetched_at").reset_index(drop=True)
    last_row = history.iloc[-1]

    predictions = []
    current_row = last_row.copy()

    for step in range(1, n_hours + 1):
        wx = (weather_forecast[step - 1]
              if weather_forecast and step <= len(weather_forecast)
              else {})

        X_row = _build_future_row(current_row, step, wx)
        X_feat = X_row[feature_cols]

        if model_name == "Ridge":
            X_feat_scaled = scaler.transform(X_feat)
            pred = float(model.predict(X_feat_scaled)[0])
        elif model_name == "LSTM":
            # LSTM needs sequence — use last 24 rows + current
            seq_df = pd.concat([history.tail(23), X_row], ignore_index=True)
            seq_arr = seq_df[feature_cols].values[-24:]
            if seq_arr.shape[0] < 24:
                pred = float(current_row.get("aqi", 80))
            else:
                pred = float(model.predict(seq_arr[np.newaxis, ...])[0][0])
        else:
            pred = float(model.predict(X_feat)[0])

        pred = max(0, min(500, round(pred)))

        category, color, emoji = get_aqi_category(pred)
        ts = pd.Timestamp(last_row["fetched_at"]) + timedelta(hours=step)

        predictions.append({
            "timestamp":     ts,
            "predicted_aqi": pred,
            "category":      category,
            "color":         color,
            "emoji":         emoji,
            "day":           ts.strftime("%A, %b %d"),
            "hour":          ts.strftime("%H:00"),
        })

        # Update current_row with prediction for next step's lag features
        current_row = X_row.iloc[0].copy()
        current_row["aqi"]       = pred
        current_row["aqi_lag_1"] = pred

    df = pd.DataFrame(predictions)
    logger.success(f"Generated {len(df)} hourly AQI predictions.")
    return df


def get_daily_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly predictions into daily summaries."""
    predictions_df["date"] = pd.to_datetime(predictions_df["timestamp"]).dt.date
    daily = predictions_df.groupby("date").agg(
        avg_aqi=("predicted_aqi", "mean"),
        max_aqi=("predicted_aqi", "max"),
        min_aqi=("predicted_aqi", "min"),
    ).reset_index()
    daily["avg_aqi"] = daily["avg_aqi"].round(1)
    daily["category"] = daily["avg_aqi"].apply(
        lambda x: get_aqi_category(x)[0]
    )
    daily["color"] = daily["avg_aqi"].apply(lambda x: get_aqi_category(x)[1])
    daily["emoji"] = daily["avg_aqi"].apply(lambda x: get_aqi_category(x)[2])
    return daily


if __name__ == "__main__":
    preds = predict_next_hours(n_hours=72)
    if not preds.empty:
        daily = get_daily_summary(preds)
        print("\n📅 3-Day AQI Forecast:")
        print(daily.to_string(index=False))
