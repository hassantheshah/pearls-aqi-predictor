"""
src/api/app.py
Flask REST API — serves AQI predictions, current data, and alerts.
"""
import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.settings import CITY_NAME, ALERT_THRESHOLD, get_aqi_category

app = Flask(__name__)
CORS(app)


def _get_predictions():
    """Helper to get predictions with graceful fallback."""
    try:
        from src.inference_pipeline.predict import predict_next_hours, get_daily_summary
        preds = predict_next_hours(n_hours=72)
        daily = get_daily_summary(preds)
        return preds, daily
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None, None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "city": CITY_NAME}), 200


@app.route("/api/current", methods=["GET"])
def current_aqi():
    """Return the most recent AQI reading."""
    try:
        from src.feature_pipeline.feature_store import read_latest_features
        df = read_latest_features(n_rows=1)
        if df.empty:
            return jsonify({"error": "No data available"}), 404
        row = df.iloc[-1]
        aqi_val = float(row.get("aqi", 0))
        category, color, emoji = get_aqi_category(aqi_val)
        return jsonify({
            "city":        CITY_NAME,
            "aqi":         aqi_val,
            "category":    category,
            "color":       color,
            "emoji":       emoji,
            "pm25":        float(row.get("pm25", 0)),
            "pm10":        float(row.get("pm10", 0)),
            "no2":         float(row.get("no2",  0)),
            "o3":          float(row.get("o3",   0)),
            "temperature": float(row.get("temperature", 0)),
            "humidity":    float(row.get("humidity", 0)),
            "wind_speed":  float(row.get("wind_speed", 0)),
            "timestamp":   str(row.get("fetched_at", "")),
        })
    except Exception as e:
        logger.error(f"/api/current error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast", methods=["GET"])
def forecast():
    """Return hourly AQI forecast for the next 72 hours."""
    preds, _ = _get_predictions()
    if preds is None:
        return jsonify({"error": "Prediction unavailable"}), 500
    return jsonify({
        "city":     CITY_NAME,
        "forecast": preds.to_dict(orient="records")
    })


@app.route("/api/forecast/daily", methods=["GET"])
def forecast_daily():
    """Return 3-day daily AQI summary."""
    _, daily = _get_predictions()
    if daily is None:
        return jsonify({"error": "Prediction unavailable"}), 500
    daily["date"] = daily["date"].astype(str)
    return jsonify({
        "city":  CITY_NAME,
        "daily": daily.to_dict(orient="records")
    })


@app.route("/api/alerts", methods=["GET"])
def alerts():
    """Return alert if AQI is forecast to exceed hazardous threshold."""
    preds, daily = _get_predictions()
    if preds is None:
        return jsonify({"alerts": []})

    triggered = preds[preds["predicted_aqi"] >= ALERT_THRESHOLD]
    alert_list = []
    if not triggered.empty:
        max_aqi = int(triggered["predicted_aqi"].max())
        first_ts = str(triggered.iloc[0]["timestamp"])
        category, color, emoji = get_aqi_category(max_aqi)
        alert_list.append({
            "type":      "HAZARDOUS_AQI",
            "message":   f"AQI forecast to reach {max_aqi} ({category}) at {first_ts}",
            "max_aqi":   max_aqi,
            "color":     color,
            "emoji":     emoji,
            "threshold": ALERT_THRESHOLD,
        })
    return jsonify({"city": CITY_NAME, "alerts": alert_list})


@app.route("/api/history", methods=["GET"])
def history():
    """Return recent historical AQI readings (last N hours)."""
    hours = int(request.args.get("hours", 48))
    try:
        from src.feature_pipeline.feature_store import read_latest_features
        df = read_latest_features(n_rows=hours)
        if df.empty:
            return jsonify({"history": []})
        cols = ["fetched_at", "aqi", "pm25", "pm10", "no2",
                "temperature", "humidity", "wind_speed"]
        cols = [c for c in cols if c in df.columns]
        df[cols] = df[cols].fillna(0)
        df["fetched_at"] = df["fetched_at"].astype(str)
        return jsonify({"city": CITY_NAME, "history": df[cols].to_dict(orient="records")})
    except Exception as e:
        logger.error(f"/api/history error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def model_metrics():
    """Return model evaluation metrics."""
    try:
        import pandas as pd
        metrics = pd.read_csv("models/metrics.csv")
        return jsonify({"metrics": metrics.to_dict(orient="records")})
    except Exception:
        return jsonify({"metrics": []})


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    logger.info(f"Starting Flask API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
