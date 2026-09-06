"""
src/feature_pipeline/pipeline.py
Main feature pipeline — fetches, engineers, stores features.
Run hourly via GitHub Actions.
"""
import sys
import os
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.feature_pipeline.fetch_data import fetch_combined_current
from src.feature_pipeline.feature_engineering import engineer_features
from src.feature_pipeline.feature_store import insert_features, save_local, _read_local_fallback


LOCAL_STORE_PATH = "data/processed/features.csv"


def run_feature_pipeline(backfill: bool = False,
                         backfill_days: int = 90) -> pd.DataFrame:
    """
    Main feature pipeline entry point.

    Args:
        backfill: If True, generate synthetic backfill data for training.
        backfill_days: Number of past days to backfill (only when backfill=True).

    Returns:
        Engineered DataFrame that was stored.
    """
    if backfill:
        logger.info(f"Running BACKFILL mode for {backfill_days} days...")
        df_raw = _generate_backfill(backfill_days)
    else:
        logger.info("Running REAL-TIME feature pipeline...")
        record = fetch_combined_current()
        if not record:
            logger.error("Empty record from API — pipeline aborted.")
            return pd.DataFrame()

        # Load existing history for lag feature computation
        existing = _read_local_fallback(LOCAL_STORE_PATH)
        if existing.empty:
            df_raw = pd.DataFrame([record])
        else:
            df_raw = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)

    # Engineer features
    df_features = engineer_features(df_raw)

    if df_features.empty:
        logger.warning("Feature engineering returned empty DataFrame.")
        return df_features

    # Store locally (always) and in Hopsworks (if key available)
    save_local(df_features, LOCAL_STORE_PATH)

    try:
        insert_features(df_features.tail(1) if not backfill else df_features)
    except Exception as e:
        logger.warning(f"Hopsworks insert skipped (will use local): {e}")

    logger.success(f"Feature pipeline complete. {len(df_features)} total rows stored.")
    return df_features


def _generate_backfill(days: int) -> pd.DataFrame:
    """Generate synthetic historical data for backfill / offline testing."""
    import numpy as np
    from config.settings import CITY_NAME

    hours = days * 24
    dates = pd.date_range(end=datetime.utcnow(), periods=hours, freq="h")
    np.random.seed(42)

    # Simulate realistic AQI pattern with daily cycle
    hour_arr = np.array(dates.hour)
    daily_pattern = 30 * np.sin((hour_arr - 6) * np.pi / 12)
    base_aqi = 80 + daily_pattern + np.random.normal(0, 15, hours)
    base_aqi = np.clip(base_aqi, 20, 300)

    df = pd.DataFrame({
        "city":        CITY_NAME,
        "fetched_at":  dates,
        "aqi":         base_aqi.astype(int),
        "pm25":        np.clip(base_aqi * 0.4 + np.random.normal(0, 5, hours), 0, None),
        "pm10":        np.clip(base_aqi * 0.6 + np.random.normal(0, 8, hours), 0, None),
        "no2":         np.random.uniform(5, 60, hours),
        "o3":          np.random.uniform(10, 80, hours),
        "co":          np.random.uniform(0.3, 2.5, hours),
        "temperature": (25 + 8 * np.sin((hour_arr - 14) * np.pi / 12)
                        + np.random.normal(0, 2, hours)),
        "humidity":    np.random.uniform(40, 90, hours),
        "wind_speed":  np.abs(np.random.normal(3, 2, hours)),
        "pressure":    np.random.uniform(1008, 1018, hours),
    })
    logger.info(f"Generated {len(df)} backfill rows.")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pearls AQI Feature Pipeline")
    parser.add_argument("--backfill", action="store_true",
                        help="Run in backfill mode")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of days to backfill")
    args = parser.parse_args()
    run_feature_pipeline(backfill=args.backfill, backfill_days=args.days)
