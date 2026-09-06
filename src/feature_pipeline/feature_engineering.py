"""
src/feature_pipeline/feature_engineering.py
Computes time-based, lag, rolling, and derived features from raw data.
"""
import pandas as pd
import numpy as np
from loguru import logger


def add_time_features(df: pd.DataFrame, timestamp_col: str = "fetched_at") -> pd.DataFrame:
    """Extract hour, day, month, weekday features from timestamp."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["hour"]        = df[timestamp_col].dt.hour
    df["day_of_week"] = df[timestamp_col].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]       = df[timestamp_col].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    logger.debug("Time features added.")
    return df


def add_lag_features(df: pd.DataFrame, aqi_col: str = "aqi",
                     lags: list = [1, 3, 6, 24]) -> pd.DataFrame:
    """Add lagged AQI features (requires sorted time-series DataFrame)."""
    df = df.copy().sort_values("fetched_at").reset_index(drop=True)
    for lag in lags:
        df[f"aqi_lag_{lag}"] = df[aqi_col].shift(lag)
    logger.debug(f"Lag features added: {lags}")
    return df


def add_rolling_features(df: pd.DataFrame, aqi_col: str = "aqi",
                         windows: list = [3, 6]) -> pd.DataFrame:
    """Add rolling mean features over given windows."""
    df = df.copy()
    for w in windows:
        df[f"aqi_rolling_mean_{w}"] = (
            df[aqi_col].rolling(window=w, min_periods=1).mean()
        )
    logger.debug(f"Rolling features added: windows={windows}")
    return df


def add_change_rate(df: pd.DataFrame, aqi_col: str = "aqi") -> pd.DataFrame:
    """Compute percentage change rate of AQI between consecutive readings."""
    df = df.copy()
    df["aqi_change_rate"] = df[aqi_col].pct_change().fillna(0)
    logger.debug("AQI change rate feature added.")
    return df


def fill_missing_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then zero-fill missing pollutant values."""
    pollutant_cols = ["pm25", "pm10", "no2", "o3", "co"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Expects a DataFrame with at minimum: fetched_at, aqi columns.
    """
    logger.info(f"Starting feature engineering on {len(df)} rows...")
    df = fill_missing_pollutants(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_rate(df)
    df = df.dropna(subset=["aqi_lag_24"])   # need at least 24 hours of history
    logger.info(f"Feature engineering complete. Output shape: {df.shape}")
    return df


def engineer_single_record(record: dict, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for a single new record given historical context.
    Appends the new record to history, engineers, returns the last row.
    """
    new_row = pd.DataFrame([record])
    combined = pd.concat([history_df, new_row], ignore_index=True)
    combined = engineer_features(combined)
    return combined.tail(1)


if __name__ == "__main__":
    # Smoke test with synthetic data
    dates = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame({
        "fetched_at": dates,
        "aqi":        np.random.randint(50, 200, 48),
        "pm25":       np.random.uniform(10, 80, 48),
        "pm10":       np.random.uniform(20, 100, 48),
        "no2":        np.random.uniform(5, 50, 48),
        "o3":         np.random.uniform(10, 60, 48),
        "co":         np.random.uniform(0.5, 2.0, 48),
        "temperature": np.random.uniform(20, 40, 48),
        "humidity":   np.random.uniform(40, 90, 48),
        "wind_speed": np.random.uniform(0, 10, 48),
        "pressure":   np.random.uniform(1005, 1020, 48),
    })
    result = engineer_features(df)
    print(result[["fetched_at", "aqi", "hour", "aqi_lag_1", "aqi_rolling_mean_3", "aqi_change_rate"]].tail())
