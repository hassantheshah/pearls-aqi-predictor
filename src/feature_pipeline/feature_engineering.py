"""
src/feature_pipeline/feature_engineering.py

Leakage-safe feature engineering for AQI forecasting.
"""

import numpy as np
import pandas as pd
from loguru import logger


def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "fetched_at",
) -> pd.DataFrame:
    """Extract calendar/time features from the timestamp."""
    df = df.copy()

    if df.empty:
        df["hour"] = pd.Series(dtype="int64")
        df["day_of_week"] = pd.Series(dtype="int64")
        df["month"] = pd.Series(dtype="int64")
        df["is_weekend"] = pd.Series(dtype="int64")
        return df

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    df["hour"] = df[timestamp_col].dt.hour
    df["day_of_week"] = df[timestamp_col].dt.dayofweek
    df["month"] = df[timestamp_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def add_lag_features(
    df: pd.DataFrame,
    aqi_col: str = "aqi",
    lags: list = [1, 3, 6, 24],
) -> pd.DataFrame:
    """Add leakage-safe historical AQI and pollutant lag features."""

    df = df.copy().sort_values("fetched_at").reset_index(drop=True)

    # Historical AQI features
    for lag in lags:
        df[f"aqi_lag_{lag}"] = df[aqi_col].shift(lag)

    # Historical pollutant features.
    # These use only past observations, never the current/future value.
    pollutant_cols = ["pm25", "pm10", "no2", "o3", "co"]

    for pollutant in pollutant_cols:
        if pollutant not in df.columns:
            continue

        for lag in lags:
            df[f"{pollutant}_lag_{lag}"] = df[pollutant].shift(lag)

    logger.debug(
        f"AQI and pollutant lag features added: lags={lags}"
    )

    return df


def add_rolling_features(
    df: pd.DataFrame,
    aqi_col: str = "aqi",
    windows: list = [3, 6],
) -> pd.DataFrame:
    """
    Add leakage-safe rolling AQI means.

    The current AQI is excluded by shifting before calculating
    the rolling mean.
    """
    df = df.copy()

    for window in windows:
        df[f"aqi_rolling_mean_{window}"] = (
            df[aqi_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

    return df


def add_change_rate(
    df: pd.DataFrame,
    aqi_col: str = "aqi",
) -> pd.DataFrame:
    """
    Calculate AQI percentage change using only historical AQI values.

    For row t, the feature uses AQI at t-1 and t-2,
    never the target AQI at t.
    """
    df = df.copy()

    if df.empty:
        df["aqi_change_rate"] = pd.Series(dtype="float64")
        return df

    previous_aqi = df[aqi_col].shift(1)
    two_steps_back = df[aqi_col].shift(2)

    df["aqi_change_rate"] = (
        (previous_aqi - two_steps_back)
        / two_steps_back.replace(0, np.nan)
    )

    df["aqi_change_rate"] = (
        df["aqi_change_rate"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return df


def fill_missing_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then zero-fill missing pollutant values."""
    df = df.copy()

    pollutant_cols = ["pm25", "pm10", "no2", "o3", "co"]

    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete leakage-safe feature engineering pipeline.

    Requires at minimum:
        fetched_at
        aqi

    Rows without 24 hours of AQI history are removed because
    aqi_lag_24 is required for forecasting.
    """
    logger.info(f"Starting feature engineering on {len(df)} rows...")

    df = df.copy()

    if df.empty:
        # Return a valid empty DataFrame without triggering
        # pandas/numpy errors.
        df = add_time_features(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = add_change_rate(df)
        return df

    df["fetched_at"] = pd.to_datetime(
        df["fetched_at"],
        utc=True,
    )

    df = df.sort_values("fetched_at").reset_index(drop=True)

    df = fill_missing_pollutants(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_rate(df)

    # At least 24 hours of history is required.
    df = df.dropna(subset=["aqi_lag_24"]).reset_index(drop=True)

    logger.info(
        f"Feature engineering complete. Output shape: {df.shape}"
    )

    return df


def engineer_single_record(
    record: dict,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Engineer features for one new record using historical context.
    """
    new_row = pd.DataFrame([record])

    combined = pd.concat(
        [history_df, new_row],
        ignore_index=True,
    )

    combined = engineer_features(combined)

    return combined.tail(1)


if __name__ == "__main__":
    # Smoke test with synthetic data.
    dates = pd.date_range(
        "2024-01-01",
        periods=48,
        freq="h",
        tz="UTC",
    )

    df = pd.DataFrame(
        {
            "fetched_at": dates,
            "aqi": np.random.randint(50, 200, 48),
            "pm25": np.random.uniform(10, 80, 48),
            "pm10": np.random.uniform(20, 100, 48),
            "no2": np.random.uniform(5, 50, 48),
            "o3": np.random.uniform(10, 60, 48),
            "co": np.random.uniform(0.5, 2.0, 48),
            "temperature": np.random.uniform(20, 40, 48),
            "humidity": np.random.uniform(40, 90, 48),
            "wind_speed": np.random.uniform(0, 10, 48),
            "pressure": np.random.uniform(1005, 1020, 48),
        }
    )

    result = engineer_features(df)

    print(
        result[
            [
                "fetched_at",
                "aqi",
                "hour",
                "aqi_lag_1",
                "aqi_lag_24",
                "aqi_rolling_mean_3",
                "aqi_change_rate",
            ]
        ].tail()
    )
