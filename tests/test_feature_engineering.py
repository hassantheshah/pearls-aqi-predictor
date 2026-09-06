"""
tests/test_feature_engineering.py
Unit tests for feature engineering pipeline.
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_pipeline.feature_engineering import (
    add_time_features, add_lag_features,
    add_rolling_features, add_change_rate,
    fill_missing_pollutants, engineer_features,
)
from config.settings import get_aqi_category


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_sample_df(n: int = 48) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    np.random.seed(0)
    return pd.DataFrame({
        "city":        "Karachi",
        "fetched_at":  dates,
        "aqi":         np.random.randint(40, 180, n),
        "pm25":        np.random.uniform(10, 80, n),
        "pm10":        np.random.uniform(20, 100, n),
        "no2":         np.random.uniform(5, 50, n),
        "o3":          np.random.uniform(10, 60, n),
        "co":          np.random.uniform(0.5, 2.5, n),
        "temperature": np.random.uniform(20, 40, n),
        "humidity":    np.random.uniform(40, 90, n),
        "wind_speed":  np.random.uniform(0, 10, n),
        "pressure":    np.random.uniform(1005, 1020, n),
    })


# ── Time Features ─────────────────────────────────────────────────────────────

def test_add_time_features():
    df = make_sample_df(10)
    result = add_time_features(df)
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns
    assert result["hour"].between(0, 23).all()
    assert result["day_of_week"].between(0, 6).all()
    assert result["month"].between(1, 12).all()
    assert result["is_weekend"].isin([0, 1]).all()


# ── Lag Features ──────────────────────────────────────────────────────────────

def test_add_lag_features():
    df = make_sample_df(30)
    result = add_lag_features(df, lags=[1, 3, 6])
    assert "aqi_lag_1" in result.columns
    assert "aqi_lag_3" in result.columns
    assert "aqi_lag_6" in result.columns
    # First 6 rows of lag_6 should be NaN
    assert result["aqi_lag_6"].iloc[:6].isna().all()


# ── Rolling Features ──────────────────────────────────────────────────────────

def test_add_rolling_features():
    df = make_sample_df(20)
    result = add_rolling_features(df, windows=[3, 6])
    assert "aqi_rolling_mean_3" in result.columns
    assert "aqi_rolling_mean_6" in result.columns
    # Rolling mean should never exceed max AQI
    assert result["aqi_rolling_mean_3"].max() <= df["aqi"].max() + 1


# ── Change Rate ───────────────────────────────────────────────────────────────

def test_add_change_rate():
    df = make_sample_df(20)
    result = add_change_rate(df)
    assert "aqi_change_rate" in result.columns
    assert result["aqi_change_rate"].iloc[0] == 0.0  # first row is 0


# ── Missing Pollutants ────────────────────────────────────────────────────────

def test_fill_missing_pollutants():
    df = make_sample_df(10)
    df.loc[2:5, "pm25"] = np.nan
    result = fill_missing_pollutants(df)
    assert result["pm25"].isna().sum() == 0


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def test_engineer_features_output_shape():
    df = make_sample_df(50)
    result = engineer_features(df)
    # Should have dropped rows with NaN lag_24
    assert len(result) > 0
    assert len(result) <= len(df)


def test_engineer_features_columns():
    df = make_sample_df(50)
    result = engineer_features(df)
    expected = ["hour", "day_of_week", "month", "is_weekend",
                "aqi_lag_1", "aqi_lag_24", "aqi_rolling_mean_3", "aqi_change_rate"]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"


# ── AQI Category ─────────────────────────────────────────────────────────────

def test_get_aqi_category():
    assert get_aqi_category(25)[0]  == "Good"
    assert get_aqi_category(75)[0]  == "Moderate"
    assert get_aqi_category(125)[0] == "Unhealthy for Sensitive"
    assert get_aqi_category(175)[0] == "Unhealthy"
    assert get_aqi_category(250)[0] == "Very Unhealthy"
    assert get_aqi_category(400)[0] == "Hazardous"


def test_aqi_category_color_format():
    _, color, emoji = get_aqi_category(50)
    assert color.startswith("#")
    assert len(emoji) >= 1


# ── Edge Cases ────────────────────────────────────────────────────────────────

def test_engineer_features_empty_df():
    df = pd.DataFrame(columns=["fetched_at", "aqi", "pm25"])
    result = engineer_features(df)
    assert isinstance(result, pd.DataFrame)


def test_engineer_features_minimal_data():
    """With fewer than 24 rows, all lag_24 will be NaN → result is empty."""
    df = make_sample_df(20)
    result = engineer_features(df)
    assert isinstance(result, pd.DataFrame)
