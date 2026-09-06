"""
tests/test_forecasting.py

Tests for the one-hour-ahead AQI forecasting contract.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
    ),
)

from config.settings import FEATURE_COLUMNS
from src.training_pipeline.train import load_data
from src.feature_pipeline.fetch_data import fetch_openmeteo_historical_data
from src.inference_pipeline.predict import (
    _calculate_aqi_features,
    _build_future_row,
)


def make_training_df(n=30):
    dates = pd.date_range(
        "2026-01-01",
        periods=n,
        freq="h",
    )

    data = {
        "fetched_at": dates,
        "aqi": np.arange(50, 50 + n, dtype=float),
        "temperature": np.full(n, 25.0),
        "humidity": np.full(n, 60.0),
        "wind_speed": np.full(n, 3.0),
        "pressure": np.full(n, 1013.0),
        "hour": dates.hour,
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "is_weekend": (dates.dayofweek >= 5).astype(int),
        "aqi_lag_1": np.arange(49, 49 + n, dtype=float),
        "aqi_lag_3": np.arange(47, 47 + n, dtype=float),
        "aqi_lag_6": np.arange(44, 44 + n, dtype=float),
        "aqi_lag_24": np.arange(26, 26 + n, dtype=float),
        "aqi_rolling_mean_3": np.full(n, 50.0),
        "aqi_rolling_mean_6": np.full(n, 50.0),
        "aqi_change_rate": np.zeros(n),
    }

    return pd.DataFrame(data)


def test_aqi_is_not_a_prediction_feature():
    assert "aqi" not in FEATURE_COLUMNS


def test_future_pollutants_are_not_prediction_features():
    forbidden = {
        "pm25",
        "pm10",
        "no2",
        "o3",
        "co",
    }

    assert forbidden.isdisjoint(set(FEATURE_COLUMNS))


def test_load_data_shifts_target_one_hour_forward(tmp_path):
    df = make_training_df()

    file_path = tmp_path / "features.csv"
    df.to_csv(file_path, index=False)

    result = load_data(str(file_path))

    assert "target_aqi" in result.columns

    # AQI at time t should predict AQI at time t+1.
    assert result.loc[0, "aqi"] == 50.0
    assert result.loc[0, "target_aqi"] == 51.0

    # Final row has no future target and must be removed.
    assert len(result) == len(df) - 1


def test_aqi_history_features_use_only_history():
    history = [
        100.0,
        110.0,
        120.0,
        130.0,
        140.0,
        150.0,
    ]

    features = _calculate_aqi_features(history)

    assert features["aqi_lag_1"] == 150.0
    assert features["aqi_lag_3"] == 130.0
    assert features["aqi_lag_6"] == 100.0

    assert features["aqi_rolling_mean_3"] == 140.0
    assert features["aqi_rolling_mean_6"] == 125.0

    expected_change = (150.0 - 140.0) / 140.0
    assert np.isclose(
        features["aqi_change_rate"],
        expected_change,
    )


def test_future_row_contains_only_deployable_features():
    current_time = pd.Timestamp(
        "2026-01-03 15:00:00"
    )

    history = [
        float(value)
        for value in range(100, 130)
    ]

    weather = {
        "temperature": 27.0,
        "humidity": 70.0,
        "wind_speed": 4.0,
        "pressure": 1010.0,
    }

    row = _build_future_row(
        current_time=current_time,
        aqi_history=history,
        weather=weather,
        last_weather=weather,
    )

    assert list(row.columns) == FEATURE_COLUMNS

    assert "aqi" not in row.columns
    assert "pm25" not in row.columns
    assert "pm10" not in row.columns
    assert "no2" not in row.columns
    assert "o3" not in row.columns
    assert "co" not in row.columns


def test_recursive_history_changes_after_new_prediction():
    history = [100.0, 110.0, 120.0]

    features_before = _calculate_aqi_features(history)

    history.append(130.0)

    features_after = _calculate_aqi_features(history)

    assert features_before["aqi_lag_1"] == 120.0
    assert features_after["aqi_lag_1"] == 130.0

    assert (
        features_before["aqi_rolling_mean_3"]
        != features_after["aqi_rolling_mean_3"]
    )


def test_openmeteo_historical_data_is_merged(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    air_payload = {
        "hourly": {
            "time": [
                "2026-08-01T00:00",
                "2026-08-01T01:00",
            ],
            "us_aqi": [80, 85],
            "pm2_5": [20, 22],
            "pm10": [35, 38],
            "nitrogen_dioxide": [10, 11],
            "ozone": [40, 42],
            "carbon_monoxide": [0.5, 0.6],
        }
    }

    weather_payload = {
        "hourly": {
            "time": [
                "2026-08-01T00:00",
                "2026-08-01T01:00",
            ],
            "temperature_2m": [28.0, 28.5],
            "relative_humidity_2m": [70, 72],
            "wind_speed_10m": [3.0, 3.5],
            "pressure_msl": [1010, 1011],
        }
    }

    responses = iter(
        [
            FakeResponse(air_payload),
            FakeResponse(weather_payload),
        ]
    )

    def fake_get(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(
        "src.feature_pipeline.fetch_data.requests.get",
        fake_get,
    )

    result = fetch_openmeteo_historical_data(
        "2026-08-01",
        "2026-08-01",
    )

    assert len(result) == 2
    assert result["city"].notna().all()

    assert list(result["aqi"]) == [80, 85]
    assert list(result["pm25"]) == [20, 22]
    assert list(result["pm10"]) == [35, 38]

    assert list(result["temperature"]) == [28.0, 28.5]
    assert list(result["humidity"]) == [70, 72]
    assert list(result["wind_speed"]) == [3.0, 3.5]
    assert list(result["pressure"]) == [1010, 1011]

    expected_columns = [
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

    assert list(result.columns) == expected_columns
