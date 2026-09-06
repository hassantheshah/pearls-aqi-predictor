"""
src/inference_pipeline/predict.py

Generate recursive one-hour-ahead AQI forecasts.

The model is trained as:

    features at time t -> AQI at time t+1

Forecasting is recursive:

    prediction t+1 becomes history
    prediction t+2 uses that updated history
    prediction t+3 uses the updated history
    ...
"""

import os
import sys
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    ),
)

from config.settings import (
    FEATURE_COLUMNS,
    FORECAST_DAYS,
    get_aqi_category,
)

from src.feature_pipeline.feature_store import (
    read_latest_features,
)


MODEL_DIR = "models"

FEATURE_COLS = list(FEATURE_COLUMNS)


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────


def load_best_model():
    """Load the best trained model and required preprocessing."""

    best_model_path = f"{MODEL_DIR}/best_model.txt"

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            "best_model.txt not found. "
            "Run the training pipeline first."
        )

    with open(
        best_model_path,
        "r",
    ) as file:
        best_name = file.read().strip()

    logger.info(
        f"Loading best model: {best_name}"
    )

    if best_name == "RandomForest":

        model = joblib.load(
            f"{MODEL_DIR}/random_forest.pkl"
        )

        scaler = None

    elif best_name == "Ridge":

        model = joblib.load(
            f"{MODEL_DIR}/ridge.pkl"
        )

        scaler = joblib.load(
            f"{MODEL_DIR}/scaler.pkl"
        )

    elif best_name == "LSTM":

        import tensorflow as tf

        model = tf.keras.models.load_model(
            f"{MODEL_DIR}/lstm_model.keras"
        )

        scaler = {
            "x": joblib.load(
                f"{MODEL_DIR}/lstm_x_scaler.pkl"
            ),
            "y": joblib.load(
                f"{MODEL_DIR}/lstm_y_scaler.pkl"
            ),
        }

    else:

        raise ValueError(
            f"Unknown model: {best_name}"
        )

    feature_cols = joblib.load(
        f"{MODEL_DIR}/feature_cols.pkl"
    )

    return (
        model,
        scaler,
        best_name,
        feature_cols,
    )


# ─────────────────────────────────────────────────────────────
# AQI history features
# ─────────────────────────────────────────────────────────────


def _calculate_aqi_features(
    aqi_history: list,
    pollutant_history: dict | None = None,
) -> dict:
    """
    Calculate leakage-safe historical AQI and pollutant features.

    Only previously observed AQI and pollutant values are used.
    """

    if not aqi_history:
        aqi_history = [80.0]

    pollutant_history = pollutant_history or {}

    lags = [1, 3, 6, 24]

    features = {}

    # Historical AQI features
    for lag in lags:

        if len(aqi_history) >= lag:
            value = aqi_history[-lag]
        else:
            value = aqi_history[0]

        features[f"aqi_lag_{lag}"] = float(value)

    # Rolling AQI features
    features["aqi_rolling_mean_3"] = float(
        np.mean(aqi_history[-3:])
    )

    features["aqi_rolling_mean_6"] = float(
        np.mean(aqi_history[-6:])
    )

    # AQI change rate
    if len(aqi_history) >= 2:

        previous = float(
            aqi_history[-2]
        )

        current = float(
            aqi_history[-1]
        )

        if previous != 0:
            change_rate = (
                current - previous
            ) / previous

        else:
            change_rate = 0.0

    else:

        change_rate = 0.0

    if not np.isfinite(change_rate):
        change_rate = 0.0

    features["aqi_change_rate"] = float(
        change_rate
    )

    # Historical pollutant features
    pollutant_cols = [
        "pm25",
        "pm10",
        "no2",
        "o3",
        "co",
    ]

    for pollutant in pollutant_cols:

        history = pollutant_history.get(
            pollutant,
            [],
        )

        if not history:
            history = [0.0]

        for lag in lags:

            if len(history) >= lag:
                value = history[-lag]

            else:
                value = history[0]

            features[
                f"{pollutant}_lag_{lag}"
            ] = float(value)

    return features


# ─────────────────────────────────────────────────────────────
# Future feature row
# ─────────────────────────────────────────────────────────────


def _build_future_row(
    current_time: pd.Timestamp,
    aqi_history: list,
    pollutant_history: dict,
    weather: dict,
    last_weather: dict,
) -> pd.DataFrame:
    """
    Build features for the next forecast hour.

    Uses only:

    - future weather
    - calendar features
    - historical AQI
    - historical pollutant values

    No future AQI or future pollutant measurement is used.
    """

    weather = weather or {}

    last_weather = last_weather or {}

    temperature = weather.get(
        "temperature",
        last_weather.get(
            "temperature",
            25.0,
        ),
    )

    humidity = weather.get(
        "humidity",
        last_weather.get(
            "humidity",
            60.0,
        ),
    )

    wind_speed = weather.get(
        "wind_speed",
        last_weather.get(
            "wind_speed",
            3.0,
        ),
    )

    pressure = weather.get(
        "pressure",
        last_weather.get(
            "pressure",
            1013.0,
        ),
    )

    aqi_features = _calculate_aqi_features(
        aqi_history=aqi_history,
        pollutant_history=pollutant_history,
    )

    row = {
        "temperature": float(
            temperature
        ),
        "humidity": float(
            humidity
        ),
        "wind_speed": float(
            wind_speed
        ),
        "pressure": float(
            pressure
        ),
        "hour": int(
            current_time.hour
        ),
        "day_of_week": int(
            current_time.dayofweek
        ),
        "month": int(
            current_time.month
        ),
        "is_weekend": int(
            current_time.dayofweek >= 5
        ),
    }

    row.update(aqi_features)

    return pd.DataFrame(
        [
            [
                row[column]
                for column in FEATURE_COLUMNS
            ]
        ],
        columns=FEATURE_COLUMNS,
    )


# ─────────────────────────────────────────────────────────────
# Historical feature row for LSTM
# ─────────────────────────────────────────────────────────────


def _build_historical_feature_row(
    history: pd.DataFrame,
    row_index,
) -> dict:
    """
    Build one leakage-safe historical feature row.

    All AQI and pollutant features use observations
    available at or before this historical timestamp.
    """

    hist_row = history.loc[row_index]

    # AQI history up to this timestamp
    aqi_history = (
        pd.to_numeric(
            history.loc[
                :row_index,
                "aqi",
            ],
            errors="coerce",
        )
        .dropna()
        .tolist()
    )

    # Pollutant history up to this timestamp
    pollutant_history = {}

    for pollutant in [
        "pm25",
        "pm10",
        "no2",
        "o3",
        "co",
    ]:

        if pollutant in history.columns:

            pollutant_history[pollutant] = (
                pd.to_numeric(
                    history.loc[
                        :row_index,
                        pollutant,
                    ],
                    errors="coerce",
                )
                .dropna()
                .tolist()
            )

        else:

            pollutant_history[pollutant] = [
                0.0
            ]

    hist_aqi_features = _calculate_aqi_features(
        aqi_history=aqi_history,
        pollutant_history=pollutant_history,
    )

    timestamp = pd.Timestamp(
        hist_row["fetched_at"]
    )

    historical_feature = {
        "temperature": float(
            hist_row.get(
                "temperature",
                25.0,
            )
        ),
        "humidity": float(
            hist_row.get(
                "humidity",
                60.0,
            )
        ),
        "wind_speed": float(
            hist_row.get(
                "wind_speed",
                3.0,
            )
        ),
        "pressure": float(
            hist_row.get(
                "pressure",
                1013.0,
            )
        ),
        "hour": int(
            timestamp.hour
        ),
        "day_of_week": int(
            timestamp.dayofweek
        ),
        "month": int(
            timestamp.month
        ),
        "is_weekend": int(
            timestamp.dayofweek >= 5
        ),
    }

    historical_feature.update(
        hist_aqi_features
    )

    return historical_feature


# ─────────────────────────────────────────────────────────────
# Model prediction
# ─────────────────────────────────────────────────────────────


def _predict_one(
    model,
    scaler,
    model_name: str,
    feature_row: pd.DataFrame,
    sequence_history: list | None = None,
) -> float:
    """Predict the next-hour AQI."""

    X = feature_row[
        FEATURE_COLS
    ]

    if model_name == "RandomForest":

        prediction = model.predict(
            X
        )[0]

    elif model_name == "Ridge":

        X_scaled = scaler.transform(
            X
        )

        prediction = model.predict(
            X_scaled
        )[0]

    elif model_name == "LSTM":

        if sequence_history is None:
            raise ValueError(
                "LSTM requires sequence history."
            )

        if len(sequence_history) < 24:
            raise ValueError(
                "LSTM requires at least "
                "24 historical feature rows."
            )

        x_scaler = scaler["x"]

        y_scaler = scaler["y"]

        sequence_df = pd.DataFrame(
            sequence_history[-24:]
        )

        sequence = sequence_df[
            FEATURE_COLS
        ]

        sequence_scaled = x_scaler.transform(
            sequence
        )

        sequence_scaled = (
            sequence_scaled
            .reshape(
                1,
                24,
                len(FEATURE_COLS),
            )
        )

        prediction_scaled = model.predict(
            sequence_scaled,
            verbose=0,
        )[0][0]

        prediction = (
            y_scaler.inverse_transform(
                np.array(
                    [[prediction_scaled]]
                )
            )[0][0]
        )

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return float(
        np.clip(
            prediction,
            0,
            500,
        )
    )


# ─────────────────────────────────────────────────────────────
# 72-hour recursive forecast
# ─────────────────────────────────────────────────────────────


def predict_next_hours(
    n_hours: int | None = None,
    weather_forecast: list | None = None,
) -> pd.DataFrame:
    """
    Generate recursive AQI predictions.

    Every prediction is appended to AQI history and becomes
    available to subsequent predictions.

    Future pollutant observations are unavailable, so the
    latest observed pollutant values are carried forward
    during recursive forecasting.
    """

    if n_hours is None:
        n_hours = (
            FORECAST_DAYS * 24
        )

    (
        model,
        scaler,
        model_name,
        trained_feature_cols,
    ) = load_best_model()

    # Make sure the saved model contract matches the
    # current application feature contract.
    if list(trained_feature_cols) != list(
        FEATURE_COLUMNS
    ):
        raise ValueError(
            "Model feature columns do not match "
            "config FEATURE_COLUMNS."
        )

    # Load recent historical context.
    history = read_latest_features(
        n_rows=48
    )

    if history.empty:

        logger.error(
            "No recent feature data available."
        )

        return pd.DataFrame()

    history = (
        history
        .sort_values("fetched_at")
        .reset_index(drop=True)
    )

    # Last known timestamp.
    last_time = pd.Timestamp(
        history.iloc[-1]["fetched_at"]
    )

    # Historical AQI values.
    aqi_history = (
        pd.to_numeric(
            history["aqi"],
            errors="coerce",
        )
        .dropna()
        .tolist()
    )

    # Historical pollutant values used to
    # build pollutant lag features.
    pollutant_history = {}

    for pollutant in [
        "pm25",
        "pm10",
        "no2",
        "o3",
        "co",
    ]:

        if pollutant in history.columns:

            pollutant_history[pollutant] = (
                pd.to_numeric(
                    history[pollutant],
                    errors="coerce",
                )
                .ffill()
                .fillna(0.0)
                .tolist()
            )

        else:

            pollutant_history[pollutant] = [
                0.0
            ]

    if not aqi_history:

        logger.error(
            "No historical AQI values available."
        )

        return pd.DataFrame()

    # Last known weather values used only
    # as fallback when future weather is unavailable.
    last_row = history.iloc[-1]

    last_weather = {
        "temperature": last_row.get(
            "temperature",
            25.0,
        ),
        "humidity": last_row.get(
            "humidity",
            60.0,
        ),
        "wind_speed": last_row.get(
            "wind_speed",
            3.0,
        ),
        "pressure": last_row.get(
            "pressure",
            1013.0,
        ),
    }

    predictions = []

    # LSTM needs 24 historical feature rows.
    sequence_history = []

    if model_name == "LSTM":

        if len(history) < 24:
            raise ValueError(
                "LSTM requires at least "
                "24 historical rows."
            )

        for row_index in history.tail(
            24
        ).index:

            historical_feature = (
                _build_historical_feature_row(
                    history=history,
                    row_index=row_index,
                )
            )

            sequence_history.append(
                historical_feature
            )

    # ─────────────────────────────────────────
    # Recursive forecast loop
    # ─────────────────────────────────────────

    for step in range(
        1,
        n_hours + 1,
    ):

        future_time = (
            last_time
            + timedelta(
                hours=step
            )
        )

        # Use supplied future weather when available.
        if (
            weather_forecast
            and step <= len(
                weather_forecast
            )
        ):

            weather = (
                weather_forecast[
                    step - 1
                ]
            )

        else:

            weather = {}

        # Build the deployable feature row.
        feature_row = _build_future_row(
            current_time=future_time,
            aqi_history=aqi_history,
            pollutant_history=pollutant_history,
            weather=weather,
            last_weather=last_weather,
        )

        # Predict AQI.
        prediction = _predict_one(
            model=model,
            scaler=scaler,
            model_name=model_name,
            feature_row=feature_row,
            sequence_history=sequence_history,
        )

        prediction = int(
            round(
                prediction
            )
        )

        category, color, emoji = (
            get_aqi_category(
                prediction
            )
        )

        predictions.append(
            {
                "timestamp": future_time,
                "predicted_aqi": prediction,
                "category": category,
                "color": color,
                "emoji": emoji,
                "day": future_time.strftime(
                    "%A, %b %d"
                ),
                "hour": future_time.strftime(
                    "%H:00"
                ),
            }
        )

        # ─────────────────────────────────────
        # CRITICAL:
        # The prediction becomes historical
        # information for the next forecast.
        # ─────────────────────────────────────

        aqi_history.append(
            prediction
        )

        # Future pollutant observations are unavailable.
        #
        # Carry the latest observed pollutant values
        # forward for subsequent recursive steps.
        for pollutant in [
            "pm25",
            "pm10",
            "no2",
            "o3",
            "co",
        ]:

            history_values = (
                pollutant_history.get(
                    pollutant,
                    [0.0],
                )
            )

            latest_value = float(
                feature_row.iloc[0][
                    f"{pollutant}_lag_1"
                ]
            )

            history_values.append(
                latest_value
            )

            if len(history_values) > 200:

                history_values = (
                    history_values[-200:]
                )

            pollutant_history[
                pollutant
            ] = history_values

        # Prevent unbounded AQI history growth.
        if len(aqi_history) > 200:

            aqi_history = (
                aqi_history[-200:]
            )

        # For LSTM, the newly constructed
        # future feature row becomes history
        # for the next recursive step.
        if model_name == "LSTM":

            sequence_history.append(
                feature_row.iloc[
                    0
                ].to_dict()
            )

            if len(sequence_history) > 200:

                sequence_history = (
                    sequence_history[-200:]
                )

        # Update weather fallback.
        last_weather = {
            "temperature": feature_row.iloc[
                0
            ]["temperature"],
            "humidity": feature_row.iloc[
                0
            ]["humidity"],
            "wind_speed": feature_row.iloc[
                0
            ]["wind_speed"],
            "pressure": feature_row.iloc[
                0
            ]["pressure"],
        }

    result = pd.DataFrame(
        predictions
    )

    logger.success(
        f"Generated {len(result)} "
        f"hourly AQI predictions."
    )

    return result


# ─────────────────────────────────────────────────────────────
# Daily summaries
# ─────────────────────────────────────────────────────────────


def get_daily_summary(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate hourly predictions into daily summaries."""

    if predictions_df.empty:
        return pd.DataFrame()

    df = predictions_df.copy()

    df["date"] = pd.to_datetime(
        df["timestamp"]
    ).dt.date

    daily = (
        df.groupby("date")
        .agg(
            avg_aqi=(
                "predicted_aqi",
                "mean",
            ),
            max_aqi=(
                "predicted_aqi",
                "max",
            ),
            min_aqi=(
                "predicted_aqi",
                "min",
            ),
        )
        .reset_index()
    )

    daily["avg_aqi"] = (
        daily["avg_aqi"]
        .round(1)
    )

    daily["category"] = (
        daily["avg_aqi"]
        .apply(
            lambda value:
            get_aqi_category(
                value
            )[0]
        )
    )

    daily["color"] = (
        daily["avg_aqi"]
        .apply(
            lambda value:
            get_aqi_category(
                value
            )[1]
        )
    )

    daily["emoji"] = (
        daily["avg_aqi"]
        .apply(
            lambda value:
            get_aqi_category(
                value
            )[2]
        )
    )

    return daily


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":

    predictions = predict_next_hours()

    if not predictions.empty:

        daily = get_daily_summary(
            predictions
        )

        print(
            "\n📅 3-Day AQI Forecast:"
        )

        print(
            daily.to_string(
                index=False
            )
        )
