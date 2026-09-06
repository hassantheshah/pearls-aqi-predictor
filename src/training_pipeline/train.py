"""
src/training_pipeline/train.py

Train AQI forecasting models.

The forecasting problem is explicitly:

    features at time t  ->  AQI at time t+1

Only information available at prediction time is used as a feature.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from config.settings import FEATURE_COLUMNS, TARGET_COLUMN


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = list(FEATURE_COLUMNS)


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────

def load_data(path: str = "data/processed/features.csv") -> pd.DataFrame:
    """Load and validate the processed feature dataset."""

    df = pd.read_csv(
        path,
        parse_dates=["fetched_at"],
    )

    df = (
        df.sort_values("fetched_at")
        .reset_index(drop=True)
    )

    missing = [
        column
        for column in FEATURE_COLS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in training data: {missing}"
        )

    # Create the actual forecasting target:
    #
    # features at time t -> AQI at time t+1
    df["target_aqi"] = df[TARGET_COLUMN].shift(-1)

    # The final row has no future AQI available.
    df = df.dropna(
        subset=FEATURE_COLS + ["target_aqi"]
    ).reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} rows for one-hour-ahead forecasting."
    )

    logger.info(
        f"Feature count: {len(FEATURE_COLS)}"
    )

    logger.info(
        f"Features: {FEATURE_COLS}"
    )

    return df


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(
    y_true,
    y_pred,
    name: str,
) -> dict:
    """Calculate regression metrics."""

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    logger.info(
        f"[{name}] "
        f"RMSE={rmse:.2f} "
        f"MAE={mae:.2f} "
        f"R²={r2:.4f}"
    )

    return {
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def evaluate_naive_baseline(
    df: pd.DataFrame,
    test_idx,
) -> dict:
    """
    Naive persistence baseline.

    Prediction:
        AQI(t+1) = AQI(t)

    This is the minimum benchmark our ML model should try
    to beat.
    """

    y_true = df.loc[test_idx, "target_aqi"].to_numpy()

    naive_pred = df.loc[
        test_idx,
        TARGET_COLUMN,
    ].to_numpy()

    return evaluate(
        y_true,
        naive_pred,
        "NaivePersistence",
    )


# ─────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────

def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    preds = model.predict(X_test)

    metrics = evaluate(
        y_test,
        preds,
        "RandomForest",
    )

    joblib.dump(
        model,
        f"{MODEL_DIR}/random_forest.pkl",
    )

    return {
        **metrics,
        "model_obj": model,
        "preds": preds,
    }


# ─────────────────────────────────────────────────────────────
# Ridge
# ─────────────────────────────────────────────────────────────

def train_ridge(
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = Ridge(
        alpha=1.0
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    preds = model.predict(
        X_test_scaled
    )

    metrics = evaluate(
        y_test,
        preds,
        "Ridge",
    )

    joblib.dump(
        model,
        f"{MODEL_DIR}/ridge.pkl",
    )

    joblib.dump(
        scaler,
        f"{MODEL_DIR}/scaler.pkl",
    )

    return {
        **metrics,
        "model_obj": model,
        "preds": preds,
    }


# ─────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────

def train_lstm(
    X_train,
    y_train,
    X_test,
    y_test,
    seq_len: int = 24,
) -> dict:
    """
    Train an LSTM using 24 hours of historical feature context.

    The target remains AQI at t+1.
    """

    try:
        import tensorflow as tf

        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential

    except Exception as exc:
        logger.warning(
            f"TensorFlow unavailable. Skipping LSTM: {exc}"
        )
        return {}

    # Scale features for neural-network training.
    x_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(
        X_train
    )

    X_test_scaled = x_scaler.transform(
        X_test
    )

    # Scale target.
    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(
        np.asarray(y_train).reshape(-1, 1)
    ).ravel()

    def make_sequences(
        X,
        y,
        sequence_length,
    ):
        X_sequences = []
        y_targets = []

        for i in range(
            sequence_length,
            len(X),
        ):
            X_sequences.append(
                X[
                    i - sequence_length:i
                ]
            )

            y_targets.append(
                y[i]
            )

        return (
            np.asarray(X_sequences),
            np.asarray(y_targets),
        )

    X_train_seq, y_train_seq = make_sequences(
        X_train_scaled,
        y_train_scaled,
        seq_len,
    )

    X_test_seq, y_test_seq = make_sequences(
        X_test_scaled,
        np.asarray(y_test),
        seq_len,
    )

    if len(X_train_seq) < 20:
        logger.warning(
            "Not enough data for LSTM training."
        )
        return {}

    model = Sequential(
        [
            LSTM(
                64,
                input_shape=(
                    seq_len,
                    X_train.shape[1],
                ),
                return_sequences=True,
            ),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    early_stopping = EarlyStopping(
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
    )

    scaled_predictions = (
        model.predict(
            X_test_seq,
            verbose=0,
        )
        .flatten()
    )

    predictions = y_scaler.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).ravel()

    # y_test_seq is already in the original AQI scale.
    metrics = evaluate(
        y_test_seq,
        predictions,
        "LSTM",
    )

    model.save(
        f"{MODEL_DIR}/lstm_model.keras"
    )

    joblib.dump(
        x_scaler,
        f"{MODEL_DIR}/lstm_x_scaler.pkl",
    )

    joblib.dump(
        y_scaler,
        f"{MODEL_DIR}/lstm_y_scaler.pkl",
    )

    return {
        **metrics,
        "model_obj": model,
        "preds": predictions,
    }


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_training_pipeline(
    data_path: str = "data/processed/features.csv",
) -> str:

    logger.info(
        "Starting one-hour-ahead AQI forecasting training."
    )

    df = load_data(
        data_path
    )

    X = df[FEATURE_COLS]
    y = df["target_aqi"]

    # Chronological split.
    # No random shuffling.
    tscv = TimeSeriesSplit(
        n_splits=5
    )

    splits = list(
        tscv.split(X)
    )

    train_idx, test_idx = splits[-1]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    logger.info(
        f"Train size: {len(X_train)}, "
        f"Test size: {len(X_test)}"
    )

    # Baseline
    baseline_result = evaluate_naive_baseline(
        df,
        test_idx,
    )

    results = [
        baseline_result
    ]

    # Machine-learning models
    rf_result = train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    results.append(
        rf_result
    )

    ridge_result = train_ridge(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    results.append(
        ridge_result
    )

    lstm_result = train_lstm(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    if lstm_result:
        results.append(
            lstm_result
        )

    # Save metrics.
    metrics_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in [
                    "model_obj",
                    "preds",
                ]
            }
            for result in results
        ]
    )

    metrics_df.to_csv(
        f"{MODEL_DIR}/metrics.csv",
        index=False,
    )

    logger.info(
        "\n"
        + metrics_df.to_string(
            index=False
        )
    )

    # Only ML models compete for deployment.
    ml_results = [
        result
        for result in results
        if result["model"]
        != "NaivePersistence"
    ]

    best = min(
        ml_results,
        key=lambda result: result["rmse"],
    )

    best_name = best["model"]

    baseline_rmse = baseline_result[
        "rmse"
    ]

    if best["rmse"] < baseline_rmse:
        logger.success(
            f"Best model beats naive baseline: "
            f"{best_name} RMSE={best['rmse']:.2f} "
            f"vs baseline RMSE={baseline_rmse:.2f}"
        )
    else:
        logger.warning(
            f"Best ML model does NOT beat naive baseline: "
            f"{best_name} RMSE={best['rmse']:.2f} "
            f"vs baseline RMSE={baseline_rmse:.2f}"
        )

    logger.success(
        f"Best model: {best_name} "
        f"(RMSE={best['rmse']:.2f})"
    )

    # Save best model reference.
    with open(
        f"{MODEL_DIR}/best_model.txt",
        "w",
    ) as file:
        file.write(
            best_name
        )

    # Save feature list.
    joblib.dump(
        FEATURE_COLS,
        f"{MODEL_DIR}/feature_cols.pkl",
    )

    return best_name


if __name__ == "__main__":
    best_model = run_training_pipeline()

    print(
        f"\nTraining complete. "
        f"Best model: {best_model}"
    )
