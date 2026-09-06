"""
src/training_pipeline/train.py
Trains Random Forest, Ridge Regression, and LSTM models on AQI features.
Evaluates with RMSE, MAE, R² and registers best model.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.settings import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [c for c in FEATURE_COLUMNS if c != TARGET_COLUMN]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path: str = "data/processed/features.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["fetched_at"])
    df = df.sort_values("fetched_at").reset_index(drop=True)
    missing = [c for c in FEATURE_COLS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COLUMN])
    logger.info(f"Loaded {len(df)} rows for training.")
    return df


def evaluate(y_true, y_pred, name: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    logger.info(f"[{name}] RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ── Models ────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_split=4, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds, "RandomForest")
    joblib.dump(model, f"{MODEL_DIR}/random_forest.pkl")
    return {**metrics, "model_obj": model, "preds": preds}


def train_ridge(X_train, y_train, X_test, y_test, scaler) -> dict:
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(X_tr_sc, y_train)
    preds = model.predict(X_te_sc)
    metrics = evaluate(y_test, preds, "Ridge")
    joblib.dump(model,  f"{MODEL_DIR}/ridge.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    return {**metrics, "model_obj": model, "preds": preds}


def train_lstm(X_train, y_train, X_test, y_test, seq_len: int = 24) -> dict:
    """Train a simple LSTM for sequential AQI forecasting."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        def make_sequences(X, y, seq):
            Xs, ys = [], []
            for i in range(len(X) - seq):
                Xs.append(X[i:i+seq])
                ys.append(y[i+seq])
            return np.array(Xs), np.array(ys)

        X_tr_seq, y_tr_seq = make_sequences(X_train.values, y_train.values, seq_len)
        X_te_seq, y_te_seq = make_sequences(X_test.values,  y_test.values,  seq_len)

        if len(X_tr_seq) < 10:
            logger.warning("Not enough data for LSTM training — skipping.")
            return {}

        model = Sequential([
            LSTM(64, input_shape=(seq_len, X_train.shape[1]),
                 return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(
            X_tr_seq, y_tr_seq,
            validation_split=0.1,
            epochs=50, batch_size=32,
            callbacks=[es], verbose=0
        )
        preds = model.predict(X_te_seq).flatten()
        metrics = evaluate(y_te_seq, preds, "LSTM")
        model.save(f"{MODEL_DIR}/lstm_model.keras")
        return {**metrics, "model_obj": model, "preds": preds}

    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        return {}


# ── Main Training Pipeline ────────────────────────────────────────────────────

def run_training_pipeline(data_path: str = "data/processed/features.csv") -> str:
    """Full training pipeline. Returns name of the best model."""
    df = load_data(data_path)

    X = df[FEATURE_COLS]
    y = df[TARGET_COLUMN]

    # Time-series split (no data leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    scaler = StandardScaler()
    results = []

    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(rf_result)

    ridge_result = train_ridge(X_train, y_train, X_test, y_test, scaler)
    results.append(ridge_result)

    lstm_result = train_lstm(X_train, y_train, X_test, y_test)
    if lstm_result:
        results.append(lstm_result)

    # Save metrics
    metrics_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ["model_obj", "preds"]}
         for r in results]
    )
    metrics_df.to_csv(f"{MODEL_DIR}/metrics.csv", index=False)
    logger.info(f"\n{metrics_df.to_string(index=False)}")

    # Select best model by RMSE
    best = min(results, key=lambda r: r["rmse"])
    best_name = best["model"]
    logger.success(f"Best model: {best_name} (RMSE={best['rmse']:.2f})")

    # Save best model reference
    with open(f"{MODEL_DIR}/best_model.txt", "w") as f:
        f.write(best_name)

    # Save feature list
    joblib.dump(FEATURE_COLS, f"{MODEL_DIR}/feature_cols.pkl")

    return best_name


if __name__ == "__main__":
    best = run_training_pipeline()
    print(f"\n✅ Training complete. Best model: {best}")
