"""
src/feature_pipeline/feature_store.py
Hopsworks Feature Store integration — read/write feature groups and retrieve training data.
"""
import pandas as pd
from loguru import logger
import sys
sys.path.append("../..")
from config.settings import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

# Feature Group config
FG_NAME    = "aqi_features"
FG_VERSION = 1
FV_NAME    = "aqi_feature_view"
FV_VERSION = 1


def get_feature_store():
    """Connect to Hopsworks and return the feature store handle."""
    try:
        import hopsworks
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT
        )
        fs = project.get_feature_store()
        logger.info(f"Connected to Hopsworks project: {HOPSWORKS_PROJECT}")
        return fs
    except Exception as e:
        logger.error(f"Hopsworks connection failed: {e}")
        raise


def get_or_create_feature_group(fs):
    """Get or create the AQI feature group."""
    try:
        fg = fs.get_or_create_feature_group(
            name=FG_NAME,
            version=FG_VERSION,
            primary_key=["city", "fetched_at"],
            description="Hourly AQI and weather features for Pearls AQI Predictor",
            online_enabled=True,
        )
        logger.info(f"Feature group '{FG_NAME}' ready.")
        return fg
    except Exception as e:
        logger.error(f"Failed to get/create feature group: {e}")
        raise


def insert_features(df: pd.DataFrame) -> None:
    """Insert engineered feature rows into the Hopsworks feature group."""
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"]).astype(str)
    fg.insert(df, write_options={"wait_for_job": False})
    logger.info(f"Inserted {len(df)} rows into feature store.")


def read_training_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Read historical features from Hopsworks for model training.
    Falls back to local CSV if Hopsworks unavailable.
    """
    try:
        fs = get_feature_store()
        fv = fs.get_feature_view(name=FV_NAME, version=FV_VERSION)
        df, _ = fv.training_data()
        logger.info(f"Read {len(df)} rows from feature view.")
        if start_date:
            df = df[df["fetched_at"] >= start_date]
        if end_date:
            df = df[df["fetched_at"] <= end_date]
        return df
    except Exception as e:
        logger.warning(f"Hopsworks read failed ({e}), trying local CSV fallback...")
        return _read_local_fallback()


def read_latest_features(n_rows: int = 72) -> pd.DataFrame:
    """Read the most recent N rows from the feature store for inference."""
    try:
        fs = get_feature_store()
        fg = get_or_create_feature_group(fs)
        df = fg.read()
        df = df.sort_values("fetched_at").tail(n_rows)
        logger.info(f"Read {len(df)} recent rows from feature store.")
        return df
    except Exception as e:
        logger.warning(f"Online store read failed: {e}, using local fallback.")
        return _read_local_fallback().tail(n_rows)


def _read_local_fallback(path: str = "data/processed/features.csv") -> pd.DataFrame:
    """Load processed features from local CSV (offline fallback)."""
    try:
        df = pd.read_csv(path, parse_dates=["fetched_at"])
        logger.info(f"Loaded {len(df)} rows from local CSV: {path}")
        return df
    except FileNotFoundError:
        logger.error(f"Local fallback CSV not found: {path}")
        return pd.DataFrame()


def save_local(df: pd.DataFrame, path: str = "data/processed/features.csv") -> None:
    """Save features locally as a CSV backup."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")
