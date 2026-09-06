"""
src/feature_pipeline/pipeline.py

Main feature pipeline:
- real-time mode fetches current AQI + weather
- backfill mode fetches real historical AQI + weather from Open-Meteo
- demo-backfill mode generates synthetic data for offline testing

Run hourly via GitHub Actions.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.feature_pipeline.fetch_data import (
    fetch_combined_current,
    fetch_openmeteo_historical_data,
)
from src.feature_pipeline.feature_engineering import engineer_features
from src.feature_pipeline.feature_store import (
    insert_features,
    save_local,
    _read_local_fallback,
)


LOCAL_STORE_PATH = "data/processed/features.csv"


def run_feature_pipeline(
    backfill: bool = False,
    backfill_days: int = 90,
    demo_backfill: bool = False,
) -> pd.DataFrame:
    """
    Main feature pipeline entry point.

    Args:
        backfill:
            Fetch real historical AQI and weather data from Open-Meteo.

        backfill_days:
            Number of historical days to fetch.

        demo_backfill:
            Generate synthetic data for offline/demo testing only.

    Returns:
        Engineered DataFrame that was stored.
    """

    if backfill and demo_backfill:
        raise ValueError(
            "Use either --backfill or --demo-backfill, not both."
        )

    if backfill:
        logger.info(
            f"Running REAL HISTORICAL BACKFILL for {backfill_days} days..."
        )

        end_date = datetime.utcnow().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=backfill_days - 1)

        df_raw = fetch_openmeteo_historical_data(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        if df_raw.empty:
            logger.error(
                "Historical backfill returned no data."
            )
            return pd.DataFrame()

    elif demo_backfill:
        logger.warning(
            "Running DEMO BACKFILL with synthetic data. "
            "This mode must not be used as production training data."
        )

        df_raw = _generate_demo_backfill(backfill_days)

    else:
        logger.info(
            "Running REAL-TIME feature pipeline..."
        )

        record = fetch_combined_current()

        if not record:
            logger.error(
                "Empty record from API — pipeline aborted."
            )
            return pd.DataFrame()

        existing = _read_local_fallback(LOCAL_STORE_PATH)

        if existing.empty:
            df_raw = pd.DataFrame([record])
        else:
            df_raw = pd.concat(
                [
                    existing,
                    pd.DataFrame([record]),
                ],
                ignore_index=True,
            )

    df_features = engineer_features(df_raw)

    if df_features.empty:
        logger.warning(
            "Feature engineering returned empty DataFrame."
        )
        return df_features

    # Historical backfill replaces the bundled/demo dataset.
    # Real-time mode continues appending to the existing store.
    if backfill:
        save_local(
            df_features,
            LOCAL_STORE_PATH,
        )
    elif demo_backfill:
        save_local(
            df_features,
            LOCAL_STORE_PATH,
        )
    else:
        save_local(
            df_features,
            LOCAL_STORE_PATH,
        )

    try:
        if backfill or demo_backfill:
            insert_features(df_features)
        else:
            insert_features(df_features.tail(1))
    except Exception as exc:
        logger.warning(
            f"Hopsworks insert skipped (will use local): {exc}"
        )

    logger.success(
        f"Feature pipeline complete. "
        f"{len(df_features)} rows stored."
    )

    return df_features


def _generate_demo_backfill(days: int) -> pd.DataFrame:
    """
    Generate synthetic historical data for offline/demo testing only.

    This function is intentionally NOT used by normal --backfill.
    """

    import numpy as np

    from config.settings import CITY_NAME

    hours = days * 24

    dates = pd.date_range(
        end=datetime.utcnow(),
        periods=hours,
        freq="h",
    )

    np.random.seed(42)

    hour_arr = np.array(dates.hour)

    daily_pattern = (
        30
        * np.sin(
            (hour_arr - 6)
            * np.pi
            / 12
        )
    )

    base_aqi = (
        80
        + daily_pattern
        + np.random.normal(
            0,
            15,
            hours,
        )
    )

    base_aqi = np.clip(
        base_aqi,
        20,
        300,
    )

    df = pd.DataFrame(
        {
            "city": CITY_NAME,
            "fetched_at": dates,
            "aqi": base_aqi.astype(int),
            "pm25": np.clip(
                base_aqi * 0.4
                + np.random.normal(
                    0,
                    5,
                    hours,
                ),
                0,
                None,
            ),
            "pm10": np.clip(
                base_aqi * 0.6
                + np.random.normal(
                    0,
                    8,
                    hours,
                ),
                0,
                None,
            ),
            "no2": np.random.uniform(
                5,
                60,
                hours,
            ),
            "o3": np.random.uniform(
                10,
                80,
                hours,
            ),
            "co": np.random.uniform(
                0.3,
                2.5,
                hours,
            ),
            "temperature": (
                25
                + 8
                * np.sin(
                    (hour_arr - 14)
                    * np.pi
                    / 12
                )
                + np.random.normal(
                    0,
                    2,
                    hours,
                )
            ),
            "humidity": np.random.uniform(
                40,
                90,
                hours,
            ),
            "wind_speed": np.abs(
                np.random.normal(
                    3,
                    2,
                    hours,
                )
            ),
            "pressure": np.random.uniform(
                1008,
                1018,
                hours,
            ),
        }
    )

    logger.info(
        f"Generated {len(df)} synthetic demo rows."
    )

    return df


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Pearls AQI feature pipeline"
    )

    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Fetch real historical data from Open-Meteo.",
    )

    parser.add_argument(
        "--demo-backfill",
        action="store_true",
        help="Generate synthetic data for offline/demo testing only.",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of historical days to fetch or generate.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.days < 1:
        raise ValueError(
            "--days must be at least 1."
        )

    run_feature_pipeline(
        backfill=args.backfill,
        backfill_days=args.days,
        demo_backfill=args.demo_backfill,
    )
