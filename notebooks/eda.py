"""
notebooks/eda.py
Exploratory Data Analysis for AQI data.
Run with: python notebooks/eda.py
Generates plots in data/processed/plots/
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PLOT_DIR = "data/processed/plots"
DATA_PATH = "data/processed/features.csv"
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "text.color":       "#e2e8f0",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#1e293b",
})


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}. Generating synthetic data for EDA demo...")
        from src.feature_pipeline.pipeline import _generate_backfill
        from src.feature_pipeline.feature_engineering import engineer_features
        df_raw = _generate_backfill(days=60)
        return engineer_features(df_raw)
    df = pd.read_csv(DATA_PATH, parse_dates=["fetched_at"])
    return df.sort_values("fetched_at").reset_index(drop=True)


def plot_aqi_timeseries(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(df["fetched_at"], df["aqi"], color="#38bdf8", lw=1.5, alpha=0.9)
    axes[0].fill_between(df["fetched_at"], df["aqi"], alpha=0.15, color="#38bdf8")
    axes[0].axhline(150, color="#ef4444", linestyle="--", lw=1, label="Alert Threshold (150)")
    axes[0].set_ylabel("AQI")
    axes[0].set_title("AQI Time Series", color="#f8fafc", fontsize=14)
    axes[0].legend()

    if "pm25" in df.columns:
        axes[1].plot(df["fetched_at"], df["pm25"], color="#f472b6", lw=1.2, label="PM2.5")
    if "pm10" in df.columns:
        axes[1].plot(df["fetched_at"], df["pm10"], color="#fb923c", lw=1.2, label="PM10")
    axes[1].set_ylabel("Concentration (μg/m³)")
    axes[1].set_title("Pollutant Trends", color="#f8fafc", fontsize=14)
    axes[1].legend()

    plt.tight_layout()
    path = f"{PLOT_DIR}/aqi_timeseries.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_hourly_pattern(df: pd.DataFrame):
    if "hour" not in df.columns:
        return
    hourly = df.groupby("hour")["aqi"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(hourly["hour"],
                    hourly["mean"] - hourly["std"],
                    hourly["mean"] + hourly["std"],
                    alpha=0.2, color="#818cf8")
    ax.plot(hourly["hour"], hourly["mean"], color="#818cf8", lw=2.5, marker="o", ms=6)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean AQI")
    ax.set_title("Average AQI by Hour of Day", color="#f8fafc", fontsize=14)
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    path = f"{PLOT_DIR}/hourly_pattern.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = ["aqi", "pm25", "pm10", "no2", "o3", "co",
                "temperature", "humidity", "wind_speed", "pressure"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", color="#f8fafc", fontsize=14)
    plt.tight_layout()
    path = f"{PLOT_DIR}/correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_aqi_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["aqi"], bins=40, color="#6366f1", edgecolor="#1e293b", alpha=0.9)
    axes[0].set_xlabel("AQI")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("AQI Distribution", color="#f8fafc")

    axes[1].boxplot(df["aqi"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#6366f1", color="#818cf8"),
                    medianprops=dict(color="#f8fafc", linewidth=2),
                    whiskerprops=dict(color="#94a3b8"),
                    capprops=dict(color="#94a3b8"),
                    flierprops=dict(markerfacecolor="#f472b6", marker="o", markersize=4))
    axes[1].set_ylabel("AQI")
    axes[1].set_title("AQI Box Plot", color="#f8fafc")

    plt.tight_layout()
    path = f"{PLOT_DIR}/aqi_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_weekday_pattern(df: pd.DataFrame):
    if "day_of_week" not in df.columns:
        return
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = df.groupby("day_of_week")["aqi"].mean().reset_index()
    weekly["day_name"] = weekly["day_of_week"].apply(lambda x: days[x])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#6366f1" if i < 5 else "#f472b6" for i in range(7)]
    ax.bar(weekly["day_name"], weekly["aqi"], color=colors, edgecolor="#1e293b")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean AQI")
    ax.set_title("AQI by Day of Week", color="#f8fafc", fontsize=14)
    plt.tight_layout()
    path = f"{PLOT_DIR}/weekday_pattern.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_summary(df: pd.DataFrame):
    print("\n" + "═" * 50)
    print("  EDA SUMMARY — PEARLS AQI PREDICTOR")
    print("═" * 50)
    print(f"  Total rows:         {len(df):,}")
    print(f"  Date range:         {df['fetched_at'].min()} → {df['fetched_at'].max()}")
    print(f"  AQI mean:           {df['aqi'].mean():.1f}")
    print(f"  AQI std:            {df['aqi'].std():.1f}")
    print(f"  AQI min/max:        {df['aqi'].min()} / {df['aqi'].max()}")
    hazardous = (df["aqi"] >= 150).sum()
    print(f"  Hazardous readings: {hazardous} ({100*hazardous/len(df):.1f}%)")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print_summary(df)
    print("Generating EDA plots...")
    plot_aqi_timeseries(df)
    plot_hourly_pattern(df)
    plot_correlation_heatmap(df)
    plot_aqi_distribution(df)
    plot_weekday_pattern(df)
    print(f"\n✅ All EDA plots saved to {PLOT_DIR}/")
