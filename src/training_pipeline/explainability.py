"""
src/training_pipeline/explainability.py
SHAP-based feature importance explanations for the trained AQI model.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.settings import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_DIR  = "models"
PLOT_DIR   = "data/processed/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURE_COLS = [c for c in FEATURE_COLUMNS if c != TARGET_COLUMN]


def load_model_and_data():
    """Load the best model and recent feature data."""
    best_name = open(f"{MODEL_DIR}/best_model.txt").read().strip()
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")

    if best_name == "RandomForest":
        model = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    elif best_name == "Ridge":
        model = joblib.load(f"{MODEL_DIR}/ridge.pkl")
    else:
        logger.warning("SHAP only supported for tree/linear models in this module.")
        return None, None, None, best_name

    df = pd.read_csv("data/processed/features.csv")
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols].tail(500)   # use recent 500 rows for SHAP
    return model, X, feature_cols, best_name


def compute_shap_values(model, X: pd.DataFrame, model_name: str):
    """Compute SHAP values for the given model and feature data."""
    logger.info(f"Computing SHAP values for {model_name}...")
    if model_name == "RandomForest":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
    return shap_values, explainer


def plot_shap_summary(shap_values, X: pd.DataFrame,
                      save_path: str = None) -> str:
    """Generate and save a SHAP summary beeswarm plot."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Importance — AQI Predictor", fontsize=14)
    plt.tight_layout()
    path = save_path or f"{PLOT_DIR}/shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary saved to {path}")
    return path


def plot_shap_bar(shap_values, X: pd.DataFrame,
                  save_path: str = None) -> str:
    """Generate mean absolute SHAP bar chart."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X.columns).sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importance.tail(15).plot(kind="barh", color="#00b4d8")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top Feature Importances (SHAP)", fontsize=13)
    plt.tight_layout()
    path = save_path or f"{PLOT_DIR}/shap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP bar chart saved to {path}")
    return path


def get_top_features(shap_values, X: pd.DataFrame,
                     top_n: int = 10) -> pd.DataFrame:
    """Return a DataFrame of top N features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature":    X.columns,
        "importance": mean_abs
    }).sort_values("importance", ascending=False).head(top_n)
    return importance


def run_explainability() -> dict:
    """Full explainability pipeline. Returns paths to generated plots."""
    model, X, feature_cols, model_name = load_model_and_data()
    if model is None:
        return {}

    shap_values, explainer = compute_shap_values(model, X, model_name)
    summary_path = plot_shap_summary(shap_values, X)
    bar_path     = plot_shap_bar(shap_values, X)
    top_features = get_top_features(shap_values, X)

    logger.success("Explainability pipeline complete.")
    return {
        "shap_summary_plot": summary_path,
        "shap_bar_plot":     bar_path,
        "top_features":      top_features,
    }


if __name__ == "__main__":
    result = run_explainability()
    print("\nTop Features:")
    print(result.get("top_features"))
