# 🌫️ Pearls AQI Predictor

> **Predict Air Quality Index (AQI) for the next 3 days using an automated cloud-based ML pipeline.**

[![Feature Pipeline](https://github.com/hassantheshah/pearls-aqi-predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/hassantheshah/pearls-aqi-predictor/actions)
[![Training Pipeline](https://github.com/hassantheshah/pearls-aqi-predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/hassantheshah/pearls-aqi-predictor/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)

---

## 📐 Architecture

```
Weather & Pollution APIs
        │
        ▼
┌───────────────────┐      GitHub Actions (every 1h)
│  Feature Pipeline  │ ──────────────────────────────────┐
│  fetch + engineer  │                                   │
└────────┬──────────┘                                   ▼
         │ features                          ┌──────────────────┐
         ▼                                   │  Feature Store   │
┌──────────────────┐     GitHub Actions       │  (Hopsworks /    │
│ Training Pipeline │◄── (every 24h) ─────── │   local CSV)     │
│  RF + Ridge + LSTM│                         └──────────────────┘
└────────┬─────────┘                                   ▲
         │ model                                       │
         ▼                                             │
┌──────────────────┐    ┌──────────────┐              │
│  Model Registry   │───►│  Flask API   │◄─────────────┘
└──────────────────┘    │  /forecast   │
                        │  /current    │
                        │  /alerts     │
                        └──────┬───────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Streamlit Dashboard  │
                    │  - AQI Gauge          │
                    │  - 3-Day Forecast     │
                    │  - EDA & Trends       │
                    │  - SHAP Explanations  │
                    │  - Hazard Alerts      │
                    └──────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/pearls-aqi-predictor.git
cd pearls-aqi-predictor
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Required API keys:**
| Key | Where to get |
|-----|-------------|
| `AQICN_API_KEY` | https://aqicn.org/data-platform/token/ (free) |
| `OPENWEATHER_API_KEY` | https://openweathermap.org/api (free tier) |
| `HOPSWORKS_API_KEY` | https://app.hopsworks.ai (free tier) |

### 3. Backfill historical data (first time only)

```bash
python -m src.feature_pipeline.pipeline --backfill --days 90
```

### 4. Train models

```bash
python -m src.training_pipeline.train
```

### 5. Generate SHAP explanations

```bash
python -m src.training_pipeline.explainability
```

### 6. Run EDA

```bash
python notebooks/eda.py
```

### 7. Start Flask API

```bash
python -m src.api.app
# API running at http://localhost:5000
```

### 8. Launch Streamlit dashboard

```bash
streamlit run src/dashboard/app.py
# Dashboard at http://localhost:8501
```

---

## 📁 Project Structure

```
pearls-aqi-predictor/
├── config/
│   └── settings.py              # Central config (API keys, city, features)
├── src/
│   ├── feature_pipeline/
│   │   ├── fetch_data.py        # AQICN + OpenWeather API calls
│   │   ├── feature_engineering.py  # Time, lag, rolling, change rate features
│   │   ├── feature_store.py     # Hopsworks read/write
│   │   └── pipeline.py          # Main feature pipeline runner
│   ├── training_pipeline/
│   │   ├── train.py             # RandomForest + Ridge + LSTM training
│   │   └── explainability.py    # SHAP feature importance
│   ├── inference_pipeline/
│   │   └── predict.py           # 72-hour AQI forecasting
│   ├── api/
│   │   └── app.py               # Flask REST API
│   └── dashboard/
│       └── app.py               # Streamlit dashboard
├── notebooks/
│   └── eda.py                   # Exploratory Data Analysis
├── tests/
│   └── test_feature_engineering.py  # Unit tests
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml # Runs every 1 hour
│       ├── training_pipeline.yml # Runs every 24 hours
│       └── backfill.yml         # Manual backfill trigger
├── data/
│   ├── raw/                     # Raw API responses (auto-created)
│   └── processed/               # Feature CSV + SHAP plots
├── models/                      # Trained model files
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/current` | GET | Latest AQI reading |
| `/api/forecast` | GET | 72-hour hourly forecast |
| `/api/forecast/daily` | GET | 3-day daily summary |
| `/api/alerts` | GET | Hazardous AQI alerts |
| `/api/history?hours=48` | GET | Historical readings |
| `/api/metrics` | GET | Model evaluation metrics |

---

## 🤖 ML Models

| Model | Library | Notes |
|-------|---------|-------|
| **Random Forest** | scikit-learn | 200 trees, best for tabular AQI data |
| **Ridge Regression** | scikit-learn | Fast linear baseline |
| **LSTM** | TensorFlow | Sequence model for temporal patterns |

**Evaluation metrics:** RMSE, MAE, R²
**Selection:** Best model by RMSE on time-series test split

---

## ⚙️ GitHub Actions CI/CD

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| `feature_pipeline.yml` | Every hour | Fetches API data, engineers features, stores in Hopsworks |
| `training_pipeline.yml` | Daily at 02:00 UTC | Trains all models, generates SHAP plots |
| `backfill.yml` | Manual trigger | Generates N days of historical training data |

**GitHub Secrets required:**
```
AQICN_API_KEY
OPENWEATHER_API_KEY
HOPSWORKS_API_KEY
HOPSWORKS_PROJECT
```

**GitHub Variables:**
```
CITY_NAME=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011
AQICN_CITY_SLUG=karachi
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 🎨 Dashboard Pages

1. **🏠 Dashboard** — Live AQI gauge, pollutant metrics, 3-day forecast cards
2. **📈 Forecast** — Interactive 72-hour AQI line chart with AQI band overlays
3. **📊 EDA & History** — Historical trends, distribution analysis, scatter plots
4. **🧠 Explainability** — SHAP beeswarm and bar charts
5. **⚙️ Model Metrics** — RMSE/MAE/R² comparison across models

---

## 📊 AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0–50 | Good 😊 | Air quality is satisfactory |
| 51–100 | Moderate 😐 | Acceptable for most people |
| 101–150 | Unhealthy for Sensitive 😷 | Sensitive groups affected |
| 151–200 | Unhealthy 🚨 | Everyone may experience effects |
| 201–300 | Very Unhealthy ⚠️ | Health alert |
| 301–500 | Hazardous ☠️ | Emergency conditions |

---

## 📋 Submission Checklist

- [x] End-to-end AQI prediction system
- [x] Feature pipeline (fetch → engineer → store)
- [x] Historical data backfill
- [x] Training pipeline (RF + Ridge + LSTM)
- [x] Model evaluation (RMSE, MAE, R²)
- [x] Automated CI/CD (GitHub Actions)
- [x] Flask REST API
- [x] Streamlit interactive dashboard
- [x] SHAP feature importance explanations
- [x] Hazardous AQI alerts
- [x] EDA with trend analysis
- [x] Unit tests
- [x] Documentation

---

*Built with ❤️ for Pearls Internship — AQI Predictor Project*
