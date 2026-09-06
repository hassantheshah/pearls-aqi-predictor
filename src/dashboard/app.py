"""
src/dashboard/app.py
Streamlit dashboard — Real-time AQI display, 3-day forecast,
historical trends, SHAP explainability, and hazard alerts.
"""
import os
import sys
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.settings import CITY_NAME, ALERT_THRESHOLD, get_aqi_category

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"Pearls AQI Predictor — {CITY_NAME}",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("FLASK_API_URL", "http://localhost:5000")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main { background: #0a0e1a; }
    .aqi-card {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1126 100%);
        border: 1px solid #2a3050;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .aqi-value { font-size: 4rem; font-weight: 700; line-height: 1; }
    .aqi-label { font-size: 1.1rem; opacity: 0.7; margin-top: 6px; }
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .alert-box {
        background: linear-gradient(135deg, #7f1d1d, #450a0a);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .stMetric { background: #111827; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_current():
    try:
        r = requests.get(f"{API_BASE}/api/current", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return _mock_current()


@st.cache_data(ttl=300)
def load_forecast():
    try:
        r = requests.get(f"{API_BASE}/api/forecast/daily", timeout=5)
        return r.json().get("daily", []) if r.ok else []
    except Exception:
        return _mock_daily_forecast()


@st.cache_data(ttl=300)
def load_hourly_forecast():
    try:
        r = requests.get(f"{API_BASE}/api/forecast", timeout=5)
        data = r.json().get("forecast", []) if r.ok else []
        return pd.DataFrame(data)
    except Exception:
        return _mock_hourly_forecast()


@st.cache_data(ttl=300)
def load_history():
    try:
        r = requests.get(f"{API_BASE}/api/history?hours=72", timeout=5)
        data = r.json().get("history", []) if r.ok else []
        return pd.DataFrame(data)
    except Exception:
        return _mock_history()


@st.cache_data(ttl=300)
def load_alerts():
    try:
        r = requests.get(f"{API_BASE}/api/alerts", timeout=5)
        return r.json().get("alerts", []) if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=3600)
def load_metrics():
    try:
        r = requests.get(f"{API_BASE}/api/metrics", timeout=5)
        return r.json().get("metrics", []) if r.ok else []
    except Exception:
        return []


# ── Mock data helpers (when API is offline) ───────────────────────────────────
def _mock_current():
    return {
        "city": CITY_NAME, "aqi": 87, "category": "Moderate",
        "color": "#ffff00", "emoji": "😐",
        "pm25": 32.4, "pm10": 58.1, "no2": 24.3, "o3": 41.2,
        "temperature": 32.5, "humidity": 68, "wind_speed": 4.2,
        "timestamp": datetime.utcnow().isoformat()
    }


def _mock_daily_forecast():
    dates = [(datetime.utcnow() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)]
    return [
        {"date": dates[0], "avg_aqi": 92, "max_aqi": 118, "min_aqi": 71, "category": "Moderate", "color": "#ffff00", "emoji": "😐"},
        {"date": dates[1], "avg_aqi": 134, "max_aqi": 162, "min_aqi": 98, "category": "Unhealthy for Sensitive", "color": "#ff7e00", "emoji": "😷"},
        {"date": dates[2], "avg_aqi": 78, "max_aqi": 95,  "min_aqi": 55, "category": "Moderate", "color": "#ffff00", "emoji": "😐"},
    ]


def _mock_hourly_forecast():
    hours = 72
    times = [datetime.utcnow() + timedelta(hours=i) for i in range(hours)]
    aqi = 80 + 30 * np.sin(np.linspace(0, 4 * np.pi, hours)) + np.random.normal(0, 8, hours)
    aqi = np.clip(aqi, 20, 250).astype(int)
    cats = [get_aqi_category(v) for v in aqi]
    return pd.DataFrame({
        "timestamp": [t.isoformat() for t in times],
        "predicted_aqi": aqi,
        "category": [c[0] for c in cats],
        "color":    [c[1] for c in cats],
        "emoji":    [c[2] for c in cats],
    })


def _mock_history():
    hours = 72
    times = [datetime.utcnow() - timedelta(hours=i) for i in range(hours, 0, -1)]
    aqi = 85 + 25 * np.sin(np.linspace(0, 6 * np.pi, hours)) + np.random.normal(0, 10, hours)
    return pd.DataFrame({
        "fetched_at": [t.isoformat() for t in times],
        "aqi": np.clip(aqi, 20, 300).astype(int),
        "temperature": np.random.uniform(28, 38, hours),
        "humidity": np.random.uniform(45, 85, hours),
        "pm25": np.random.uniform(10, 80, hours),
    })


# ── AQI Gauge Chart ───────────────────────────────────────────────────────────
def aqi_gauge(value: float, title: str = "Current AQI") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 18, "color": "#e2e8f0"}},
        number={"font": {"size": 48, "color": "#f8fafc"}},
        gauge={
            "axis": {"range": [0, 300], "tickcolor": "#64748b",
                     "tickfont": {"color": "#94a3b8"}},
            "bar":  {"color": get_aqi_category(value)[1], "thickness": 0.25},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0,   50],  "color": "#052e16"},
                {"range": [51,  100], "color": "#1c1917"},
                {"range": [101, 150], "color": "#1c0a00"},
                {"range": [151, 200], "color": "#1c0000"},
                {"range": [201, 300], "color": "#1a0025"},
            ],
            "threshold": {
                "line": {"color": "#f43f5e", "width": 3},
                "thickness": 0.8,
                "value": ALERT_THRESHOLD,
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(t=40, b=10, l=20, r=20),
        font={"color": "#e2e8f0"},
    )
    return fig


# ── Forecast Line Chart ────────────────────────────────────────────────────────
def forecast_line(df: pd.DataFrame) -> go.Figure:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig = go.Figure()

    # Shaded AQI bands
    for lo, hi, color, label in [
        (0, 50,   "rgba(0,228,0,0.06)",    "Good"),
        (51, 100, "rgba(255,255,0,0.06)",  "Moderate"),
        (101,150, "rgba(255,126,0,0.06)",  "Unhealthy*"),
        (151,200, "rgba(255,0,0,0.06)",    "Unhealthy"),
        (201,300, "rgba(143,63,151,0.06)", "Very Unhealthy"),
    ]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, line_width=0, annotation_text=label,
                      annotation_position="right", annotation_font_size=10,
                      annotation_font_color="#64748b")

    # Alert threshold line
    fig.add_hline(y=ALERT_THRESHOLD, line_dash="dot", line_color="#ef4444",
                  annotation_text=f"Alert Threshold ({ALERT_THRESHOLD})",
                  annotation_font_color="#ef4444")

    # AQI line + fill
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["predicted_aqi"],
        mode="lines", name="Predicted AQI",
        line=dict(color="#38bdf8", width=2.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#64748b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#64748b"),
        hovermode="x unified", height=350,
        margin=dict(t=20, b=20, l=20, r=100),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#e2e8f0"),
    )
    return fig


# ── Historical Trend Chart ─────────────────────────────────────────────────────
def history_chart(df: pd.DataFrame) -> go.Figure:
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["fetched_at"], y=df["aqi"],
        mode="lines+markers",
        marker=dict(size=4, color="#818cf8"),
        line=dict(color="#6366f1", width=2),
        name="Historical AQI",
        fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
    ))
    if "pm25" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["fetched_at"], y=df["pm25"],
            mode="lines", name="PM2.5",
            line=dict(color="#f472b6", width=1.5, dash="dot"),
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#64748b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#64748b"),
        hovermode="x unified", height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#e2e8f0"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Sidebar
with st.sidebar:
    st.markdown("## 🌫️ Pearls AQI")
    st.markdown(f"**City:** {CITY_NAME}")
    st.markdown(f"**Last updated:** {datetime.utcnow().strftime('%H:%M UTC')}")
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Dashboard", "📈 Forecast", "📊 EDA & History",
        "🧠 Explainability", "⚙️ Model Metrics"
    ])
    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Load all data
current   = load_current()
daily_fc  = load_forecast()
hourly_fc = load_hourly_forecast()
history   = load_history()
alerts    = load_alerts()
metrics   = load_metrics()

# ── ALERTS BANNER ──────────────────────────────────────────────────────────────
if alerts:
    for alert in alerts:
        st.markdown(f"""
        <div class="alert-box">
            ⚠️ <strong>HAZARDOUS AIR QUALITY ALERT</strong> — {alert['message']}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title(f"🌫️ {CITY_NAME} Air Quality Dashboard")

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        aqi_val  = current.get("aqi", 0)
        cat, col, emj = get_aqi_category(aqi_val)
        st.plotly_chart(aqi_gauge(aqi_val), use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center; margin-top:-20px;">
            <span style="font-size:2.5rem;">{emj}</span>
            <p style="color:{col}; font-size:1.3rem; font-weight:600; margin:4px 0;">{cat}</p>
            <p style="color:#64748b; font-size:0.85rem;">{current.get('timestamp','')[:16]} UTC</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 💨 Pollutants")
        st.metric("PM2.5",      f"{current.get('pm25', 0):.1f} μg/m³")
        st.metric("PM10",       f"{current.get('pm10', 0):.1f} μg/m³")
        st.metric("NO₂",        f"{current.get('no2',  0):.1f} μg/m³")
        st.metric("O₃ (Ozone)", f"{current.get('o3',   0):.1f} μg/m³")

    with col3:
        st.markdown("#### 🌡️ Weather")
        st.metric("Temperature", f"{current.get('temperature', 0):.1f} °C")
        st.metric("Humidity",    f"{current.get('humidity', 0):.0f}%")
        st.metric("Wind Speed",  f"{current.get('wind_speed', 0):.1f} m/s")

    st.divider()

    # 3-Day forecast cards
    st.markdown("### 📅 3-Day AQI Forecast")
    if daily_fc:
        cols = st.columns(3)
        for i, day in enumerate(daily_fc[:3]):
            with cols[i]:
                color = day.get("color", "#888")
                st.markdown(f"""
                <div class="aqi-card">
                    <p style="color:#94a3b8; font-size:0.85rem;">{day['date']}</p>
                    <p class="aqi-value" style="color:{color};">{int(day['avg_aqi'])}</p>
                    <p class="aqi-label">{day['emoji']} {day['category']}</p>
                    <p style="color:#64748b; font-size:0.8rem; margin-top:10px;">
                        ↑ {int(day['max_aqi'])} &nbsp;|&nbsp; ↓ {int(day['min_aqi'])}
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Forecast":
    st.title("📈 72-Hour AQI Forecast")
    if not hourly_fc.empty:
        st.plotly_chart(forecast_line(hourly_fc), use_container_width=True)
        st.markdown("#### Hourly Breakdown")
        display_cols = ["timestamp", "predicted_aqi", "category", "emoji"]
        display_cols = [c for c in display_cols if c in hourly_fc.columns]
        st.dataframe(
            hourly_fc[display_cols].head(48),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No forecast data available. Make sure the Flask API is running.")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & History":
    st.title("📊 Exploratory Data Analysis")
    if not history.empty:
        history["fetched_at"] = pd.to_datetime(history["fetched_at"])

        st.markdown("#### 🕒 Historical AQI (Last 72 Hours)")
        st.plotly_chart(history_chart(history), use_container_width=True)

        st.markdown("#### 📊 AQI Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(
                history, x="aqi", nbins=30,
                color_discrete_sequence=["#6366f1"],
                title="AQI Frequency Distribution"
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            if "temperature" in history.columns:
                fig_scatter = px.scatter(
                    history, x="temperature", y="aqi",
                    color="aqi", color_continuous_scale="RdYlGn_r",
                    title="Temperature vs AQI",
                    opacity=0.7,
                )
                fig_scatter.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### 📈 Summary Statistics")
        st.dataframe(history.describe().round(2), use_container_width=True)
    else:
        st.warning("No historical data available.")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.title("🧠 Model Explainability (SHAP)")
    st.markdown("SHAP values show which features drive AQI predictions most.")

    shap_bar   = "data/processed/plots/shap_bar.png"
    shap_summ  = "data/processed/plots/shap_summary.png"

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(shap_bar):
            st.image(shap_bar, caption="Top Feature Importances (Mean |SHAP|)",
                     use_container_width=True)
        else:
            st.info("Run `python src/training_pipeline/explainability.py` to generate SHAP plots.")

    with col2:
        if os.path.exists(shap_summ):
            st.image(shap_summ, caption="SHAP Beeswarm Summary",
                     use_container_width=True)
        else:
            st.info("SHAP summary plot not yet generated.")

    st.markdown("""
    #### 📖 How to interpret SHAP
    - **Positive SHAP value** → feature pushes prediction **higher**
    - **Negative SHAP value** → feature pushes prediction **lower**
    - **Bar height** → overall importance of that feature across all predictions
    """)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Metrics":
    st.title("⚙️ Model Performance Metrics")
    if metrics:
        df_m = pd.DataFrame(metrics)
        st.markdown("#### Evaluation Results (Test Set)")
        st.dataframe(df_m.round(4), use_container_width=True, hide_index=True)

        # RMSE bar chart
        if "rmse" in df_m.columns:
            fig = px.bar(df_m, x="model", y="rmse",
                         color="rmse", color_continuous_scale="RdYlGn_r",
                         title="Model RMSE Comparison (lower is better)")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train models first to see metrics. Run: `python src/training_pipeline/train.py`")

    st.markdown("""
    #### 📐 Metrics Guide
    - **RMSE** — Root Mean Squared Error (lower is better, same unit as AQI)
    - **MAE** — Mean Absolute Error (average absolute AQI prediction error)
    - **R²** — Coefficient of determination (1.0 = perfect, 0 = baseline mean)
    """)
