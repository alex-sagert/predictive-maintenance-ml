# -*- coding: utf-8 -*-
"""
Streamlit-Demo · Predictive Maintenance
----------------------------------------
Sensor-Slider -> Ausfallrisiko (Tacho) + Kostenrechner.

Start:  streamlit run app.py     (aus dem Projektordner)

Hinweis: Läuft sofort, auch ohne fertiges Modell. Wenn 'model.joblib' fehlt,
trainiert die App beim Start ein einfaches Fallback-Modell aus 'data/ai4i2020.csv'.
Mittwoch ersetzt du es einfach durch dein getuntes Modell (Abschnitt 10 im Notebook).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------- Konstanten
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "ai4i2020.csv"
MODEL_PATH = APP_DIR / "model.joblib"
RANDOM_STATE = 42

SENSORS = {
    "Air temperature [K]":      dict(min=295.0, max=305.0, default=300.0, step=0.1),
    "Process temperature [K]":  dict(min=305.0, max=314.0, default=310.0, step=0.1),
    "Rotational speed [rpm]":   dict(min=1160.0, max=2890.0, default=1500.0, step=10.0),
    "Torque [Nm]":              dict(min=3.0,  max=77.0,  default=40.0, step=0.5),
    "Tool wear [min]":          dict(min=0.0,  max=255.0, default=100.0, step=1.0),
}

st.set_page_config(page_title="Predictive Maintenance Demo",
                   page_icon="🔧", layout="wide")


# ----------------------------------------------------------------- Feature-Bau
def build_features(d: pd.DataFrame) -> pd.DataFrame:
    """Identisches Feature Engineering für Training UND Vorhersage.

    XGBoost erlaubt keine Spaltennamen mit [ ] < — daher saubere Feature-Namen.
    """
    d = d.copy()
    out = pd.DataFrame(index=d.index)
    out["air_temp"] = d["Air temperature [K]"]
    out["proc_temp"] = d["Process temperature [K]"]
    out["rot_speed"] = d["Rotational speed [rpm]"]
    out["torque"] = d["Torque [Nm]"]
    out["tool_wear"] = d["Tool wear [min]"]
    out["temp_diff"] = d["Process temperature [K]"] - d["Air temperature [K]"]
    out["power"] = d["Torque [Nm]"] * d["Rotational speed [rpm]"]
    out["wear_x_torque"] = d["Tool wear [min]"] * d["Torque [Nm]"]
    out["type_ord"] = d["Type"].map({"L": 0, "M": 1, "H": 2}).fillna(1)
    return out


# ----------------------------------------------------------------- Modell laden
@st.cache_resource(show_spinner="Modell wird geladen/trainiert …")
def load_model():
    """Lädt model.joblib, sonst Fallback-Training auf der CSV."""
    if MODEL_PATH.exists():
        import joblib
        return joblib.load(MODEL_PATH), "Geladen: dein trainiertes model.joblib"

    # ---- Fallback: schnelles XGBoost direkt aus der CSV ----
    from xgboost import XGBClassifier
    df = pd.read_csv(DATA_PATH)
    X = build_features(df)
    y = df["Machine failure"].astype(int)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X, y)
    return model, "Fallback-Modell (live aus CSV trainiert) – ersetze es durch dein model.joblib"


def predict_proba(model, inputs: dict) -> float:
    row = pd.DataFrame([inputs])
    X = build_features(row)
    try:
        return float(model.predict_proba(X)[0, 1])
    except Exception:
        # Pipeline, die rohe Spalten erwartet
        return float(model.predict_proba(pd.DataFrame([inputs]))[0, 1])


# ----------------------------------------------------------------- Tacho (SVG)
def gauge_svg(p: float) -> str:
    """Halbkreis-Tacho 0–100 % als reines SVG (keine Extra-Abhängigkeit)."""
    pct = max(0.0, min(1.0, p))
    angle = 180 * pct                      # 0..180 Grad
    a = np.radians(180 - angle)
    cx, cy, r = 150, 150, 120
    x, y = cx + r * np.cos(a), cy - r * np.sin(a)
    large = 1 if angle > 180 else 0
    color = "#16a34a" if pct < 0.30 else ("#f59e0b" if pct < 0.60 else "#dc2626")
    return f"""
    <svg viewBox="0 0 300 180" width="320" height="200">
      <path d="M30 150 A120 120 0 0 1 270 150" fill="none" stroke="#e5e7eb" stroke-width="22"/>
      <path d="M30 150 A120 120 0 {large} 1 {x:.1f} {y:.1f}" fill="none"
            stroke="{color}" stroke-width="22" stroke-linecap="round"/>
      <text x="150" y="120" text-anchor="middle" font-size="42" font-weight="700"
            fill="{color}">{pct*100:.1f}%</text>
      <text x="150" y="150" text-anchor="middle" font-size="15" fill="#6b7280">Ausfallrisiko</text>
    </svg>"""


# ================================================================= UI
st.title("🔧 Predictive Maintenance – Live-Risikobewertung")
st.caption("Projektarbeit ML · Alexander Sagert · Juni 2026")

if not DATA_PATH.exists():
    st.error(f"Datensatz nicht gefunden: {DATA_PATH}")
    st.stop()

model, model_info = load_model()
st.info(model_info, icon="ℹ️")

# ---- Sidebar: Sensor-Eingaben ----
st.sidebar.header("Maschinen-Sensorwerte")
maschinentyp = st.sidebar.selectbox("Produktqualität / Typ", ["L", "M", "H"], index=0,
                                     help="L = low, M = medium, H = high")
inputs = {"Type": maschinentyp}
for name, cfg in SENSORS.items():
    inputs[name] = st.sidebar.slider(name, cfg["min"], cfg["max"], cfg["default"], cfg["step"])

st.sidebar.markdown("---")
st.sidebar.subheader("💶 Kostenannahmen")
cost_planned = st.sidebar.number_input("Geplante Wartung (€)", value=500, step=50)
cost_failure = st.sidebar.number_input("Ungeplanter Ausfall (€)", value=8000, step=500)

# ---- Vorhersage ----
proba = predict_proba(model, inputs)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### Risiko")
    st.markdown(gauge_svg(proba), unsafe_allow_html=True)
    if proba < 0.30:
        st.success("Niedriges Risiko – Normalbetrieb.", icon="✅")
    elif proba < 0.60:
        st.warning("Erhöhtes Risiko – Wartung einplanen.", icon="⚠️")
    else:
        st.error("Hohes Risiko – Wartung dringend empfohlen!", icon="🚨")

with col2:
    st.markdown("### 💶 Erwarteter Kostenvorteil")
    # Erwartete Kosten: ohne Eingriff vs. geplante Wartung jetzt
    exp_cost_donothing = proba * cost_failure
    exp_cost_maintain = cost_planned
    saving = exp_cost_donothing - exp_cost_maintain
    st.metric("Erwartete Ausfallkosten (ohne Eingriff)", f"{exp_cost_donothing:,.0f} €")
    st.metric("Kosten geplante Wartung jetzt", f"{exp_cost_maintain:,.0f} €")
    st.metric("Erwartete Ersparnis durch Eingriff", f"{saving:,.0f} €",
              delta="Wartung lohnt sich" if saving > 0 else "Beobachten")
    st.caption("Erwartete Ausfallkosten = Ausfallwahrscheinlichkeit × Kosten ungeplanter Ausfall.")

with st.expander("ℹ️ Wie funktioniert das?"):
    st.markdown(
        "- Das Modell schätzt aus den Sensorwerten die **Ausfallwahrscheinlichkeit**.\n"
        "- Der **Kostenrechner** vergleicht die erwarteten Ausfallkosten mit den Kosten "
        "einer geplanten Wartung → zeigt, ab wann sich ein Eingriff rechnet.\n"
        "- Schwellenwert (30 %/60 %) ist eine **Geschäftsentscheidung**, kein fixer ML-Wert."
    )
