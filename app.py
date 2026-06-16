# -*- coding: utf-8 -*-
"""
Streamlit-Demo · Predictive Maintenance
----------------------------------------
Tab 1: Sensor-Slider -> Ausfallrisiko (Tacho) + Kostenrechner.
Tab 2: Forecast entlang der Betriebszeit -> Wartungsfahrplan mit Terminen + Flotten-ROI.
Tab 3: Analysen -> Clustering/3D, Anomalie-Erkennung, Ensembles & SHAP, AutoML-Benchmark.

Start:  streamlit run app.py     (aus dem Projektordner)

Hinweis: Läuft sofort, auch ohne fertiges Modell. Wenn 'model.joblib' fehlt,
trainiert die App beim Start ein einfaches Fallback-Modell aus 'data/ai4i2020.csv'.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------- Konstanten
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "ai4i2020.csv"
MODEL_PATH = APP_DIR / "model.joblib"
VIZ_DIR = APP_DIR / "visualisierung"
RANDOM_STATE = 42

MODEL_THRESHOLD = 0.092   # Recall-orientierte Alarmschwelle aus dem Notebook (Abschnitt 7)
MODEL_RECALL = 0.91       # gemessener Recall des finalen Modells auf der Testmenge
MAX_TOOL_WEAR = 253.0     # max. beobachteter Verschleiß im Datensatz (~Werkzeug-Lebensende)

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


def predict_proba_frame(model, frame: pd.DataFrame) -> np.ndarray:
    """Wahrscheinlichkeiten für einen DataFrame mit Roh-Sensorspalten."""
    return model.predict_proba(build_features(frame))[:, 1]


def show_img(name: str, caption: str = ""):
    """Zeigt eine gerenderte Analyse-Grafik aus visualisierung/ (robust, falls Datei fehlt)."""
    p = VIZ_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Grafik nicht gefunden: {name}")


# ----------------------------------------------------------------- Tacho (SVG)
def gauge_svg(p: float) -> str:
    """Halbkreis-Tacho 0–100 % als reines SVG (keine Extra-Abhängigkeit)."""
    pct = max(0.0, min(1.0, p))
    angle = 180 * pct
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
cost_planned = st.sidebar.number_input("Geplante Wartung (€)", value=500, step=50, min_value=1)
cost_failure = st.sidebar.number_input("Ungeplanter Ausfall (€)", value=8000, step=500, min_value=1)

# ---- Aktuelle Vorhersage ----
proba = float(predict_proba_frame(model, pd.DataFrame([inputs]))[0])

tab_now, tab_forecast, tab_analysen = st.tabs(
    ["🔧 Risiko jetzt", "📅 Forecast & Wartungsfahrplan", "📊 Analysen"])

# ================================================================= Tab 1: Risiko jetzt
with tab_now:
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

# ================================================================= Tab 2: Forecast
with tab_forecast:
    st.markdown("### 📅 Risiko-Forecast entlang der Betriebszeit")
    st.caption(
        "Simulation: Der **Werkzeugverschleiß wächst mit jeder Betriebsminute** und dient als "
        "Zeitachse. Die übrigen Sensorwerte werden konstant auf den aktuellen Werten gehalten.")

    cfg1, cfg2, cfg3 = st.columns(3)
    wear_per_day = cfg1.number_input("Werkzeugeinsatz pro Tag (min)", min_value=5,
                                     max_value=480, value=30, step=5,
                                     help="Wie viele Minuten pro Tag ist das Werkzeug im Eingriff?")
    n_machines = cfg2.number_input("Maschinen im Park", min_value=1, max_value=1000, value=20)
    invest = cfg3.number_input("Projektinvestition (€)", min_value=0, max_value=1_000_000,
                               value=25_000, step=1_000,
                               help="Einmalkosten für Einführung des Predictive-Maintenance-Systems")

    current_wear = float(inputs["Tool wear [min]"])
    if current_wear >= MAX_TOOL_WEAR - 1:
        st.warning("Werkzeug ist bereits am Lebensende (max. Verschleiß) – Wartung sofort fällig.",
                   icon="🚨")
        wear_path = np.array([current_wear])
    else:
        wear_path = np.arange(current_wear, MAX_TOOL_WEAR + 1, 1.0)

    sim = pd.DataFrame([inputs] * len(wear_path))
    sim["Tool wear [min]"] = wear_path
    probs = predict_proba_frame(model, sim)

    days = (wear_path - current_wear) / wear_per_day
    today = pd.Timestamp.today().normalize()
    dates = today + pd.to_timedelta(days, unit="D")

    # Schwellen: wirtschaftlich (Wartung billiger als erwartete Ausfallkosten) + Modell-Alarm
    econ_thr = cost_planned / cost_failure

    def first_crossing(threshold):
        idx = np.where(probs >= threshold)[0]
        return int(idx[0]) if len(idx) else None

    i_econ = first_crossing(econ_thr)
    i_model = first_crossing(MODEL_THRESHOLD)
    i_end = len(wear_path) - 1

    # ---------------- Risiko-Kurve (Plotly) ----------------
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=probs * 100, mode="lines",
                             name="Ausfallrisiko", line=dict(color="#e03131", width=3)))
    fig.add_hline(y=econ_thr * 100, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"wirtschaftliche Schwelle ({econ_thr:.1%})",
                  annotation_position="top left")
    fig.add_hline(y=MODEL_THRESHOLD * 100, line_dash="dot", line_color="#1971c2",
                  annotation_text=f"Modell-Alarmschwelle ({MODEL_THRESHOLD:.1%})",
                  annotation_position="bottom right")
    for i, lbl, col in [(i_econ, "Wartung wirtschaftlich", "#f59e0b"),
                        (i_model, "Modell-Alarm", "#1971c2")]:
        if i is not None and i > 0:
            fig.add_trace(go.Scatter(x=[dates[i]], y=[probs[i] * 100], mode="markers+text",
                                     marker=dict(size=12, color=col, symbol="diamond"),
                                     text=[lbl], textposition="top center", showlegend=False))
    fig.update_layout(height=420, margin=dict(t=30, b=10),
                      yaxis_title="Ausfallrisiko (%)", xaxis_title="Datum",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Wartungsfahrplan ----------------
    st.markdown("### 🗓️ Wartungsfahrplan")

    def plan_row(label, i):
        if i is None:
            return {"Meilenstein": label, "Datum": "—", "in Tagen": "—",
                    "Risiko": "wird im Horizont nicht erreicht"}
        return {"Meilenstein": label,
                "Datum": dates[i].strftime("%d.%m.%Y"),
                "in Tagen": f"{days[i]:.0f}",
                "Risiko": f"{probs[i]:.1%}"}

    fahrplan = pd.DataFrame([
        plan_row("Heute (aktueller Zustand)", 0),
        plan_row(f"Wartung wirtschaftlich sinnvoll (Risiko ≥ {econ_thr:.1%})", i_econ),
        plan_row(f"Modell-Alarmschwelle erreicht (Risiko ≥ {MODEL_THRESHOLD:.1%})", i_model),
        plan_row("Werkzeug-Lebensende (max. Verschleiß)", i_end),
    ])
    st.dataframe(fahrplan, hide_index=True, use_container_width=True)

    if i_econ is not None and i_econ == 0:
        st.error("Das Risiko liegt **schon heute** über der wirtschaftlichen Schwelle – "
                 "Wartung jetzt einplanen.", icon="🚨")
    elif i_econ is not None:
        st.success(f"Empfehlung: Wartung bis **{dates[i_econ].strftime('%d.%m.%Y')}** einplanen "
                   f"(in {days[i_econ]:.0f} Tagen). Ab dann übersteigen die erwarteten "
                   f"Ausfallkosten die Wartungskosten.", icon="✅")
    else:
        st.info("Im Simulationshorizont bleibt das Risiko unter der wirtschaftlichen Schwelle – "
                "regulärer Werkzeugwechsel am Lebensende genügt.", icon="ℹ️")

    # ---------------- Kosten-Forecast ----------------
    st.markdown("### 💶 Kosten-Forecast: Weiterbetrieb vs. geplante Wartung")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates, y=probs * cost_failure, mode="lines",
                              name="Erwartete Ausfallkosten bei Weiterbetrieb",
                              line=dict(color="#e03131", width=3)))
    fig2.add_hline(y=cost_planned, line_dash="dash", line_color="#16a34a",
                   annotation_text=f"Geplante Wartung ({cost_planned:,.0f} €)",
                   annotation_position="top left")
    fig2.update_layout(height=340, margin=dict(t=30, b=10),
                       yaxis_title="Erwartete Kosten (€)", xaxis_title="Datum",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Erwartete Ausfallkosten(t) = Ausfallwahrscheinlichkeit(t) × Kosten ungeplanter "
               "Ausfall. Wo die rote Kurve die grüne Linie schneidet, wird Wartung wirtschaftlich.")

    # ---------------- Flotten-ROI ----------------
    st.markdown("### 🏭 Flotten-ROI: Wann amortisiert sich die Investition?")

    failures_per_year = st.slider(
        "Ungeplante Ausfälle pro Maschine und Jahr (ohne Predictive Maintenance)",
        min_value=0.5, max_value=12.0, value=2.0, step=0.5,
        help="Branchenannahme als Baseline – wie oft fällt eine Maschine heute ungeplant aus?")

    # Ehrliche Rechnung: Das Modell fängt (gemessener Recall) 91 % der Ausfälle ab.
    # Jeder abgefangene Ausfall kostet statt eines ungeplanten Stillstands nur eine geplante Wartung.
    avoided_per_year = n_machines * failures_per_year * MODEL_RECALL
    saving_per_avoided = max(cost_failure - cost_planned, 0)
    annual_fleet_savings = avoided_per_year * saving_per_avoided
    monthly_savings = annual_fleet_savings / 12.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Abgefangene Ausfälle pro Jahr (Park)", f"{avoided_per_year:,.1f}",
              help=f"Maschinen × Ausfälle/Jahr × Modell-Recall ({MODEL_RECALL:.0%})")
    m2.metric("Erwartete Ersparnis pro Jahr (Park)", f"{annual_fleet_savings:,.0f} €",
              help="Abgefangene Ausfälle × (Kosten ungeplanter Ausfall − geplante Wartung)")
    if monthly_savings > 0:
        breakeven_months = invest / monthly_savings
        be_date = today + pd.DateOffset(months=int(np.ceil(breakeven_months)))
        m3.metric("Break-Even der Investition", f"{breakeven_months:.1f} Monate",
                  delta=f"ca. {be_date.strftime('%m/%Y')}")
    else:
        breakeven_months = None
        m3.metric("Break-Even der Investition", "—")

    months = np.arange(0, 37)
    cum_savings = months * monthly_savings
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=months, y=cum_savings, mode="lines",
                              name="Kumulierte Ersparnis (Park)",
                              line=dict(color="#16a34a", width=3)))
    fig3.add_hline(y=invest, line_dash="dash", line_color="#6b7280",
                   annotation_text=f"Investition ({invest:,.0f} €)",
                   annotation_position="bottom right")
    if breakeven_months is not None and breakeven_months <= 36:
        fig3.add_trace(go.Scatter(x=[breakeven_months], y=[invest], mode="markers+text",
                                  marker=dict(size=13, color="#16a34a", symbol="star"),
                                  text=["Break-Even"], textposition="top center",
                                  showlegend=False))
    fig3.update_layout(height=340, margin=dict(t=30, b=10),
                       yaxis_title="Kumulierte Ersparnis (€)", xaxis_title="Monate ab heute",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        f"Annahmen: {n_machines} Maschinen · {failures_per_year:g} ungeplante Ausfälle je Maschine "
        f"und Jahr (Baseline) · das Modell fängt davon {MODEL_RECALL:.0%} ab (gemessener Recall auf "
        f"der Testmenge) · Ersparnis je abgefangenem Ausfall = {cost_failure:,.0f} € − "
        f"{cost_planned:,.0f} € = {saving_per_avoided:,.0f} €. Vereinfachte Modellrechnung – alle "
        f"Parameter anpassbar.")

# ================================================================= Tab 3: Analysen
with tab_analysen:
    st.markdown("### 📊 Vertiefende Modell-Analysen")
    st.caption("Ergebnisse aus den Begleit-Notebooks (15–18): unüberwachtes Lernen, "
               "Anomalie-Erkennung, Ensemble-Vergleich, Erklärbarkeit und AutoML-Benchmark.")

    sub_clust, sub_anom, sub_ens, sub_auto = st.tabs(
        ["🧭 Clustering & 3D", "🔎 Anomalie", "🧩 Ensembles & SHAP", "🤖 AutoML"])

    # ---------- Clustering & 3D ----------
    with sub_clust:
        st.markdown("**Wo im Merkmalsraum sitzen die Ausfälle?** Per PCA in den 3D-Raum projiziert "
                    "— Ausfälle liegen strukturiert am Rand, Fehlertypen in eigenen Zonen.")
        c1, c2 = st.columns(2)
        with c1:
            show_img("01_pca_3d_ausfaelle.png", "Ausfälle (rot) im 3D-Raum")
        with c2:
            show_img("02_pca_3d_fehlertypen.png", "Fehlertypen in unterschiedlichen Zonen")
        c3, c4 = st.columns(2)
        with c3:
            show_img("05_kmeans_3d_risikozonen.png", "K-Means: Risikozone hervorgehoben")
        with c4:
            show_img("06_methodenvergleich.png", "K-Means vs. Ward vs. DBSCAN")
        st.info("Befund: Unüberwachtes Clustering findet dieselben Risikotreiber (hohe mechanische "
                "Last) wie das überwachte Modell.", icon="💡")

    # ---------- Anomalie ----------
    with sub_anom:
        st.markdown("**Auffällige Maschinen ohne Labels finden** (Isolation Forest, LOF, Elliptic "
                    "Envelope) — der Cold-Start-Fall, wenn noch keine Ausfall-Historie existiert.")
        c1, c2 = st.columns(2)
        with c1:
            show_img("08_anomalie_scoreverteilung.png", "Anomalie-Score: Ausfälle liegen höher")
        with c2:
            show_img("09_anomalien_3d.png", "Anomalien im 3D-Raum (grün = Treffer)")
        st.info("Ohne je ein Label gesehen zu haben, trifft Isolation Forest echte Ausfälle ~10× "
                "besser als Zufall (Precision 0,33 bei Basisrate 0,034, ROC-AUC 0,86).", icon="💡")

    # ---------- Ensembles & SHAP ----------
    with sub_ens:
        st.markdown("**Die ganze Ensemble-Progression** — ein Baum → Bagging → Boosting → Stacking "
                    "— und **SHAP** erklärt jede einzelne Vorhersage.")
        c1, c2 = st.columns(2)
        with c1:
            show_img("10_ensemble_vergleich.png", "Ensemble-Vergleich (PR-AUC & Recall)")
        with c2:
            show_img("13_shap_waterfall.png", "SHAP: warum diese Maschine als riskant gilt")
        st.info("Keine Black Box: Für eine Risiko-Maschine treiben Verschleiß × Drehmoment, "
                "Drehmoment, Drehzahl und Leistung das Risiko — physikalisch nachvollziehbar.",
                icon="💡")

    # ---------- AutoML ----------
    with sub_auto:
        st.markdown("**Handgebaut + Optuna vs. AutoML (PyCaret).** Auch die automatische Pipeline "
                    "landet bei Boosting (Sieger LightGBM).")
        show_img("14_pycaret_leaderboard.png", "PyCaret AutoML-Leaderboard (AUC, 5-fach-CV)")
        st.info("Die wichtigste Lektion liefert der Dummy-Classifier: 96,6 % Accuracy bei AUC 0,5 "
                "und Recall 0 % — der Beweis, warum dieses Projekt auf Recall/PR-AUC statt Accuracy "
                "optimiert.", icon="💡")
