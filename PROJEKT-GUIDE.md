# Projekt-Guide: Notebooks & Risiko-Forecast erklärt

Dieser Leitfaden erklärt **(A)** was jedes Notebook im Projekt tut und **(B)** wie der
**Risiko-Forecast** in der Streamlit-App funktioniert — Schritt für Schritt, mit Formeln,
Fundstellen im Code und Ideen zur Illustration.

---

## A · Die Notebooks im Überblick

| Datei | Thema | Kernergebnis |
|---|---|---|
| `14-...-projektarbeit-AS.ipynb` | **Haupt-Notebook**: EDA → Feature Engineering → Baselines → SMOTE → XGBoost + Optuna → Evaluation → Modell-Export | Getuntes XGBoost, ROC-AUC 0,98 / PR-AUC 0,89; bei recall-optimiertem Schwellenwert 62 von 68 Ausfällen erkannt |
| `15-...-clusteranalyse-3d-AS.ipynb` | **Unüberwacht & 3D**: PCA in den 3D-Raum, K-Means/DBSCAN/hierarchisch, interaktiver Plotly-Plot | Ausfälle liegen strukturiert am Rand; Cluster = Risikozonen |
| `16-...-anomalie-erkennung-AS.ipynb` | **Ausreißer/Anomalie** ohne Labels: Isolation Forest, LOF, Elliptic Envelope | Trifft echte Ausfälle ~10× besser als Zufall (Precision 0,33 bei Basisrate 0,034) |
| `17-...-ensembles-erklaerbarkeit-AS.ipynb` | **Ensembles & Erklärbarkeit**: Baum → Bagging (RF) → Boosting → Stacking; Permutation Importance; SHAP | Boosting gewinnt (PR-AUC 0,89); SHAP erklärt jede Vorhersage |
| `18-...-automl-pycaret-AS.ipynb` | **AutoML-Benchmark**: PyCaret (~15 Modelle, isolierte Umgebung) vs. handgebautes Modell | AutoML-Sieger LightGBM (AUC 0,97); Dummy beweist die „Accuracy täuscht"-These |

> **Roter Faden:** Notebook 14 ist das Herz (Vorhersage). 15–18 sind Erweiterungen, die das
> Modell *verstehen* (Cluster/Anomalie), *vervollständigen* (Bagging/Stacking) und *begründen*
> (SHAP) bzw. *einordnen* (AutoML).

Empfohlene Reihenfolge zum Durchlesen: **14 → 15 → 16 → 17 → 18**. Jedes Notebook ist
eigenständig lauffähig (eigener Daten-Loader, identisches `build_features`).

---

## B · Der Risiko-Forecast (Streamlit-App, Tab 2)

**Fundstelle:** `app.py`, Abschnitt `with tab_forecast:` (Tab „📅 Forecast & Wartungsfahrplan").

### B.1 Die Grundidee

Der AI4I-Datensatz hat **keine Zeitstempel**. Aber der **Werkzeugverschleiß** (`Tool wear [min]`)
wächst mit jeder Betriebsminute — er ist also ein natürlicher **Ersatz für die Zeit**.

Der Forecast ist damit eine **Was-wäre-wenn-Simulation**:
> „Wenn diese Maschine so weiterläuft (alle anderen Sensorwerte bleiben gleich) und nur der
> Verschleiß steigt — wie entwickelt sich dann ihr Ausfallrisiko über die nächsten Tage?"

### B.2 Schritt für Schritt (mit Code)

**1) Verschleiß-Pfad als Zeitachse**
```python
current_wear = float(inputs["Tool wear [min]"])
wear_path = np.arange(current_wear, MAX_TOOL_WEAR + 1, 1.0)   # MAX_TOOL_WEAR = 253
```
Wir erzeugen alle Verschleißwerte vom aktuellen Stand bis zum Werkzeug-Lebensende.

**2) Risikokurve vom Modell**
```python
sim = pd.DataFrame([inputs] * len(wear_path))   # aktuellen Zustand vervielfachen
sim["Tool wear [min]"] = wear_path              # nur den Verschleiß variieren
probs = predict_proba_frame(model, sim)         # Risiko je Verschleiß-Schritt
```
Das Modell liefert für jeden Verschleiß-Schritt eine Ausfallwahrscheinlichkeit → die rote Kurve.

**3) Verschleiß → Kalenderdatum**
```python
days  = (wear_path - current_wear) / wear_per_day   # wear_per_day = Minuten Werkzeugeinsatz/Tag
dates = pd.Timestamp.today().normalize() + pd.to_timedelta(days, unit="D")
```
Über den täglichen Werkzeugeinsatz wird aus „Verschleiß" ein echtes Datum auf der x-Achse.

**4) Die zwei Schwellenwerte**
```python
econ_thr        = cost_planned / cost_failure   # z.B. 500/8000 = 0,0625 = 6,25 %
MODEL_THRESHOLD = 0.092                          # recall-optimiert aus Notebook 14
```
- **Wirtschaftliche Schwelle:** Wartung lohnt sich, sobald die *erwarteten* Ausfallkosten die
  Wartungskosten übersteigen. Erwartete Ausfallkosten = Risiko × Ausfallkosten. Setzt man das
  gleich den Wartungskosten, kürzt sich heraus: **Risiko ≥ geplante Wartung / ungeplanter Ausfall**.
- **Modell-Alarmschwelle:** der Schwellenwert, ab dem das Modell offiziell „Ausfall" meldet
  (auf Recall ≥ 90 % gesetzt).

**5) Erste Überschreitung → Wartungsfahrplan**
```python
def first_crossing(threshold):
    idx = np.where(probs >= threshold)[0]
    return int(idx[0]) if len(idx) else None
```
Der erste Tag, an dem die Kurve die jeweilige Schwelle reißt, wird zum Meilenstein im Fahrplan
(Heute → Wartung wirtschaftlich → Modell-Alarm → Werkzeug-Lebensende), jeweils mit Datum,
Tagen und Risiko.

**6) Flotten-ROI (Break-Even)**
```python
avoided_per_year   = n_machines * failures_per_year * MODEL_RECALL   # MODEL_RECALL = 0,91
saving_per_avoided = cost_failure - cost_planned                     # je vermiedenem Ausfall
annual_savings     = avoided_per_year * saving_per_avoided
monthly_savings    = annual_savings / 12
breakeven_months   = invest / monthly_savings                        # Amortisation
```
Logik: Das Modell fängt 91 % der Ausfälle ab (gemessener Recall). Jeder abgefangene Ausfall
kostet statt 8.000 € nur 500 € → Ersparnis 7.500 €. Hochgerechnet auf den Park ergibt sich eine
jährliche Ersparnis, aus der sich der Break-Even der Projektinvestition berechnet.

### B.3 Was die App daraus macht (4 Visualisierungen)

1. **Risikokurve über Datum** mit beiden Schwellenlinien und Markern an den Überschreitungen.
2. **Wartungsfahrplan-Tabelle** mit konkreten Terminen.
3. **Kosten-Forecast:** erwartete Ausfallkosten(t) vs. Wartungskosten-Linie — der Schnittpunkt
   ist der wirtschaftliche Wartungszeitpunkt.
4. **Flotten-ROI:** kumulierte Ersparnis über 36 Monate mit Break-Even-Stern.

### B.4 Ehrliche Grenzen

- Es ist eine **Simulation**, kein echter Zeitreihen-Forecast: alle Sensoren außer dem Verschleiß
  bleiben konstant (ceteris paribus).
- Der **lineare Verschleiß-zu-Zeit-Zusammenhang** ist eine Annahme (`wear_per_day`).
- Das Modell sagt **Risiko**, keine **Restlebensdauer** in Stunden — die Zeitachse entsteht erst
  durch die Verschleiß-Annahme.

### B.5 Wie man das (noch besser) illustrieren könnte

- **Risikozonen einfärben:** grüne/gelbe/rote Hintergrundbänder hinter der Kurve (0–6 % / 6–9 % / >9 %).
- **Wartungsfenster schattieren:** den Bereich zwischen wirtschaftlicher und Modell-Schwelle als
  hellgrünes Band markieren („hier warten").
- **Heute-Linie:** eine vertikale Linie bei „heute" zur Orientierung.
- **Countdown-Kachel:** „Wartung in X Tagen" als große Metrik über dem Diagramm.
- **Mehrere Szenarien:** zwei, drei Verschleiß-Geschwindigkeiten (z. B. Zwei-/Drei-Schicht-Betrieb)
  als mehrere Kurven übereinander, um den Effekt der Auslastung zu zeigen.
- **Unsicherheit andeuten:** statt einer Linie ein leichtes Band (z. B. ±5 % Risiko), um zu
  betonen, dass es eine Schätzung ist.

---

*Erstellt im Rahmen der ML-Projektarbeit · Alexander Sagert · Juni 2026*
