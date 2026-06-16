# -*- coding: utf-8 -*-
"""Läuft in der ISOLIERTEN .venv_pycaret. AutoML-Benchmark mit PyCaret auf AI4I 2020.
Exportiert das Leaderboard nach pycaret_results/leaderboard.csv (das Haupt-Notebook liest es)."""
import os
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "ai4i2020.csv"
OUTDIR = HERE / "pycaret_results"
OUTDIR.mkdir(exist_ok=True)

# --- Daten vorbereiten: gleiche Logik wie im Projekt, Leakage vermeiden ---
df = pd.read_csv(DATA)
# IDs und Fehlertyp-Spalten (faktisch Labels) entfernen
drop_cols = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
data = df.drop(columns=drop_cols)
print("PyCaret-Eingabe:", data.shape, "| Spalten:", list(data.columns))

from pycaret.classification import setup, compare_models, pull, save_model

# fix_imbalance=True -> PyCaret nutzt SMOTE intern (3,4 % Ausfälle)
setup(
    data=data,
    target="Machine failure",
    session_id=42,
    fix_imbalance=True,
    train_size=0.8,
    verbose=False,
    html=False,
    n_jobs=-1,
)

# Über mehrere Modelle vergleichen (sortiert nach AUC); fold=5 für Tempo
best = compare_models(sort="AUC", fold=5)
leaderboard = pull()
print("\n=== LEADERBOARD ===")
print(leaderboard.to_string())

leaderboard.to_csv(OUTDIR / "leaderboard.csv")
with open(OUTDIR / "best_model.txt", "w", encoding="utf-8") as f:
    f.write(str(best))
print("\nGespeichert:", OUTDIR / "leaderboard.csv")
print("Bestes Modell:", type(best).__name__)
