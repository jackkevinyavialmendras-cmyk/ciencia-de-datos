"""Tabla comparativa de los modelos para las conclusiones del informe.

Uso:
    python scripts/05_comparar.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config as C

filas = []
for f in sorted(C.REPORTS.glob("metricas_*.json")):
    filas.append(json.loads(f.read_text(encoding="utf-8")))
if not filas:
    sys.exit("No hay métricas. Ejecuta primero los scripts 03 y 04.")

df = pd.DataFrame(filas).set_index("modelo")
print(df.T.fillna("-").to_string())
df.to_csv(C.REPORTS / "comparacion_modelos.csv")
print("\nGuardado en reports/comparacion_modelos.csv")
