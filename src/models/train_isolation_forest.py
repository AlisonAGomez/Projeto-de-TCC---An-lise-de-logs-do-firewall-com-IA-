import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[2]
DATA_PATH   = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_PATH  = BASE_DIR / "models"
SCALER_PATH = MODEL_PATH / "scaler.joblib"

MODEL_PATH.mkdir(parents=True, exist_ok=True)

print("Carregando dataset...")
df = pd.read_csv(DATA_PATH)

# Separa label se existir (para avaliação posterior)
y = None
if "final_anomaly" in df.columns:
    y = df["final_anomaly"].astype(int)
    df = df.drop(columns=["final_anomaly"])
    print(f"Label encontrado: {y.sum()} anomalias de {len(y)} registros")

# Treina só com tráfego normal
if y is not None:
    df_train = df[y == 0]
    print(f"Treinando com {len(df_train)} registros normais")
else:
    df_train = df.copy()
    print(f"Treinando com {len(df_train)} registros (sem filtro)")

# Scaler — salvo aqui, reutilizado pelo Autoencoder
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train)
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler salvo em: {SCALER_PATH}")

print("Treinando Isolation Forest...")
model = IsolationForest(
    n_estimators=150,
    contamination=0.05,   # ~5% dos dados esperados como anomalia
    random_state=42,
    n_jobs=-1
)
model.fit(X_train)

joblib.dump(model, MODEL_PATH / "isolation_forest.joblib")
print("Isolation Forest salvo!")

# Avaliação rápida se tiver labels
if y is not None:
    X_all = scaler.transform(df)
    preds = model.predict(X_all)
    y_pred = (preds == -1).astype(int)
    print("\n--- Avaliação no dataset completo ---")
    print(classification_report(y, y_pred, target_names=["Normal", "Anomalia"]))
```

---

**Ordem correta para rodar:**
```
1. feature_engineer.py   → gera features.csv
2. train_isolation_forest.py  → treina IF + salva scaler
3. train_autoencoder.py  → treina AE usando o mesmo scaler
4. run_inference.py      → detecta e classifica