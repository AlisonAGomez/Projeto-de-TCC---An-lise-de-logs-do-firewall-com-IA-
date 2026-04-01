import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH    = BASE_DIR / "data" / "processed" / "features.csv"
REPORTS_PATH = BASE_DIR / "reports"
MODEL_IF     = BASE_DIR / "models" / "isolation_forest.joblib"
MODEL_AE     = BASE_DIR / "models" / "autoencoder"
SCALER_PATH  = BASE_DIR / "models" / "scaler.joblib"

REPORTS_PATH.mkdir(parents=True, exist_ok=True)

print("Carregando dados...")

df = pd.read_csv(DATA_PATH)

# 🔹 remove colunas que NÃO são features do modelo
if "label" in df.columns:
    df = df.drop(columns=["label"])

if "final_anomaly" in df.columns:
    df = df.drop(columns=["final_anomaly"])

# 🔹 pega apenas numéricas
X = df.select_dtypes(include=[np.number]).fillna(0)

# 🔹 carrega modelos
iso         = joblib.load(MODEL_IF)
autoencoder = tf.keras.models.load_model(MODEL_AE)
scaler      = joblib.load(SCALER_PATH)

X_scaled = scaler.transform(X)

# 🔹 Isolation Forest
print("Rodando Isolation Forest...")
df["iso_score"]   = iso["model"].decision_function(X_scaled)
df["iso_anomaly"] = iso["model"].predict(X_scaled) == -1

# 🔹 Autoencoder
print("Rodando Autoencoder...")
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.square(X_scaled - reconstructions), axis=1)

df["ae_mse"] = mse

# usa threshold salvo no treino (ou recalcula)
threshold = float(np.percentile(mse, 90))
df["ae_anomaly"] = mse > threshold

# 🔹 combinação final
df["final_anomaly"] = df["iso_anomaly"] | df["ae_anomaly"]

# --- Classificação simples de ataque ---
def classificar_ataque(row):
    if not row["final_anomaly"]:
        return "Normal"
    if row.get("unique_dst_ports", 0) > 50:
        return "Port Scan"
    if row.get("total_connections", 0) > 150:
        return "DDoS"
    if row.get("blocks", 0) > 100:
        return "Brute Force"
    return "Suspeito"

df["attack_type"] = df.apply(classificar_ataque, axis=1)

# 🔹 relatório final
summary = {
    "total_events": int(len(df)),
    "anomalies_detected": int(df["final_anomaly"].sum()),
    "anomaly_rate_percent": round(df["final_anomaly"].mean() * 100, 2),
    "autoencoder_threshold": float(threshold),
    "attack_breakdown": df[df["final_anomaly"]]["attack_type"].value_counts().to_dict()
}

# 🔹 salvar resultados
df.to_csv(REPORTS_PATH / "infer.csv", index=False)

with open(REPORTS_PATH / "infer.json", "w") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print("\n✅ Inferência concluída!")
print(json.dumps(summary, indent=4, ensure_ascii=False))