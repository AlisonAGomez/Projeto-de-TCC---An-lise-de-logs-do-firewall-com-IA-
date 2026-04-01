import pandas as pd
import os
import numpy as np
import joblib
import json
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH    = BASE_DIR / "data" / "processed" / "features.csv"
IP_PATH      = BASE_DIR / "data" / "processed" / "src_ip_index.csv"
REPORTS_PATH = BASE_DIR / "reports"
MODEL_IF     = BASE_DIR / "models" / "isolation_forest.joblib"
MODEL_AE     = BASE_DIR / "models" / "autoencoder"
SCALER_PATH  = BASE_DIR / "models" / "scaler.joblib"

REPORTS_PATH.mkdir(parents=True, exist_ok=True)

print("Carregando dados...")
df = pd.read_csv(DATA_PATH)
ip_index = pd.read_csv(IP_PATH)

iso        = joblib.load(MODEL_IF)
autoencoder = tf.keras.models.load_model(MODEL_AE)
scaler     = joblib.load(SCALER_PATH)

X = scaler.transform(df)

print("Rodando Isolation Forest...")
df["iso_score"]   = iso.decision_function(X)
df["iso_anomaly"] = iso.predict(X) == -1

print("Rodando Autoencoder...")
reconstructions = autoencoder.predict(X)
mse = np.mean(np.square(X - reconstructions), axis=1)
df["ae_mse"]     = mse
threshold        = np.percentile(mse, 95)
df["ae_anomaly"] = mse > threshold

df["final_anomaly"] = df["iso_anomaly"] | df["ae_anomaly"]

# --- Classificação de tipo de ataque ---
def classificar_ataque(row):
    if not row["final_anomaly"]:
        return "Normal"
    if row.get("port_scan_score", 0) > 0.5:
        return "Port Scan"
    if row.get("brute_force_score", 0) > 50:
        return "Brute Force"
    if row.get("max_conn_per_window", 0) > 100:
        return "DDoS"
    return "Suspeito"

df["attack_type"] = df.apply(classificar_ataque, axis=1)

# Reanexa IPs ao relatório
result = pd.concat([ip_index, df], axis=1)

summary = {
    "total_events":           len(df),
    "anomalies_detected":     int(df["final_anomaly"].sum()),
    "anomaly_rate_percent":   round(df["final_anomaly"].mean() * 100, 2),
    "autoencoder_threshold":  float(threshold),
    "attack_breakdown": df[df["final_anomaly"]]["attack_type"].value_counts().to_dict()
}

result.to_csv(REPORTS_PATH / "infer.csv", index=False)
with open(REPORTS_PATH / "infer.json", "w") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print("Inferência concluída!")
print(summary)