import pandas as pd
import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
REPORTS_PATH = os.path.join(BASE_DIR, "reports")
MODEL_IF = os.path.join(BASE_DIR, "models", "isolation_forest.joblib")
MODEL_AE = os.path.join(BASE_DIR, "models", "autoencoder")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

os.makedirs(REPORTS_PATH, exist_ok=True)

print("Carregando dados...")
df = pd.read_csv(DATA_PATH)

# Carregar modelos
iso = joblib.load(MODEL_IF)
autoencoder = tf.keras.models.load_model(MODEL_AE)
scaler = joblib.load(SCALER_PATH)

# Normalizar
X = scaler.transform(df)

# Isolation Forest
print("Rodando Isolation Forest...")
df["iso_score"] = iso.decision_function(X)
df["iso_anomaly"] = iso.predict(X) == -1

# Autoencoder
print("Rodando Autoencoder...")
reconstructions = autoencoder.predict(X)
mse = np.mean(np.square(X - reconstructions), axis=1)
df["ae_mse"] = mse

# Threshold automático (95%)
threshold = np.percentile(mse, 95)
df["ae_anomaly"] = mse > threshold

# Consolidação
df["final_anomaly"] = df["iso_anomaly"] | df["ae_anomaly"]

# Classificação textual
df["threat_level"] = df["final_anomaly"].map({
    True: "Suspeito",
    False: "Normal"
})

# Estatísticas
summary = {
    "total_events": len(df),
    "anomalies_detected": int(df["final_anomaly"].sum()),
    "anomaly_rate_percent": round(df["final_anomaly"].mean() * 100, 2),
    "autoencoder_threshold": float(threshold)
}

# Salvar relatório
df.to_csv(os.path.join(REPORTS_PATH, "infer.csv"), index=False)

import json
with open(os.path.join(REPORTS_PATH, "infer.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("Inferência concluída!")
print(summary)

