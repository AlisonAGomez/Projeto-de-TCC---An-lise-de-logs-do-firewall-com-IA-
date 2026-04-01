import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[2]
DATA_PATH   = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_DIR   = BASE_DIR / "models" / "autoencoder"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Carregando dataset...")
df = pd.read_csv(DATA_PATH)

# Treina apenas com tráfego normal (se coluna existir)
# O Autoencoder aprende o padrão do normal — desvios = anomalia
if "final_anomaly" in df.columns:
    df_train = df[df["final_anomaly"] == False].drop(columns=["final_anomaly"])
    print(f"Treinando com {len(df_train)} registros normais (filtrado)")
else:
    df_train = df.copy()
    print(f"Treinando com {len(df_train)} registros (sem filtro de anomalia)")

# Carrega o scaler já treinado pelo Isolation Forest
# (ambos os modelos devem usar o mesmo scaler)
if SCALER_PATH.exists():
    from sklearn.preprocessing import StandardScaler
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(df_train)
    print("Scaler carregado de models/scaler.joblib")
else:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df_train)
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler criado e salvo (rode o Isolation Forest primeiro na próxima vez)")

input_dim = X.shape[1]
print(f"Dimensão de entrada: {input_dim} features")

# Arquitetura do Autoencoder
input_layer  = layers.Input(shape=(input_dim,))
encoded      = layers.Dense(64, activation="relu")(input_layer)
encoded      = layers.Dense(32, activation="relu")(encoded)
latent       = layers.Dense(16, activation="relu")(encoded)
decoded      = layers.Dense(32, activation="relu")(latent)
decoded      = layers.Dense(64, activation="relu")(decoded)
output_layer = layers.Dense(input_dim, activation="linear")(decoded)

autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# Early stopping: para quando a validação não melhora por 5 épocas
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("Treinando Autoencoder...")
history = autoencoder.fit(
    X, X,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    shuffle=True,
    callbacks=[early_stop],
    verbose=1
)

autoencoder.save(MODEL_DIR)
print(f"Autoencoder salvo em: {MODEL_DIR}")
print(f"Épocas treinadas: {len(history.history['loss'])}")