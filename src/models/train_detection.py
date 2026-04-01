#!/usr/bin/env python3
"""
Treina IsolationForest + Autoencoder a partir de features de logs pfSense.
Salva scaler, modelos e thresholds em models/
"""
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

def load_features(features_csv):
    df = pd.read_csv(features_csv)
    # Separa label se existir (para avaliação)
    y = None
    if "label" in df.columns:
        y = df["label"].astype(int)
        df = df.drop(columns=["label"])
    if "final_anomaly" in df.columns:
        if y is None:
            y = df["final_anomaly"].astype(int)
        df = df.drop(columns=["final_anomaly"])

    X = df.select_dtypes(include=[np.number]).fillna(0)
    return X, y

def train_isolation(X_train, X_all, y, out_path, scaler, contamination=0.05, n_estimators=200):
    X_scaled = scaler.transform(X_train)
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_scaled)

    scores = -clf.score_samples(X_scaled)
    thresh = float(np.percentile(scores, 90))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "threshold": thresh}, out_path)
    print(f"[ok] IsolationForest salvo em {out_path} (threshold={thresh:.6f})")

    # Avaliação rápida se tiver labels
    if y is not None:
        X_all_scaled = scaler.transform(X_all)
        preds = clf.predict(X_all_scaled)
        y_pred = (preds == -1).astype(int)
        print("\n--- IsolationForest: avaliação no dataset completo ---")
        print(classification_report(y, y_pred, target_names=["Normal", "Anomalia"], digits=4))

    return thresh

def build_autoencoder(n_features, latent=16):
    inp = tf.keras.Input(shape=(n_features,))
    x   = tf.keras.layers.Dense(128, activation="relu")(inp)
    x   = tf.keras.layers.Dense(64,  activation="relu")(x)
    z   = tf.keras.layers.Dense(latent, activation="relu")(x)
    x   = tf.keras.layers.Dense(64,  activation="relu")(z)
    x   = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(n_features, activation=None)(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

def train_autoencoder(X_train, out_dir, scaler, epochs=50, batch_size=64, latent=16):
    X_scaled = scaler.transform(X_train).astype("float32")
    n = X_scaled.shape[1]

    model = build_autoencoder(n, latent=latent)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1
    )

    recon = model.predict(X_scaled, verbose=0)
    mse   = np.mean((X_scaled - recon) ** 2, axis=1)
    thresh = float(np.percentile(mse, 90))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save(out_dir)

    report = {
        "threshold": thresh,
        "last_loss": float(history.history["loss"][-1]),
        "epochs_trained": len(history.history["loss"])
    }
    with open(Path(out_dir) / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[ok] Autoencoder salvo em {out_dir} (threshold={thresh:.6f})")
    print(f"     Épocas treinadas: {report['epochs_trained']}")
    return thresh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",      required=True,                              help="CSV de features (por src_ip)")
    ap.add_argument("--out_isof",      default=str(BASE_DIR / "models" / "isolation_forest.joblib"))
    ap.add_argument("--out_auto",      default=str(BASE_DIR / "models" / "autoencoder"))
    ap.add_argument("--out_scaler",    default=str(BASE_DIR / "models" / "scaler.joblib"))
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--n_estimators",  type=int,   default=200)
    ap.add_argument("--epochs",        type=int,   default=50)
    ap.add_argument("--batch",         type=int,   default=64)
    ap.add_argument("--latent",        type=int,   default=16)
    args = ap.parse_args()

    print(f"Carregando features: {args.features}")
    X, y = load_features(args.features)
    print(f"Total: {len(X)} registros | Features: {X.shape[1]}")

    # Treina scaler com tráfego normal (ou tudo, se sem label)
    X_train = X[y == 0] if y is not None else X
    print(f"Treinando com {len(X_train)} registros normais")

    scaler = StandardScaler()
    scaler.fit(X_train)
    job