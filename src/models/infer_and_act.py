#!/usr/bin/env python3
"""
Inferência + classificação de ataques + relatório + (opcional) bloqueio.
Por padrão roda em --dry (não executa bloqueios reais).
"""
import os
import json
import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[2]

# --- Loaders ---

def load_isof(path):
    obj = joblib.load(path)
    clf    = obj.get("model", obj) if isinstance(obj, dict) else obj
    thresh = obj.get("threshold", None) if isinstance(obj, dict) else None
    return clf, thresh

def load_auto(path):
    model  = tf.keras.models.load_model(path)
    report = Path(path) / "report.json"
    thresh = None
    if report.exists():
        with open(report) as f:
            thresh = json.load(f).get("threshold")
    return model, thresh

# --- Scores ---

def compute_isof_score(clf, X):
    return -clf.score_samples(X)

def compute_auto_mse(model, X):
    recon = model.predict(X, verbose=0)
    mse   = np.mean((X - recon) ** 2, axis=1)
    return mse, recon

def explain_top_features(X_row, recon_row, columns, topk=3):
    errors = np.abs(X_row - recon_row)
    idx    = np.argsort(-errors)[:topk]
    return [(columns[i], float(errors[i])) for i in idx]

# --- Classificação de ataque ---

def classificar_ataque(row, feature_cols):
    """
    Regras heurísticas baseadas nas features do feature_engineer.py.
    Retorna: Port Scan | Brute Force | DDoS | Suspeito
    """
    def get(col, default=0):
        return float(row.get(col, default)) if col in feature_cols else default

    port_scan_score   = get("port_scan_score")
    brute_force_score = get("brute_force_score")
    max_conn          = get("max_conn_per_window")

    if port_scan_score > 0.5:
        return "Port Scan"
    if brute_force_score > 50:
        return "Brute Force"
    if max_conn > 100:
        return "DDoS"
    return "Suspeito"

# --- Bloqueio ---

def attempt_block(ip, dry=True):
    cmd = f"iptables -A INPUT -s {ip} -j DROP"
    if dry:
        return {"cmd": cmd, "executed": False, "note": "dry-run"}
    if platform.system().lower() != "linux":
        return {"cmd": cmd, "executed": False, "note": "bloqueio só suportado em Linux"}
    try:
        res = subprocess.run(["sudo"] + cmd.split(), capture_output=True, text=True)
        return {
            "cmd": cmd,
            "executed": res.returncode == 0,
            "stdout": res.stdout,
            "stderr": res.stderr
        }
    except Exception as e:
        return {"cmd": cmd, "executed": False, "error": str(e)}

# --- Main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--isof",     default=str(BASE_DIR / "models" / "isolation_forest.joblib"))
    ap.add_argument("--auto",     default=str(BASE_DIR / "models" / "autoencoder"))
    ap.add_argument("--scaler",   default=str(BASE_DIR / "models" / "scaler.joblib"))
    ap.add_argument("--out",      default=str(BASE_DIR / "reports" / "infer.json"))
    ap.add_argument("--outcsv",   default=str(BASE_DIR / "reports" / "infer.csv"))
    ap.add_argument("--dry",      action="store_true", default=True)
    ap.add_argument("--block",    action="store_true",  help="executa bloqueio real (apenas lab, Linux)")
    ap.add_argument("--topk",     type=int, default=3)
    args = ap.parse_args()

    if args.block and args.dry:
        print("[warn] --block ignorado pois --dry está ativo. Use --no-dry para bloqueio real.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Carrega features
    df = pd.read_csv(args.features)
    if "src_ip" not in df.columns and "src" in df.columns:
        df = df.rename(columns={"src": "src_ip"})
    if "src_ip" not in df.columns:
        print("[error] CSV precisa ter coluna 'src_ip'. Colunas:", df.columns.tolist())
        sys.exit(1)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X_raw = df[numeric_cols].fillna(0).values.astype("float32")

    # Aplica scaler
    scaler = joblib.load(args.scaler)
    X = scaler.transform(X_raw)
    print(f"[ok] Scaler aplicado. Shape: {X.shape}")

    # Carrega modelos
    isof_clf, isof_thresh = load_isof(args.isof)
    print(f"[ok] IsolationForest carregado. threshold={isof_thresh}")

    auto_model, auto_thresh = load_auto(args.auto) if Path(args.auto).exists() else (None, None)
    if auto_model:
        print(f"[ok] Autoencoder carregado. threshold={auto_thresh}")
    else:
        print("[warn] Autoencoder não encontrado — usando só IsolationForest")

    # Scores
    isof_scores = compute_isof_score(isof_clf, X)
    auto_mse, auto_recon = compute_auto_mse(auto_model, X) if auto_model else (np.zeros(len(X)), np.zeros_like(X))

    # Thresholds fallback
    if isof_thresh is None:
        isof_thresh = float(np.percentile(isof_scores, 95))
        print(f"[info] isof threshold via percentil 95: {isof_thresh:.6f}")
    if auto_thresh is None:
        auto_thresh = float(np.percentile(auto_mse, 95))
        print(f"[info] auto threshold via percentil 95: {auto_thresh:.6f}")

    isof_flag    = (isof_scores >= isof_thresh).astype(int)
    auto_flag    = (auto_mse   >= auto_thresh).astype(int)
    combined     = ((isof_flag == 1) | (auto_flag == 1)).astype(int)

    results      = []
    rows_for_csv = []

    for i, row in df.iterrows():
        src       = row["src_ip"]
        is_anomaly = bool(combined[i])

        attack_type = classificar_ataque(row, numeric_cols) if is_anomaly else "Normal"

        rec = {
            "src_ip":        src,
            "isof_score":    float(isof_scores[i]),
            "isof_flag":     int(isof_flag[i]),
            "auto_mse":      float(auto_mse[i]),
            "auto_flag":     int(auto_flag[i]),
            "combined_flag": int(combined[i]),
            "final_anomaly": is_anomaly,
            "attack_type":   attack_type
        }

        if auto_model is not None:
            rec["auto_top_features"] = explain_top_features(X[i], auto_recon[i], numeric_cols, topk=args.topk)

        # Ação de bloqueio
        if is_anomaly:
            rec["action"] = attempt_block(src, dry=not (args.block and not args.dry))
        else:
            rec["action"] = {"cmd": None, "executed": False, "note": "no action"}

        results.append(rec)

        rows_for_csv.append({
            "src_ip":          src,
            "isof_score":      rec["isof_score"],
            "isof_flag":       rec["isof_flag"],
            "auto_mse":        rec["auto_mse"],
            "auto_flag":       rec["auto_flag"],
            "combined_flag":   rec["combined_flag"],
            "final_anomaly":   rec["final_anomaly"],
            "attack_type":     rec["attack_type"],
            "action_cmd":      rec["action"].get("cmd"),
            "action_executed": rec["action"].get("executed", False),
            "action_note":     rec["action"].get("note", "")
        })

    # Breakdown por tipo de ataque
    attack_df       = pd.DataFrame(rows_for_csv)
    attack_breakdown = (
        attack_df[attack_df["final_anomaly"]]
        .groupby("attack_type")
        .size()
        .to_dict()
    )

    summary = {
        "total_events":          len(df),
        "anomalies_detected":    int(np.sum(combined)),
        "anomaly_rate_percent":  round(float(np.mean(combined)) * 100, 2),
        "isof_threshold":        float(isof_thresh),
        "auto_threshold":        float(auto_thresh),
        "attack_breakdown":      attack_breakdown
    }

    out_data = {"summary": summary, "results": results}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    attack_df.to_csv(args.outcsv, index=False)

    print(f"\n[ok] JSON: {args.out}")
    print(f"[ok] CSV:  {args.outcsv}")
    print("Resumo:", json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()