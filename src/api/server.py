#!/usr/bin/env python3
"""
API de integração da IA de Cibersegurança
------------------------------------------
Lê reports/infer.json (resumo) e reports/infer.csv (resultados por IP)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parents[2]
REPORT_JSON = BASE_DIR / "reports" / "infer.json"
REPORT_CSV  = BASE_DIR / "reports" / "infer.csv"

app = FastAPI(
    title="Cyber IA Security API",
    description="API para integração do sistema de IA com o painel de monitoramento",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---

def load_summary() -> dict:
    if not REPORT_JSON.exists():
        raise HTTPException(status_code=404, detail="Execute a inferência primeiro (run_inference.py)")
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def load_results() -> pd.DataFrame:
    if not REPORT_CSV.exists():
        raise HTTPException(status_code=404, detail="Arquivo infer.csv não encontrado")
    df = pd.read_csv(REPORT_CSV)
    # Garante que colunas booleanas não quebrem o JSON
    for col in ["iso_anomaly", "ae_anomaly", "final_anomaly"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df

# --- Endpoints ---

@app.get("/")
def root():
    return {
        "message": "Cyber IA Security API v2",
        "endpoints": ["/summary", "/results", "/alerts", "/attacks", "/host/{ip}"]
    }

@app.get("/summary")
def get_summary():
    """Resumo geral: total de eventos, anomalias e breakdown por tipo de ataque."""
    return load_summary()

@app.get("/results")
def get_results(
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0)
):
    """Lista paginada de todos os hosts analisados."""
    df = load_results()
    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": page.to_dict(orient="records")
    }

@app.get("/alerts")
def get_alerts(limit: int = Query(100, ge=1, le=5000)):
    """Apenas os hosts marcados como anomalia."""
    df = load_results()
    alerts = df[df["final_anomaly"] == True].head(limit)
    return {
        "total_alerts": int(df["final_anomaly"].sum()),
        "alerts": alerts.to_dict(orient="records")
    }

@app.get("/attacks")
def get_attacks():
    """Distribuição de alertas por tipo de ataque (para gráficos no dashboard)."""
    df = load_results()
    if "attack_type" not in df.columns:
        raise HTTPException(status_code=500, detail="Coluna attack_type não encontrada no relatório")
    
    breakdown = (
        df[df["final_anomaly"] == True]
        .groupby("attack_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict(orient="records")
    )
    return {"attack_breakdown": breakdown}

@app.get("/host/{ip}")
def get_host(ip: str):
    """Detalhes completos de um IP específico."""
    df = load_results()
    if "src_ip" not in df.columns:
        raise HTTPException(status_code=500, detail="Coluna src_ip não encontrada")
    
    match = df[df["src_ip"] == ip]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"IP {ip} não encontrado no relatório")
    
    return match.iloc[0].to_dict()