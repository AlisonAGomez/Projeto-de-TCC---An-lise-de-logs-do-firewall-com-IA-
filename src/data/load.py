import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

PFSENSE_COLUMNS = [
    "timestamp", "action", "interface",
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol"
]

def load_pfsense_logs(filename="pfsense_logs.csv"):
    log_file = RAW_PATH / filename

    print(f"Carregando logs: {log_file}")
    df = pd.read_csv(log_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Garante coluna de timestamp como datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Remove linhas sem IP de origem
    df = df.dropna(subset=["src_ip"])

    output_file = PROCESSED_PATH / "logs_clean.csv"
    df.to_csv(output_file, index=False)

    print(f"Logs carregados: {len(df)} registros")
    print(f"Arquivo salvo em: {output_file}")
    return df

if __name__ == "__main__":
    load_pfsense_logs()