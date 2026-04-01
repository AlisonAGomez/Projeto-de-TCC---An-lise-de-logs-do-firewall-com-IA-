import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT  = BASE_DIR / "data" / "raw" / "pfsense_logs_synthetic.csv"
OUTPUT = BASE_DIR / "data" / "processed" / "features.csv"

def main():
    df = pd.read_csv(INPUT)

    # 🔹 timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")

    # 🔹 agrupar por src_ip (comportamento por IP)
    features = df.groupby("src_ip").agg({
        "dst_port": "nunique",
        "src_port": "count",
        "protocol": "nunique",
        "action": lambda x: (x == "block").sum()
    }).rename(columns={
        "dst_port": "unique_dst_ports",
        "src_port": "total_connections",
        "protocol": "unique_protocols",
        "action": "blocks"
    })

    # 🔹 taxa de bloqueio
    features["block_rate"] = features["blocks"] / features["total_connections"]

    # 🔹 label (se existir)
    if "label" in df.columns:
        labels = df.groupby("src_ip")["label"].max()
        features["label"] = labels

    features = features.reset_index()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT, index=False)

    print(f"[ok] Features geradas: {OUTPUT}")
    print(features.head())

if __name__ == "__main__":
    main()