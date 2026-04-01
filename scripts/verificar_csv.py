import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def verificar(label, path):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  {path}")
    if not path.exists():
        print("  ❌ Arquivo não encontrado")
        return
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"\n  Primeiras linhas:")
    print(df.head().to_string())
    numerics = df.select_dtypes(include=[np.number])
    if not numerics.empty:
        print(f"\n  Estatísticas numéricas:")
        print(numerics.describe().round(4).to_string())
    if "attack_type" in df.columns:
        print(f"\n  Distribuição por tipo de ataque:")
        print(df["attack_type"].value_counts().to_string())
    if "label" in df.columns:
        print(f"\n  Distribuição de labels:")
        print(df["label"].value_counts().to_string())

verificar("Logs pfSense (raw)",    BASE_DIR / "data" / "raw"       / "pfsense_logs.csv")
verificar("Features (processed)",  BASE_DIR / "data" / "processed" / "features.csv")
verificar("Relatório de inferência", BASE_DIR / "reports"          / "infer.csv")