import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

files = {
    "Logs pfSense (raw)":   BASE_DIR / "data" / "raw"       / "pfsense_logs.csv",
    "Features (processed)": BASE_DIR / "data" / "processed" / "features.csv",
    "Inferência (report)":  BASE_DIR / "reports"            / "infer.csv",
}

for label, path in files.items():
    print(f"\n{'='*50}")
    print(f"📄 {label}")
    print(f"   Caminho: {path}")
    if not path.exists():
        print("   ❌ Arquivo não encontrado")
        continue
    try:
        df = pd.read_csv(path)
        print(f"   ✅ Carregado com sucesso")
        print(f"   Linhas: {len(df)} | Colunas: {df.shape[1]}")
        print(f"   Colunas: {list(df.columns)}")
        print(f"\n   Prévia:")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"   ⚠️ Erro: {e}")