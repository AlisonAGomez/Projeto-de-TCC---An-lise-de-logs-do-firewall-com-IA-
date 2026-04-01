# debug_teste_csv.py
import os
import sys
import traceback
import pandas as pd
from pathlib import Path

print(">>> Iniciando debug do pipeline pfSense -> CSV")
print("Working dir:", os.getcwd())
print("Python:", sys.executable)
print()

BASE_DIR     = Path(__file__).resolve().parent
RAW_PATH     = BASE_DIR / "data" / "raw"
PROCESSED    = BASE_DIR / "data" / "processed"
LOG_FILE     = RAW_PATH / "pfsense_logs.csv"
FEATURES_CSV = PROCESSED / "features.csv"

print("Verificando caminhos:")
print(f"  Raw dir:      {RAW_PATH}     -> existe: {RAW_PATH.exists()}")
print(f"  Log pfSense:  {LOG_FILE}  -> existe: {LOG_FILE.exists()}")
print(f"  Processed:    {PROCESSED}    -> existe: {PROCESSED.exists()}")
print()

PROCESSED.mkdir(parents=True, exist_ok=True)

# --- Verifica log pfSense ---
if not LOG_FILE.exists():
    print(f"[erro] Arquivo de log não encontrado: {LOG_FILE}")
    print("       Gere logs simulados com: python generate_pfsense_logs.py")
    sys.exit(1)

try:
    print(f"Lendo log pfSense: {LOG_FILE}")
    df = pd.read_csv(LOG_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Registros lidos: {len(df)}")
    print(f"Colunas:         {list(df.columns)}")
    print()
except Exception as e:
    print("[erro] Falha ao ler log pfSense:", e)
    traceback.print_exc()
    sys.exit(1)

# --- Valida colunas mínimas ---
REQUIRED = {"src_ip", "dst_ip", "dst_port", "protocol", "action"}
missing  = REQUIRED - set(df.columns)
if missing:
    print(f"[aviso] Colunas ausentes: {missing}")
    print(f"        Colunas encontradas: {list(df.columns)}")
else:
    print("[ok] Todas as colunas obrigatórias presentes.")

# --- Amostra de dados ---
print("\nPrimeiras 5 linhas:")
print(df.head().to_string())
print()

# --- Estatísticas básicas ---
print("=== ESTATÍSTICAS ===")
print(f"IPs de origem únicos:    {df['src_ip'].nunique() if 'src_ip' in df.columns else 'N/A'}")
print(f"Portas destino únicas:   {df['dst_port'].nunique() if 'dst_port' in df.columns else 'N/A'}")
if "action" in df.columns:
    print(f"Distribuição de ações:\n{df['action'].value_counts().to_string()}")
print()

# --- Verifica features geradas ---
if FEATURES_CSV.exists():
    df_feat = pd.read_csv(FEATURES_CSV)
    print(f"[ok] Features encontradas: {FEATURES_CSV}")
    print(f"     Shape: {df_feat.shape}")
    print(f"     Colunas: {list(df_feat.columns)}")
else:
    print(f"[aviso] Features ainda não geradas. Rode: python src/features/feature_engineer.py")

print("\n>>> Debug finalizado.")