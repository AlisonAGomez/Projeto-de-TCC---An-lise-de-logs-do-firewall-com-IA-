import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[2]
REPORT_CSV  = BASE_DIR / "reports" / "infer.csv"

def avaliar(csv_path=REPORT_CSV):
    df = pd.read_csv(csv_path)

    # Coluna de predição — aceita os dois nomes
    if "final_anomaly" in df.columns:
        y_pred = df["final_anomaly"].astype(int)
    elif "combined_flag" in df.columns:
        y_pred = df["combined_flag"].astype(int)
    else:
        print("[erro] Coluna final_anomaly ou combined_flag não encontrada.")
        return

    # --- Avaliação supervisionada (se tiver label) ---
    if "label" in df.columns:
        y_true = df["label"].astype(int)

        print("\n=== MATRIZ DE CONFUSÃO ===")
        print(confusion_matrix(y_true, y_pred))

        print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
        print(classification_report(y_true, y_pred,
                                    target_names=["Normal", "Anomalia"],
                                    digits=4))

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        print("=== RESUMO ===")
        print(f"Ataques reais no dataset:            {int((y_true == 1).sum())}")
        print(f"Ataques detectados pela IA:          {int((y_pred == 1).sum())}")
        print(f"Verdadeiros Positivos (TP):          {tp}")
        print(f"Falsos Positivos (FP):               {fp}")
        print(f"Falsos Negativos (FN — não detectados): {fn}")
    else:
        print("[info] Coluna 'label' não encontrada — avaliação não supervisionada.")
        print(f"Total de eventos:   {len(df)}")
        print(f"Anomalias detectadas: {int(y_pred.sum())} ({y_pred.mean()*100:.2f}%)")

    # --- Breakdown por tipo de ataque (sempre exibe se disponível) ---
    if "attack_type" in df.columns:
        print("\n=== DISTRIBUIÇÃO POR TIPO DE ATAQUE ===")
        breakdown = (
            df[y_pred == 1]
            .groupby("attack_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print(breakdown.to_string(index=False))

    # --- Scores de anomalia (resumo estatístico) ---
    print("\n=== SCORES DE ANOMALIA ===")
    for col in ["isof_score", "auto_mse", "ae_mse"]:
        if col in df.columns:
            print(f"{col}: min={df[col].min():.4f} | "
                  f"mean={df[col].mean():.4f} | "
                  f"95p={np.percentile(df[col], 95):.4f} | "
                  f"max={df[col].max():.4f}")

if __name__ == "__main__":
    avaliar()