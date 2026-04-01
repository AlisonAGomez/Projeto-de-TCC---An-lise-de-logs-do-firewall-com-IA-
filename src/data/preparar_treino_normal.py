import pandas as pd
from pathlib import Path

# Caminhos
entrada = Path("Data/Processed/features.csv")
saida = Path("Data/Processed/unsw_nb15_treino_normal.csv")

# Ler dataset
df = pd.read_csv(entrada)

# Manter apenas tráfego normal
df_normal = df[df["label"] == 0]

# Remover coluna label (modelo não deve ver isso)
df_normal = df_normal.drop(columns=["label"])

df_normal.to_csv(saida, index=False)

print(f"[ok] Dataset de treino normal criado: {saida}")
print("Total de amostras normais:", len(df_normal))
