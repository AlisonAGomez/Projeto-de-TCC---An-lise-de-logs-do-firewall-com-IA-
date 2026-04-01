# 🔐 CyberIA - Análise de Logs de Firewall com IA

Este projeto tem como objetivo detectar anomalias e possíveis ataques em logs de firewall (pfSense) utilizando técnicas de Inteligência Artificial.

---

## 🚀 Execução

```bash
python scripts/generate_pfsense_logs.py
python src/models/train_detection.py
python src/models/run_inference.py

O sistema segue um pipeline de detecção de anomalias:

Geração de ataques (Kali Linux)
Coleta de logs (pfSense)
Processamento de dados
Detecção com IA
Classificação de ataques
Exposição via API

📂 Estrutura do Projeto
data/ → dados brutos e processados
models/ → modelos treinados
scripts/ → scripts auxiliares
src/ → código principal
reports/ → resultados da análise

🎯 Objetivo

Automatizar a detecção de comportamentos suspeitos em redes, auxiliando na resposta a incidentes de segurança.

🛡️ Tecnologias
Python
Scikit-Learn
Pandas
pfSense
Kali Linux