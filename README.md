# 🔐 CyberIA - Análise de Logs de Firewall com IA

Este projeto tem como objetivo detectar anomalias e possíveis ataques em logs de firewall (pfSense) utilizando técnicas de Inteligência Artificial.

---

## 🎯 Objetivo

Automatizar a detecção de comportamentos suspeitos em redes, auxiliando na identificação de possíveis incidentes de segurança.

---

## 🚀 Execução

```bash
python scripts/generate_pfsense_logs.py
python src/models/train_detection.py
python src/models/run_inference.py

🧠 Arquitetura do Sistema

O sistema segue um pipeline de detecção de anomalias:

Geração de ataques (Kali Linux)
Coleta de logs (pfSense)
Processamento de dados
Detecção com IA
Classificação de ataques
Exposição via API

📂 Estrutura do Projeto
data/      → dados brutos (raw) e processados (processed)  
models/    → modelos treinados  
scripts/   → scripts auxiliares  
src/       → código principal do sistema  
reports/   → resultados da análise  

🛡️ Tecnologias Utilizadas
Python
Scikit-Learn
Pandas
pfSense
Kali Linux
📊 Datasets Utilizados

O projeto utiliza datasets públicos amplamente reconhecidos na área de segurança:

CICIDS 2017 (Canadian Institute for Cybersecurity)
UNSW-NB15 (University of New South Wales)

Esses datasets contêm tráfego legítimo e diferentes tipos de ataques, sendo utilizados para treinamento e validação dos modelos de detecção.

⚠️ Por serem arquivos grandes, não estão incluídos neste repositório.

📥 Como obter os dados

Os datasets podem ser baixados nos links oficiais:

CICIDS 2017: https://www.unb.ca/cic/datasets/ids-2017.html
UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Após o download, coloque os arquivos na pasta:

data/raw/

💡 Observações
O projeto também permite a utilização de logs reais provenientes do pfSense.
Os dados podem ser gerados sinteticamente utilizando scripts incluídos no projeto.
A arquitetura foi pensada para permitir reprodutibilidade e escalabilidade do pipeline de Machine Learning.