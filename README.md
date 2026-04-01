# 🔐 CyberIA - Análise de Logs de Firewall com IA

Sistema de detecção de anomalias em logs de firewall (pfSense) utilizando técnicas de Machine Learning e Deep Learning.

---

## 🎯 Objetivo

Detectar comportamentos suspeitos e possíveis ataques em redes corporativas de forma automatizada, auxiliando na identificação precoce de incidentes de segurança.

---

## 🧠 Arquitetura do Sistema

O projeto segue um pipeline completo de análise de segurança:

1. Geração de tráfego/ataques (Kali Linux)
2. Coleta de logs (pfSense)
3. Processamento e engenharia de features
4. Detecção de anomalias (IA)
5. Classificação de ataques
6. Geração de relatórios

---

## ⚙️ Pipeline de Execução

```bash
# 1. Gerar logs simulados
python scripts/generate_pfsense_logs.py

# 2. Gerar features a partir dos logs
python src/features/build_features.py

# 3. Treinar os modelos
python src/models/train_detection.py

# 4. Rodar inferência (detecção)
python src/features/feature_engineer.py

📊 Modelos Utilizados

O sistema combina múltiplas abordagens de detecção:

🌲 Isolation Forest (detecção de anomalias)
🧠 Autoencoder (Deep Learning)
📈 Engenharia de Features baseada em comportamento de rede
📂 Estrutura do Projeto
data/
 ├── raw/         # Logs brutos (pfSense ou datasets)
 └── processed/   # Features geradas

models/           # Modelos treinados
scripts/          # Scripts auxiliares (geração de dados)
src/
 ├── features/    # Engenharia de features
 └── models/      # Treinamento e inferência

reports/          # Resultados e análises

📊 Exemplo de Saída
{
    "total_events": 311,
    "anomalies_detected": 71,
    "anomaly_rate_percent": 22.83,
    "attack_breakdown": {
        "Port Scan": 28,
        "DDoS": 18,
        "Suspeito": 16,
        "Brute Force": 9
    }
}

🛡️ Tecnologias Utilizadas
Python
Pandas
Scikit-Learn
TensorFlow / Keras
pfSense
Kali Linux
📊 Datasets Utilizados

O projeto pode utilizar datasets públicos amplamente reconhecidos:

CICIDS 2017 (Canadian Institute for Cybersecurity)
UNSW-NB15 (University of New South Wales)

⚠️ Os arquivos não estão incluídos no repositório devido ao tamanho.

📥 Como obter os dados

Baixe os datasets nos links oficiais:

CICIDS 2017: https://www.unb.ca/cic/datasets/ids-2017.html
UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Após o download:

data/raw/
💡 Observações
O sistema aceita logs reais do pfSense
É possível gerar dados sintéticos para testes
Arquitetura projetada para reprodutibilidade e escalabilidade
🚀 Possíveis Evoluções
Dashboard em tempo real
Integração com SIEM
API REST para análise contínua
Alertas via Telegram/Email
Deploy em ambiente cloud
👨‍💻 Autor

Projeto desenvolvido como Trabalho de Conclusão de Curso (TCC).


---

# 💥 O QUE EU MELHOREI (IMPORTANTE)

### ✔️ Pipeline mais claro
Antes tava implícito → agora está **executável**

### ✔️ Separação correta:
- build_features  
- treino  
- inferência  

👉 isso é padrão de ML real

---

### ✔️ Adicionei:
- exemplo de saída (MUITO importante pra banca)
- seção de modelos
- seção de evolução (isso impressiona muito)

---

### ✔️ Corrigi fluxo real do seu código
Antes:

train → run_inference


Agora:

logs → features → treino → inferência


👉 isso mostra maturidade técnica

---

# 🔥 Resultado final

Seu projeto agora parece:

```txt
Projeto acadêmico + nível profissional + pronto pra portfólio
