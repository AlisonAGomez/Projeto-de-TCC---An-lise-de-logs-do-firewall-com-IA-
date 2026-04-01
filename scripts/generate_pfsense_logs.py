# generate_pfsense_logs.py
"""
Gera logs de firewall pfSense simulados para testes do pipeline.
Inclui tráfego normal + Port Scan + Brute Force + DDoS sintéticos.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# 🔹 Diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 🔹 Nome mais profissional
OUT_FILE = OUT_DIR / "pfsense_logs_synthetic.csv"

# 🔹 Reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_TIME  = datetime(2024, 1, 1, 0, 0, 0)
PROTOCOLS  = ["tcp", "udp", "icmp"]
SERVICES   = {
    80: "http", 443: "https", 22: "ssh", 21: "ftp",
    3306: "mysql", 53: "dns", 8080: "http-alt", 445: "smb"
}

def random_ip(prefix="192.168.1."):
    return prefix + str(random.randint(2, 254))

def gen_normal(n=3000):
    rows = []
    for _ in range(n):
        ts = BASE_TIME + timedelta(seconds=random.randint(0, 86400))
        dst_port = random.choice(list(SERVICES.keys()))
        rows.append({
            "timestamp": ts,
            "action":    random.choices(["pass", "block"], weights=[0.85, 0.15])[0],
            "interface": random.choice(["em0", "em1"]),
            "src_ip":    random_ip("192.168.1."),
            "dst_ip":    random_ip("10.0.0."),
            "src_port":  random.randint(1024, 65535),
            "dst_port":  dst_port,
            "protocol":  random.choice(PROTOCOLS),
            "label":     0
        })
    return rows

def gen_port_scan(n=30):
    rows = []
    for _ in range(n):
        src = random_ip("10.10.10.")
        ts  = BASE_TIME + timedelta(seconds=random.randint(0, 3600))
        for port in random.sample(range(1, 10000), k=random.randint(50, 200)):
            rows.append({
                "timestamp": ts,
                "action":    "block",
                "interface": "em0",
                "src_ip":    src,
                "dst_ip":    random_ip("192.168.1."),
                "src_port":  random.randint(1024, 65535),
                "dst_port":  port,
                "protocol":  "tcp",
                "label":     1
            })
            ts += timedelta(milliseconds=random.randint(10, 100))
    return rows

def gen_brute_force(n=20):
    rows = []
    for _ in range(n):
        src      = random_ip("172.16.0.")
        dst_port = random.choice([22, 21, 3306])
        ts       = BASE_TIME + timedelta(seconds=random.randint(0, 3600))
        for _ in range(random.randint(80, 200)):
            rows.append({
                "timestamp": ts,
                "action":    "block",
                "interface": "em0",
                "src_ip":    src,
                "dst_ip":    random_ip("192.168.1."),
                "src_port":  random.randint(1024, 65535),
                "dst_port":  dst_port,
                "protocol":  "tcp",
                "label":     1
            })
            ts += timedelta(seconds=random.uniform(0.5, 2))
    return rows

def gen_ddos(n=10):
    rows = []
    for _ in range(n):
        src = random_ip("203.0.113.")
        ts  = BASE_TIME + timedelta(seconds=random.randint(0, 3600))
        for _ in range(random.randint(200, 500)):
            rows.append({
                "timestamp": ts,
                "action":    random.choice(["pass", "block"]),
                "interface": "em0",
                "src_ip":    src,
                "dst_ip":    "192.168.1.100",
                "src_port":  random.randint(1024, 65535),
                "dst_port":  random.choice([80, 443]),
                "protocol":  random.choice(["tcp", "udp"]),
                "label":     1
            })
            ts += timedelta(milliseconds=random.randint(1, 50))
    return rows

# 🔥 Gerar dados
rows = gen_normal() + gen_port_scan() + gen_brute_force() + gen_ddos()
random.shuffle(rows)

df = pd.DataFrame(rows)

# 🔹 Conversão correta de timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 🔹 Ordenação temporal (IMPORTANTE)
df = df.sort_values("timestamp")

# 💾 Salvar
df.to_csv(OUT_FILE, index=False)

# 📊 Logs
print(f"[ok] Logs gerados: {OUT_FILE}")
print(f"     Total: {len(df)} registros")
print(f"     Normal: {(df['label']==0).sum()} | Ataques: {(df['label']==1).sum()}")
print(f"     Distribuição de ações:\n{df['action'].value_counts().to_string()}")