import pandas as pd
from scapy.all import rdpcap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pcap", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--max", type=int, default=1000)
args = parser.parse_args()

packets = rdpcap(args.pcap)
rows = []

for pkt in packets[:args.max]:
    rows.append({
        "time": pkt.time,
        "src": pkt[0][1].src if hasattr(pkt[0][1], 'src') else "",
        "dst": pkt[0][1].dst if hasattr(pkt[0][1], 'dst') else "",
        "proto": pkt[0][1].name if hasattr(pkt[0][1], 'name') else "",
        "length": len(pkt),
        "info": str(pkt.summary())
    })

df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print("CSV gerado com sucesso!")
