"""Bounded telemetry sampling during a manually launched experiment."""
import json
import argparse
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="results/first-run/gpu_samples.json")
args = parser.parse_args()
samples = []
for _ in range(60):
    raw = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
    samples.append(dict(time=time.time(), values=raw))
    time.sleep(1)
Path(args.output).write_text(json.dumps(samples, indent=2))
values = [[float(v) for v in row["values"].split(",")] for row in samples]
print(json.dumps(dict(samples=len(samples), mean_utilization=sum(v[0] for v in values)/len(values), max_utilization=max(v[0] for v in values), max_total_vram_mib=max(v[1] for v in values), max_temperature=max(v[2] for v in values))))
