"""Short BF16 batch-size profile for the configured AdamLM architecture."""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch

from adamlm.model import DecoderTransformer, ModelConfig
from adamlm.training import update


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/bpe512-local.json")
parser.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128])
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--output", default="results/gpu-profile.json")
args = parser.parse_args()
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("A BF16 CUDA GPU is required")
cfg = json.loads(Path(args.config).read_text())
rows = []
for batch_size in args.batches:
    torch.cuda.empty_cache()
    model = DecoderTransformer(ModelConfig(**cfg["model"])).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    x = torch.randint(0, cfg["model"]["vocab_size"], (batch_size, cfg["model"]["block_size"]), device="cuda")
    y = torch.randint(0, cfg["model"]["vocab_size"], x.shape, device="cuda")
    try:
        for _ in range(5):
            update(model, optimizer, [(x, y)], "bf16")
        torch.cuda.synchronize()
        samples = []
        stop = False
        def sample_gpu():
            while not stop:
                try:
                    value = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                    samples.append(float(value.splitlines()[0]))
                except Exception:
                    pass
                time.sleep(.2)
        thread = threading.Thread(target=sample_gpu, daemon=True)
        thread.start()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.steps):
            update(model, optimizer, [(x, y)], "bf16")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        stop = True
        thread.join(timeout=1)
        tokens = args.steps * batch_size * cfg["model"]["block_size"]
        rows.append({"batch_size": batch_size, "steps": args.steps, "tokens_per_second": tokens / elapsed,
                     "step_seconds": elapsed / args.steps, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                     "mean_sampled_gpu_utilization_percent": sum(samples) / len(samples) if samples else None})
    except torch.cuda.OutOfMemoryError:
        rows.append({"batch_size": batch_size, "error": "CUDA out of memory"})
    del model, optimizer, x, y
target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
