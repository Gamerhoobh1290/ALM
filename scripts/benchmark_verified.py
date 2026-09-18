"""Synchronized timed-window benchmarks; output is JSON for direct inspection."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from adamlm.model import DecoderTransformer, ModelConfig
from adamlm.training import update

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=15)
args = parser.parse_args()
torch.set_num_threads(4)
cases = [
    ("tiny-fp32", 2, 4, 128, 16, 1, "fp32", 256),
    ("small-fp32", 4, 6, 384, 8, 1, "fp32", 256),
    ("medium-fp32", 6, 8, 512, 4, 1, "fp32", 256),
    ("small-bf16", 4, 6, 384, 8, 1, "bf16", 256),
    ("small-fp16", 4, 6, 384, 8, 1, "fp16", 256),
    ("small-bf16-accum4", 4, 6, 384, 2, 4, "bf16", 256),
    ("small-bf16-context1024", 4, 6, 384, 2, 1, "bf16", 1024),
]
results = []
for name, layers, heads, width, batch, accumulation, precision, context in cases:
    torch.manual_seed(123)
    model = DecoderTransformer(ModelConfig(n_layer=layers, n_head=heads, n_embd=width, block_size=context)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    batches = [(torch.randint(256, (batch, context), device="cuda"), torch.randint(256, (batch, context), device="cuda")) for _ in range(accumulation)]
    for _ in range(10):
        update(model, opt, batches, precision, scaler)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    steps = 0
    while time.perf_counter() - start < args.seconds:
        loss = update(model, opt, batches, precision, scaler)
        torch.cuda.synchronize()
        steps += 1
    elapsed = time.perf_counter() - start
    row = dict(name=name, params=model.parameter_count(), precision=precision, batch=batch,
               accumulation=accumulation, context=context, steps=steps, seconds=elapsed,
               tokens_per_second=steps*batch*accumulation*context/elapsed,
               allocated_mib=torch.cuda.max_memory_allocated()/2**20,
               reserved_mib=torch.cuda.max_memory_reserved()/2**20, loss=loss.item())
    results.append(row)
    print(json.dumps(row), flush=True)
    del model, opt, batches, loss, scaler
    gc.collect()
    torch.cuda.empty_cache()
Path("results").mkdir(exist_ok=True)
Path("results/benchmark.json").write_text(json.dumps(results, indent=2))
