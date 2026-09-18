"""Short GPU throughput benchmark for candidate model sizes."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from adamlm.model import DecoderTransformer, ModelConfig


SIZES = {
    "tiny": (2, 4, 128, 16),
    "small": (4, 6, 384, 8),
    "medium": (6, 8, 512, 4),
}


def run(name, steps, block_size):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; benchmark requires the GPU")
    layers, heads, embd, batch = SIZES[name]
    config = ModelConfig(block_size=block_size, n_layer=layers, n_head=heads, n_embd=embd)
    model = DecoderTransformer(config).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, 256, (batch, block_size), device="cuda")
    y = torch.randint(0, 256, (batch, block_size), device="cuda")
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = steps * batch * block_size
    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    tps = tokens / elapsed
    print(f"{name}: params={model.parameter_count()/1e6:.2f}M batch={batch} seq={block_size} tokens_per_sec={tps:.1f} peak_allocated_vram_gib={peak:.2f} peak_reserved_vram_gib={reserved:.2f} step_sec={elapsed/steps:.3f} hours_per_100m_tokens={100_000_000/tps/3600:.2f}")


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--block-size", type=int, default=256)
parser.add_argument("--sizes", nargs="+", choices=tuple(SIZES), default=list(SIZES))
args = parser.parse_args()
for size in args.sizes:
    run(size, args.steps, args.block_size)
