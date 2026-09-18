"""Short, resumable training loop; intentionally no long-running default."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .checkpoint import CheckpointManager
from .data import TinyStoriesStream
from .model import DecoderTransformer, ModelConfig
from .storage import StorageBudget


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(config_path="config/default.json", steps=10, batch_size=8, resume=False, root="."):
    root = Path(root).resolve()
    config = load_config(root / config_path)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecoderTransformer(ModelConfig(**config["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    data_config = dict(config["data"])
    data_config["timeout"] = data_config.pop("request_timeout_seconds", 30)
    stream = TinyStoriesStream(**data_config)
    manager = CheckpointManager(root / "checkpoints", config["storage"]["checkpoint_keep"])
    guard = StorageBudget(root, config["storage"]["project_budget_gb"], config["storage"]["cache_limit_gb"], config["storage"]["minimum_free_disk_gb"])
    start = 0
    if resume and manager.latest():
        start = manager.load(manager.latest(), model, optimizer, stream)
    model.train()
    for step in range(start + 1, steps + 1):
        guard.wait_until_safe(config["storage"]["disk_poll_seconds"])
        x, y = stream.next_batch(batch_size, model.config.block_size)
        input_ids = torch.tensor(x, dtype=torch.long, device=device)
        targets = torch.tensor(y, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == steps:
            loss_value = float(loss.detach())
            path = manager.save(step, model, optimizer, stream.state_dict(), {"loss": loss_value, "device": device})
            print(f"step={step} loss={loss_value:.4f} checkpoint={path.name} tokens={stream.stats.tokens_consumed}")
    return model, stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()
    run(args.config, args.steps, args.batch_size, args.resume)


if __name__ == "__main__":
    main()
