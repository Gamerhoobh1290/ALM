"""Prove the randomly initialized model can memorize a tiny sample."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from adamlm.model import DecoderTransformer, ModelConfig
from adamlm.tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    torch.manual_seed(7)
    tokenizer = ByteTokenizer()
    text = "AdamLM can learn this tiny sentence. " * 8
    ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
    block = 64
    x, y = ids[:, :block], ids[:, 1:block + 1]
    model = DecoderTransformer(ModelConfig(block_size=block, n_layer=2, n_head=4, n_embd=128))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    first = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if first is None:
            first = float(loss.detach())
        loss.backward()
        optimizer.step()
    final = float(loss.detach())
    print(f"tiny_overfit first_loss={first:.4f} final_loss={final:.4f} steps={args.steps}")
    if final >= 0.15:
        raise SystemExit("tiny sample did not overfit enough")


if __name__ == "__main__":
    main()
