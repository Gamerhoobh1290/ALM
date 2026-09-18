"""Bounded CPU encoding throughput on the frozen held-out text (no fitting)."""
import json
import time
from pathlib import Path
from adamlm.bpe import BPETokenizer
from adamlm.tokenizer import ByteTokenizer

texts = json.loads(Path("results/first-run/validation.json").read_text(encoding="utf-8"))["texts"]
rows = []
for name, tokenizer in [("byte", ByteTokenizer()), ("bpe4096", BPETokenizer("tokenizers/bpe4096/tokenizer.json"))]:
    start = time.perf_counter()
    rounds = tokens = 0
    while time.perf_counter()-start < 1:
        tokens += sum(len(tokenizer.encode(t)) for t in texts)
        rounds += 1
    elapsed = time.perf_counter()-start
    rows.append(dict(tokenizer=name, seconds=elapsed, rounds=rounds, tokens_per_second=tokens/elapsed,
                     utf8_bytes_per_second=rounds*sum(len(t.encode()) for t in texts)/elapsed))
Path("results/bpe512-benchmark/tokenizer-speed.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows))
