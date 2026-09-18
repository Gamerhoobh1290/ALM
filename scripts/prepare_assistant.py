"""Curate the mixed DailyDialog + Dolly SFT diet for the assistant stage.

Reads only the existing local manifests (no download, no retraining data):
  data/dailydialog/train.jsonl (+ validation/holdout)
  data/dolly/train.jsonl        (+ validation/holdout)

Curation rules (explicit and reportable):
  - DailyDialog train: drop rows whose raw encoding exceeds the 512-token
    block (575 rows, 1.6%); truncated dialogue contexts would train the
    model on cut-off histories.
  - Dolly train: keep all rows that encode (0 failures). Truncation
    concentrates in summarization (52%), information_extraction (45%) and
    closed_qa (32%); dropping them would gut exactly the passage-grounded
    categories, and encode_example already preserves the answer span.
  - Rows are shuffled per source with a fixed seed, then interleaved by a
    token-deficit round robin so the file holds ~70% DailyDialog / 30%
    Dolly *by sampled training tokens* along its whole length (Dolly rows
    average 1.43x longer, so example counts differ by design).
  - Provenance is preserved: every row keeps its original
    instruction/context/response/category fields untouched.

Outputs (new files only; existing data is never modified):
  data/sft_assistant/train.jsonl + validation.jsonl + holdout.jsonl
  data/sft_assistant/processed-manifest.json (adamlm-response-sft-v1)
"""
import json
import os
import random
from pathlib import Path

from adamlm.downloads import sha256

ROOT = Path("data/sft_assistant")
DD = Path("data/dailydialog")
DOLLY = Path("data/dolly")
BLOCK = 512
SEED = 20260918
DD_SHARE = 0.7

# Measured with the frozen BPE via sft_data.encode_example (block 512):
#   DailyDialog train: 35,465 rows / 5.55M tokens (1.6% truncated, dropped)
#   Dolly train:       13,511 rows / 3.03M tokens (15.4% truncated, kept)
# A 1M-token run therefore covers ~0.13 epoch DD + ~0.10 epoch Dolly:
# no row repeats; repetition risk is zero for this experiment size.


def load_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def interleave(pools, share_first, token_len, seed):
    """Deterministic token-deficit round robin over [pool0, pool1]."""
    rng = random.Random(seed)
    for pool in pools:
        rng.shuffle(pool)
    order = [[0, 0] for _ in pools]  # per-pool [cursor]
    debt = [0.0, 0.0]
    shares = [share_first, 1.0 - share_first]
    out = []
    remaining = [len(p) for p in pools]
    while any(r > 0 for r in remaining):
        candidates = [i for i in (0, 1) if remaining[i] > 0]
        # Largest deficit between deserved and received token share wins.
        pick = max(candidates, key=lambda i: debt[i])
        row = pools[pick][order[pick][0]]
        order[pick][0] += 1
        remaining[pick] -= 1
        length = token_len(row)
        for i in (0, 1):
            debt[i] += shares[i] * length
        debt[pick] -= length
        out.append(row)
    return out


def main():
    import sys
    sys.path.insert(0, "src")
    from adamlm.bpe import BPETokenizer
    from adamlm.sft_data import encode_example

    tok = BPETokenizer("tokenizers/bpe4096/tokenizer.json")

    def encoded_len(row):
        inputs, _ = encode_example(row, tok, BLOCK)
        return len(inputs)

    def curated(source_rows, drop_truncated):
        kept = []
        for row in source_rows:
            try:
                inputs, targets = encode_example(row, tok, BLOCK)
            except ValueError:
                continue
            ins = row["instruction"].strip()
            ctx = row.get("context", "").strip()
            prompt = f"User:\n{ins}\n" + (f"\nContext:\n{ctx}\n" if ctx else "") + "\nAssistant:\n"
            raw = len(tok.encode(prompt)) + len(tok.encode(row["response"].strip())) + 1
            if drop_truncated and raw > BLOCK + 1:
                continue
            kept.append(row)
        return kept

    dd_train = curated(load_rows(DD / "train.jsonl"), drop_truncated=True)
    dolly_train = curated(load_rows(DOLLY / "train.jsonl"), drop_truncated=False)
    dd_tokens = sum(encoded_len(r) for r in dd_train)
    dolly_tokens = sum(encoded_len(r) for r in dolly_train)
    print(f"curated pools: DD {len(dd_train)} rows / {dd_tokens} tokens; "
          f"Dolly {len(dolly_train)} rows / {dolly_tokens} tokens")

    mixed_train = interleave([dd_train, dolly_train], DD_SHARE, encoded_len, SEED)
    mixed_tokens = sum(encoded_len(r) for r in mixed_train)
    dolly_frac = sum(encoded_len(r) for r in mixed_train if r.get("category") != "dailydialog") / mixed_tokens
    print(f"mixed train: {len(mixed_train)} rows / {mixed_tokens} tokens; dolly token share = {dolly_frac:.4f}")

    mixed_val = interleave([load_rows(DD / "validation.jsonl"), load_rows(DOLLY / "validation.jsonl")],
                           DD_SHARE, encoded_len, SEED + 1)
    mixed_hold = interleave([load_rows(DD / "holdout.jsonl"), load_rows(DOLLY / "holdout.jsonl")],
                            DD_SHARE, encoded_len, SEED + 2)

    ROOT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (("train", mixed_train), ("validation", mixed_val), ("holdout", mixed_hold)):
        target, tmp = ROOT / f"{name}.jsonl", ROOT / f"{name}.tmp"
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp, target)
        files[name] = {"path": str(target).replace("\\", "/"), "size": target.stat().st_size,
                       "sha256": sha256(target), "examples": len(rows)}
    manifest = {
        "format": "adamlm-response-sft-v1",
        "dataset": "dailydialog+dolly-assistant-v1",
        "revision": f"curated-mix-{SEED}",
        "license": "Derived locally from the DailyDialog and Dolly preparations; research use, no redistribution.",
        "sources": {
            "dailydialog": {"manifest": "data/dailydialog/processed-manifest.json",
                            "rows": len(dd_train), "tokens": dd_tokens, "dropped_truncated": 35465 - len(dd_train)},
            "dolly": {"manifest": "data/dolly/processed-manifest.json",
                      "rows": len(dolly_train), "tokens": dolly_tokens, "dropped_truncated": 0},
        },
        "mix_rule": f"per-source seeded shuffle ({SEED}) + token-deficit round robin at "
                    f"{DD_SHARE:.0%} DailyDialog / {1 - DD_SHARE:.0%} Dolly by encoded training tokens; "
                    f"row fields byte-identical to sources (category preserves provenance)",
        "row_rule": "shared adamlm-response-sft-v1 template with response-only loss masking (unchanged)",
        "files": files,
    }
    (ROOT / "processed-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: {"examples": v["examples"], "size": v["size"]} for k, v in files.items()}, indent=2))


if __name__ == "__main__":
    main()
