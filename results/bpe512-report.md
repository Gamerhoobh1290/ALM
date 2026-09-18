# BPE/512 preparation — 2026-09-16

The byte-model baseline, original results and checkpoints, legacy training commands, and generation CLI are preserved. Only tokenizer fitting, a 30-second GPU benchmark, a short resume check, and focused tests were run. The longer training run has **not** started. Its checkpoint directory is distinct from the benchmark directory and will initialize fresh random weights and AdamW state.

## Frozen tokenizer

`tokenizers/bpe4096/tokenizer.json` contains exactly 4,096 tokens, including the 256 ByteLevel alphabet symbols and `<|endofstory|>` (ID 0). BPE byte fallback is enabled; the complete byte alphabet guarantees encodability of UTF-8 text without an unknown token. Encoding does not normalize or add spaces. Story boundaries are inserted explicitly; a literal special-token spelling in ordinary input remains ordinary text. Generation stops at the boundary token.

Training used TinyStories **train rows 0–4,999 only**, 5,000 stories / 4,187,330 UTF-8 bytes, fetched in 100-story pages. Exact matches to the saved validation stories were excluded (none occurred). Validation text was not used to learn merges. The bounded corpus was held in memory and was not saved as a dataset download. Source split, row range, corpus SHA-256, and tokenizer SHA-256 are recorded in `tokenizers/bpe4096/manifest.json`. The tokenizer SHA-256 is `76ac1863baadb1067f53589809022ea9dfa82eb822b7ab63d9b89994056a94e4`.

## Same held-out text comparison

The untouched baseline validation snapshot contains 64 stories / 44,029 bytes. All 64 round-trip exactly through BPE. Focused tests also cover multilingual text, emoji, control characters, all byte alphabet symbols, and literal boundary spelling.

| Measurement | Original byte tokenizer | New BPE |
|---|---:|---:|
| Text tokens for identical held-out text | 44,029 | 11,120 |
| Tokens including explicit story boundaries | no explicit boundary | 11,184 |
| UTF-8 bytes per text token | 1 | 3.959 |
| Context length | 256 tokens | 512 tokens |
| Estimated represented text per context | 256 bytes | 2,016 bytes |
| Whole held-out stories fitting individually | 0/64 | 64/64 |

BPE reduces text-token count by 74.7%. Combining compression and doubled context gives **7.87×** the approximate text capacity, including boundary overhead (~403 versus ~51 words at this snapshot's word lengths). Context byte capacity is an estimate from corpus-average compression, not a fixed guarantee. Whole-story fit counts are measured. Both tokenizers retain complete textual coverage; this measures representation efficiency, not language-model quality.

A one-second CPU encoding microbenchmark on the held-out text measured ~882k BPE tokens/sec / 3.49 MB text/sec, versus ~335 MB/sec for the trivial byte conversion. BPE is slower to encode but remains much faster than the observed model/data pipeline. It does not require a larger dataset cache.

## Exact BF16 configuration benchmark

4 layers, width 384, 6 heads, tied embeddings, 4,096 vocabulary, 512 learned positions: **8,868,096 parameters**. BF16 autocast, FP32 parameters/AdamW state, microbatch 8, accumulation 1, gradient clipping at 1.0. Warmup/decay settings are the same as the prepared longer configuration.

The benchmark performed 238 updates / **974,848 BPE prediction tokens**. The configured 30-second training window plus final validation, checkpoint, and sample took **32.47 seconds**. End-to-end throughput was **30,023 BPE tokens/sec**, including setup, network reads, tokenization, CUDA transfers, optimizer updates, validation, storage checks, checkpointing, and sampling. CUDA was synchronized at timing boundaries and after each update. Compute-only throughput was 149,567 tokens/sec; recorded data preparation took 17.85 seconds versus 6.52 seconds of compute.

Peak PyTorch allocation: **562.8 MiB**; peak reservation: **592 MiB**. These exclude CUDA driver/context and other Windows GPU applications. The measured BPE rate roughly corresponds to 118k bytes/sec using held-out compression; the previous byte run measured 47k bytes/sec end to end, but they are not controlled quality-equivalent comparisons.

Raw first benchmark results: `results/bpe512-benchmark/benchmark-initial.json`. It is preserved separately from the resume-check summary. Initial/final validation cross-entropy was 8.417/4.159 nats per BPE token. This is not numerically comparable to byte-model cross-entropy, and this brief run does not establish quality superiority.

The real resume check restored step 238 and exactly reproduced validation loss 4.158614565. It made four further updates, then encountered **HTTP 429** from the dataset viewer. The stream cursor was rolled back for the failed batch and model/optimizer/cursor saved at step 242. A data stop now reports exit status 2; resume after the service limit clears. There is no automatic retry loop hammering the API. The throughput above is therefore a short-window measurement, **not a sustained throughput guarantee**. The viewer remains mutable and not revision-pinned; reliable unattended long runs would need a more durable data source or an agreed bounded dataset cache.

## Prepared longer run

`config/bpe512.json` specifies a 50M BPE-token target (rounded up to a full 4,096-token update), the above architecture, and a one-hour per-invocation training window. Final evaluation/save can add a few seconds. Resume continues the same total token target and schedule; it does not reset warmup.

- Linear warmup over 250 updates to 3e-4; cosine decay to 3e-5 at the target.
- Validation and atomic checkpoint every 250 updates; text sample every 1,000 updates and at session end. Final evaluation also uses the separate latter 32 validation stories as holdout. No padded targets enter the loss.
- Checkpoints contain model/optimizer, Python/PyTorch/CUDA RNG, stream cursor and buffered IDs, exact config, tokenizer hash, validation hash, and elapsed counters. Schedule position derives from restored update count and frozen configuration.
- Resume rejects different tokenizers/configs/validation snapshots and all legacy byte checkpoints. Inference checks the tokenizer hash too. A run-directory lock prevents concurrent trainers overwriting the same lineage.
- 15 GiB project budget, 2 GiB cache limit, 5 GiB drive-free floor, three retained checkpoints, space reservation before writes, cheap free-space checks each step and full scans every 30 seconds. Disk pauses and Ctrl+C fall back to the last completed atomic checkpoint; uncheckpointed work may replay.

Project size after preparation: approximately **4.96 GiB**. A full BPE AdamW checkpoint is roughly 102 MiB; three production checkpoints plus one temporary replacement need about 408 MiB. This leaves substantial room under 15 GiB. No full corpus cache was created. Only the project virtual environment gained the `tokenizers` dependency and its dependencies; pip disk caching was disabled.

## Commands (PowerShell, from C:\AdamLM)

Launch the fresh longer run when ready:

```powershell
cd C:\AdamLM
.\.venv\Scripts\python.exe -m adamlm.bpe_train --config config/bpe512.json
```

Resume the latest production checkpoint after a session limit, disk pause, or resolved network limit:

```powershell
.\.venv\Scripts\python.exe -m adamlm.bpe_train --config config/bpe512.json --resume
```

Generate from the latest resulting production checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts/generate.py results/bpe512-run/checkpoints --prompt "Once upon a time, a curious fox" --tokens 200
```

Interactive story continuation (independent prompts, not instruction-tuned chat; `/quit` exits):

```powershell
.\.venv\Scripts\python.exe scripts/generate.py results/bpe512-run/checkpoints --interactive --tokens 200
```

For an immediate smoke-test model before production training, substitute `results/bpe512-benchmark/checkpoints` in those inference commands. That model is intentionally undertrained. For example, it currently continues: `Once upon a time, he put the dog! He wanted to go home with time she saw a new one big, the`.

The original baseline still runs with:

```powershell
.\.venv\Scripts\python.exe scripts/generate.py results/first-run/checkpoints/step_00001024.pt --prompt "Once upon a time, a little girl" --tokens 100
```

Focused verification: `python -m pytest -q tests/test_bpe.py tests/test_training.py` — 10 tests pass. Both old and BPE standalone inference were exercised. Tokenizer fitting refuses to overwrite an existing frozen tokenizer. Production training was not launched.
