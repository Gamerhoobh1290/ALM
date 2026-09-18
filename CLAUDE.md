# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AdamLM is a from-scratch language-model training baseline for Windows 11 + a single RTX 3060 12GB. Everything (BPE tokenizer, transformer, training loop, checkpointing, evaluation, a local web dashboard) is implemented in-house under `src/adamlm/` — no HF `transformers`/training-framework dependency. No VCS is in use in this working directory (see project memory `AdamLM has no VCS`), so there is no commit history or rollback net; be extra careful with destructive file operations and prefer additive/reversible changes.

## Commands

```powershell
cd C:\AdamLM

# Run tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q tests/test_training.py          # single file
.\.venv\Scripts\python.exe -m pytest -q tests/test_training.py::test_x  # single test

# Local web dashboard (primary UI)
.\.venv\Scripts\python.exe -u -m adamlm.web       # http://127.0.0.1:8765/
# or double-click AdamLM.cmd (Tkinter fallback: AdamLM-desktop.cmd)

# Training CLI (production continuation, auto-resumes)
.\start-training.cmd
.\status.cmd
.\stop-training.cmd
# equivalent explicit form:
.\.venv\Scripts\python.exe -m adamlm.bpe_train --config config\bpe512-local.json --resume

# Generation / inference
.\.venv\Scripts\python.exe scripts\generate.py results\bpe512-local-run\checkpoints --prompt "..." --tokens 200
.\.venv\Scripts\python.exe scripts\generate.py results\bpe512-local-run\checkpoints --interactive --chat --tokens 120 --temperature 0.7 --top-k 40

# Evaluation across checkpoints
.\.venv\Scripts\python.exe scripts\evaluate.py <baseline_ckpt> <candidate_ckpts_dir> --output results\checkpoint-comparison.json --device cuda

# Quick baseline/throughput checks
.\.venv\Scripts\python.exe scripts\overfit_tiny.py
.\.venv\Scripts\python.exe scripts\stream_tinystories.py
.\.venv\Scripts\python.exe scripts\benchmark.py --steps 20
```

All commands run through `.venv\Scripts\python.exe` — never a system Python. `pyproject.toml` sets `pythonpath = ["src"]` for pytest, so `adamlm` imports work without installing.

## Architecture

**Config-driven runs.** Every training stage is defined by a JSON file in `config/` (model dims, tokenizer path, dataset source, batch/precision, checkpoint cadence, target tokens, LR schedule). `bpe_train.py` is the CLI entry point that loads a config, resumes or initializes a run directory under `results/<run>/`, and drives the loop in `training.py`. Compatibility between a config and an existing checkpoint (architecture, tokenizer hash, validation snapshot, dataset manifest) is strictly enforced — an incompatible resume fails loudly rather than silently reinitializing.

**Data sources are pluggable but strict.** `data.py` / `local_data.py` / `parquet_data.py` / `dataset_selection.py` implement readers for local TinyStories text, WikiText-103 and FineWeb-Edu Parquet shards, and weighted mixtures. Each reader stores enough state in the checkpoint (byte/row offset, parser version, source file hash, token buffer, tokenizer identity) to resume exactly or refuse to resume when anything upstream changed. `sft_data.py` handles instruction-tuning data (Dolly, DailyDialog) with loss masking over non-assistant tokens.

**Checkpoints are the source of truth for resumability.** `checkpoint.py` defines the atomic save/load format: model weights, optimizer state, full RNG state, dataset cursor, and config-identifying hashes. Nothing about a run's progress is trusted from anywhere else — `versions.py` and the web dashboard read `status.json`/`metrics.jsonl` plus checkpoint contents directly rather than tracking separate state.

**One GPU-session claim, everywhere.** `gpu_session.py` implements a single non-blocking claim file (`results/.gpu-session.json`) that training, inference, and evaluation all check before touching the GPU. A second workload is refused with a clear message instead of contending for VRAM. CLI trainers and the web backend's trainer handoff both go through this same claim; a crashed holder's claim expires by age/lock proof, never by PID probing — don't "fix" contention by adding PID-based checks.

**Web dashboard (`web.py`, largest module) is a thin layer over the same primitives** the CLI uses — it shells out to/reuses `bpe_train`, `gpu_session`, `storage`, and checkpoint compatibility rules rather than reimplementing them. It has four tabs with distinct trust boundaries:
- **Chat** always serves the one approved checkpoint recorded in `results/assistant_default.json` (via `chatstore.py`/`inference.py`) — never a training checkpoint mid-run, never picker-selectable.
- **Research** (`research.py`, `auto_train.py`, `providers.py`) runs a bounded inspect→propose→train→test→decide loop. An optional OpenRouter-compatible provider is used *only* for planning decisions, never for Chat replies; its API key lives in `config/.research-api-key` and is never sent to the browser.
- **Versions** (`versions.py`) manages assistant lineage (approved/previous/candidates/experimental) with manual promotion (reason required) and rollback; comparisons between candidate and approved run with identical prompts/seed/settings on CPU for fairness.
- **Storage** (`storage.py`) enforces the project-wide disk budget (15 GiB project / 2 GiB cache / 5 GiB free-drive floor) checked before every step and before checkpoint writes; a breach pauses training with an explicit `PAUSED_LOW_DISK` marker rather than deleting anything.

**Evaluation is a frozen, versioned suite**, not ad hoc scoring. `config/eval-suite.json` pins 29 held-out prompts with exact scoring rules; `evaluation.py`/`eval_full.py` report raw completions and explicit `answer`/`hallucination`/`abstention` labels rather than a single aggregate accuracy number — preserve that transparency when extending it.

**`gui.py`/`gui_core.py`** are the legacy Tkinter desktop interface (`AdamLM-desktop.cmd`), kept working but not the primary interface — new features land in `web.py` first.

## Working conventions from README/data provenance

- Corpus and dataset files are pinned to exact revisions/hashes (see README) and verified at startup; don't add code paths that skip verification or silently fall back to unverified data.
- New dataset stages start fresh optimizer/LR state from the latest compatible checkpoint and record `parent_checkpoint`/`parent_step`/`exact_dataset_resume: false` — they never overwrite the parent run.
- Only trusted local checkpoints should ever be loaded (PyTorch checkpoints contain serialized Python state) — don't add remote/untrusted checkpoint loading.
