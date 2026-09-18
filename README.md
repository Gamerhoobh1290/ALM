# AdamLM

## Local web dashboard

Double-click `AdamLM.cmd` to start the localhost dashboard. It opens the default browser at `http://127.0.0.1:8765/` and keeps the backend terminal visible. The browser can close without stopping training; if the backend exits, any already-launched trainer may continue and can still be inspected with `status.cmd` or stopped with `stop-training.cmd`.

Exact launch command:

```powershell
cd C:\AdamLM
.\.venv\Scripts\python.exe -u -m adamlm.web
# Then open http://127.0.0.1:8765/ in the browser.
```

The dashboard can resume an unfinished immutable run or create a new continuation stage from a compatible checkpoint. New stage targets are entered as additional tokens; the plan separately shows requested additional tokens, tokens already processed in the selected stage, the effective batch-rounded target, remaining tokens, update count, learning-rate schedule, measured-throughput estimate, and checkpoint storage reserve. Production checkpoints are preferred ahead of smoke-test artifacts, which remain explicitly labeled in the selectors.

The dashboard reads the run's real `status.json`, `metrics.jsonl`, checkpoints, drive space, and `nvidia-smi` telemetry. **Graceful Stop** writes the existing stop request and waits for the trainer to save a complete checkpoint. Closing the app while a trainer it launched is running also requests a graceful stop and keeps the window open until the trainer lock is released. The existing `.cmd` launchers and Python CLIs continue to work unchanged.

The **Chat** tab is the normal experience: one assistant called AdamLM that always answers from the approved checkpoint in `results/assistant_default.json` (currently `auto-assistant-20260917-231947-s3`, step 2024). No checkpoint picker, no training knobs — just the version pill, activity status, and an honest "using X of Y turns" context note (512-token budget). Random seed by default; lock it in Advanced for reproducible tests. Saved conversations (opt-in per chat) and user-submitted corrections live under `results/conversations/` and `results/feedback/`; corrections become training-eligible only after explicit approval in Versions → Feedback, and never train automatically.

The **Research** tab runs an optional bounded inspect → propose → train → test → decide loop while AFK. Set maximum training tokens, overall session duration, and experiment count, then start explicitly. Local-heuristic planning needs no provider; the optional AI planner uses an OpenRouter-compatible API purely for decisions (never for Chat replies — Chat is always the local approved model). Configure base URL, model, and key under the provider box; keys stay in `config/.research-api-key` and are never returned to the browser. Every experiment reuses the existing `bpe_train` CLI, GPU-session exclusion, storage guards, and checkpoint compatibility rules; reports persist under `results/research/` across restarts.

The **Versions** tab shows the real assistant lineage (approved → previous → candidates → experimental branches; foundation and smoke tests separately), fair candidate-vs-approved comparisons (identical prompts, seeds, settings; CPU), scorecards with actual replies, manual promotion with a required reason, one-click rollback to the previous approved pointer, and the preserved Lab with manual checkpoint inference, story completion, and A/B comparison.

Training, inference, and evaluation share one non-blocking GPU-session claim (`results/.gpu-session.json`): a second workload is refused with a clear message instead of competing. CLI trainers (`start-training.cmd`, `bpe_train`) enforce the same claim, including the backend → trainer handoff. A crashed holder's claim expires by age/run-lock proof — never by PID probing.

`AdamLM-desktop.cmd` keeps the former native Tkinter interface available as a fallback.

Current limitations: GPU utilization, VRAM, and temperature show as unavailable when `nvidia-smi` is unavailable; duration limits are checked at optimizer boundaries and safe final saving can extend past the requested time; an ETA is shown only after actual throughput has been measured. Only trusted local checkpoints should be loaded because PyTorch checkpoints contain serialized Python state.

## Current production source: verified local TinyStoriesV2-GPT4

Training now reads local text files. The Dataset Viewer `/rows` API is **not used by the production trainer**, and training works offline once the files are present. The default configuration is `config/bpe512-local.json`; the active continuation directory is `results/bpe512-local-run`. Older configuration examples below are historical; explicitly using the old remote BPE configuration now fails safely instead of making Viewer requests.

Official files are pinned to repository revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`:

| File | Verified bytes | Published SHA-256 |
|---|---:|---|
| TinyStoriesV2-GPT4-train.txt | 2,227,753,162 | `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443` |
| TinyStoriesV2-GPT4-valid.txt | 22,502,601 | `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d` |

The published hashes and sizes came from Hugging Face's Git LFS metadata for that exact revision. Both downloaded files matched. They live under `data/tinystories-v2`; no second corpus copy or full in-memory corpus is created. The sequential parser treats a line containing `<|endoftext|>` as a boundary, decodes UTF-8 strictly, and strips surrounding story whitespace. It stores byte offset, story offset, remaining token buffer, parser version, source hash, and tokenizer identity in every checkpoint. End of file does not silently loop back. Local file verification runs at startup; Status shows `verifying_data` during that check.

## General pretraining, evaluation, and conversation stages

The current TinyStories production checkpoint remains unchanged at `results/bpe512-local-run/checkpoints/step_00012208.pt`. New general-language stages use `config/pretrain-general.json`: 20% TinyStories replay, 35% WikiText-103, and 45% FineWeb-Edu, measured by equal-sized token batches. This replay reduces abrupt forgetting while WikiText and FineWeb add broader language and factual coverage. Change the weights with `--mixture`; they are normalized and stored in every checkpoint.

The frozen evaluation suite in `config/eval-suite.json` has 29 locally authored held-out items. It separately reports grammar/coherence choices, direct general-knowledge and Windows questions, reasoning/instruction tasks, unsupported questions, and raw coherent-generation samples. It stores exact prompts, scoring rules, model/tokenizer hashes, raw completions, and transparent `answer`, `prompt_continuation`, `hallucination`, or `abstention` labels. Correct factual recall is reported as held-out task performance, not proof that a fact was never memorized.

Databricks Dolly 15k is pinned at revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` and SHA-256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`. It contains human-written instruction/response examples and is licensed CC BY-SA 3.0. `scripts/prepare_dolly.py` verifies it, normalizes text, removes eight duplicate or rejected rows, and creates deterministic train, validation, and holdout files. SFT masks user/context tokens from the loss and trains only assistant responses and their boundary. No pretrained model or external inference service is used.

Exact commands:

```powershell
cd C:\AdamLM

# Reproduce the local Dolly preparation (already complete on this machine).
.\.venv\Scripts\python.exe scripts\prepare_dolly.py

# Start or automatically resume general pretraining from the TinyStories checkpoint.
.\start-training.cmd --config config\pretrain-general.json
.\status.cmd --config config\pretrain-general.json
.\stop-training.cmd --config config\pretrain-general.json

# Start SFT after general pretraining. The parent directory selects its latest checkpoint.
.\start-training.cmd --config config\sft-dolly.json --parent-checkpoint results\general-pretrain-run\checkpoints
.\status.cmd --config config\sft-dolly.json
.\stop-training.cmd --config config\sft-dolly.json

# Explicit resume alternatives.
.\.venv\Scripts\python.exe -m adamlm.bpe_train --config config\pretrain-general.json --resume
.\.venv\Scripts\python.exe -m adamlm.bpe_train --config config\sft-dolly.json --resume

# Compare any saved checkpoints on the same held-out suite.
.\.venv\Scripts\python.exe scripts\evaluate.py results\bpe512-local-run\checkpoints\step_00012208.pt results\sft-dolly-run\checkpoints --output results\checkpoint-comparison.json --device cuda

# Ask questions interactively. --chat applies the exact SFT prompt format.
.\.venv\Scripts\python.exe scripts\generate.py results\sft-dolly-run\checkpoints --interactive --chat --tokens 120 --temperature 0.7 --top-k 40
```

The bounded one-epoch SFT benchmark is stored under `results/sft-benchmark-v2`; it is evidence that the stage runs and lowers response loss, not a recommended conversational model. Its loss improved from 5.7559 to 4.0266, but capability accuracy was 6/27 versus the TinyStories baseline's 7/27, and its generated answers remained repetitive hallucinations. General pretraining must happen before a serious SFT run. See `results/checkpoint-comparison.json` for the actual outputs.

## Local WikiText-103 and FineWeb-Edu stages

The extra datasets are optional and are selected explicitly; TinyStories remains the default. AdamLM reads Parquet one row group at a time through `pyarrow`. WikiText rows are fragments, so its reader joins the top-level title and every paragraph/section through the row before the next top-level title, including across row groups and shards. FineWeb rows already represent complete documents and retain one boundary per row. Both sources reject empty, corrupted, abnormally nonlinguistic, highly repetitive, and too-short content. Checkpoints store the manifest hash, parser/filter version, file hashes, split/range, raw row cursor, pending article fragments, token buffer, and tokenizer hash, so changed files or incompatible ranges are rejected.

The currently verified files under `data/extra` are:

| Dataset | Training files | Validation | Verified bytes |
|---|---|---|---:|
| WikiText-103 | 2 Parquet shards, 1,801,350 rows | separate 3,760-row Parquet file | 314,733,787 |
| FineWeb-Edu | 1 Parquet shard, 182,101 rows | final 2,000 training-shard rows reserved separately | 540,632,672 |

The combined local extra-data footprint is 855,366,459 bytes (0.797 GiB). The file SHA-256 values and immutable manifest are recorded in `config/extra-datasets.json`; the WikiText schema is `text`, and FineWeb-Edu's schema is `text`, `id`, `dump`, `url`, `file_path`, `language`, `language_score`, `token_count`, `score`, `int_score`.

Each new dataset stage starts from the latest compatible TinyStories BPE/512 weights, resets the optimizer and learning-rate schedule, starts its dataset cursor at zero, and records `parent_checkpoint`, `parent_step`, and `exact_dataset_resume: false`. It does not overwrite the parent or claim an exact data resume. A stage checkpoint can only resume with the same dataset manifest, tokenizer, architecture, validation snapshot, and training configuration.

Install the Parquet reader in the project environment if needed:

```powershell
cd C:\AdamLM
.\.venv\Scripts\python.exe -m pip install "pyarrow>=20,<24"
```

Launch an explicit stage (the terminal remains visible):

```powershell
.\start-training.cmd --dataset wikitext103 --run-dir results\wikitext103-stage --target-tokens 10000000 --duration 3600
.\start-training.cmd --dataset fineweb_edu --run-dir results\fineweb-edu-stage --target-tokens 10000000 --duration 3600
.\start-training.cmd --dataset mixture --mixture wikitext103=0.7,fineweb_edu=0.3 --run-dir results\mixed-stage --target-tokens 10000000 --duration 3600
```

Resume without repeating the dataset or target (the launcher reads `launcher-config.json`):

```powershell
.\start-training.cmd --run-dir results\wikitext103-stage
.\status.cmd --run-dir results\wikitext103-stage
.\stop-training.cmd --run-dir results\wikitext103-stage
```

Stop is graceful and saves a complete checkpoint at an optimizer boundary. Status distinguishes `initializing`, `verifying_data`, `running`, `saving`, `stopped`, `target_reached`, `disk_pause`, and `error`; it also reports the selected dataset and whether the referenced latest checkpoint exists. No long-term training is started by setup or verification.

**This is an explicit source continuation, not an exact old-data resume.** The GPT-4 file begins with Ben finding a vase; the old source begins with Lily finding a needle. No safe exact cursor translation is claimed. Step 89 is preserved unchanged. The disk also contained newer steps 250 and 325, so the prepared continuation preserves **step 325 / 1,331,200 global tokens**, including model tensors, populated AdamW state, RNG state, tokenizer, architecture, and learning-rate position. Only the data source/cursor and validation source change: new file at byte/story zero, empty token buffer. The record is `results/bpe512-local-run/source-transition.json`, with parent checkpoint hashes and the old dataset state. The original checkpoints remain in `results/bpe512-run/checkpoints`, outside the new run's retention scope.

The official validation file is separate from training. Its first 64 stories form a new frozen snapshot at `data/tinystories-v2/validation-snapshot.json` (32 tracking, 32 holdout). Exact snapshot-story hashes are excluded from training. Previous validation snapshots and holdout results are untouched for comparison; new and old loss figures evaluate different text sets.

Exact PowerShell commands, independent of Codex:

```powershell
cd C:\AdamLM
# Start/resume the prepared local continuation automatically:
.\start-training.cmd
# Run these in another terminal (or double-click):
.\status.cmd
.\stop-training.cmd
# Explicit resume, if preferred:
.\.venv\Scripts\python.exe -m adamlm.bpe_train --config config/bpe512-local.json --resume
# Generate from the latest production checkpoint:
.\.venv\Scripts\python.exe scripts/generate.py results/bpe512-local-run/checkpoints --prompt "Once upon a time, a curious fox" --tokens 200
# Interactive story completion:
.\.venv\Scripts\python.exe scripts/generate.py results/bpe512-local-run/checkpoints --interactive --tokens 200
```

The download is already complete. Only if preparing/recovering files, run:

```powershell
.\.venv\Scripts\python.exe scripts/download_tinystories_v2.py
```

This reuses verified completed files and resumes owned `.part` downloads using byte ranges. HTTP 429/502/503, timeouts, and interrupted transfers receive at most five attempts with exponential backoff/jitter and `Retry-After` handling. A cooldown over five minutes exits with the required delay instead of waiting indefinitely. Completed files are promoted atomically only after size/hash verification. Corrupt files are preserved and reported for inspection, not silently replaced. These network concerns apply to initial preparation only; the training loop never redownloads data. There is no service or scheduled task.

Storage remains **15 GiB total / 5 GiB free-drive floor**; measured project use after preparation and smoke-test artifacts was approximately **7.93 GiB**. The 2 GiB transient cache cap remains unchanged: these permanent corpus files are counted in the project budget, not duplicated in cache. Only the verified 248,731,111-byte disposable Parquet cache from the abandoned approach was removed; it is recoverable by downloading its pinned source again. No checkpoint, tokenizer, result, or unrelated file was deleted. NumPy 2.5.3 was installed only in `.venv`; PyTorch/NumPy interoperability passes and the old warning is gone. PyTorch itself was not reinstalled.

Verification used only `results/local-delivery-smoke`, a copy of the continuation checkpoint. A short GPU run advanced 325 → 413 (360,448 new tokens), and resume reproduced validation loss 3.952326874 exactly. Further resume/graceful-stop checks and local cursor tests passed. Production remains prepared at step 325, with no full training session started. Focused delivery tests cover local checkpoint/buffer recovery with network forbidden, cache reuse, interrupted byte-range downloads, simulated 429/502 recovery, and refusal to promote a checksum failure.

## Standalone Windows training

Codex is not needed while training. Double-click `start-training.cmd`, or run the commands below in PowerShell. The training terminal stays visible and waits for a key after exit. No service or scheduled task is installed.

```powershell
cd C:\AdamLM
.\start-training.cmd
.\status.cmd
.\stop-training.cmd
```

Start automatically resumes the latest checkpoint in `results/bpe512-run`, or initializes a fresh model if none exists. Stop and Status should be run in a **second terminal**, or opened by double-click while the training window runs. Stop requests termination at the next optimizer-update boundary, saves the complete model, optimizer, RNG and dataset cursor/buffer, then exits. Wait until Status reports `Trainer active: False` and `State: stopped` before closing the training window. A pending network request or checkpoint write can delay stopping. Do not close the window or use Ctrl+C for a graceful stop: those can lose work since the previous checkpoint. If disk safety prevents saving, Status reports the error and the previous atomic checkpoint remains available.

Configure the **total** training target for a fresh run (not additional tokens):

```powershell
.\start-training.cmd --target-tokens 10000000
```

The launcher remembers that target; subsequent `.\start-training.cmd` calls resume it without repeating the option. The default fresh target is 50 million BPE tokens from `config/bpe512.json`. Targets round up to a whole batch. Changing the target after a run has checkpoints changes the learning-rate schedule and is intentionally rejected, as are changed architecture, tokenizer or validation hashes. An incompatible or corrupt latest checkpoint causes an error: existing files are preserved and the launcher never silently resets training or falls back to an older checkpoint. Use a separate run directory to change configuration:

```powershell
.\start-training.cmd --run-dir results/my-new-run --target-tokens 10000000
.\status.cmd --run-dir results/my-new-run
.\stop-training.cmd --run-dir results/my-new-run
# Resume that same run (target remembered):
.\start-training.cmd --run-dir results/my-new-run
```

By default each invocation runs for up to one hour plus final evaluation/save time; use `--max-seconds 300` to limit a session to five minutes. Progress is written approximately every second to the run's `status.json`; validation loss is the latest evaluated loss, not necessarily from the displayed current step. Status reports checkpoint step separately from live progress, and uses the trainer lock to detect stale status after a crash. Only one trainer may use a run directory at a time. HTTP 429/network failures save at a safe boundary and exit; start again after the external limit clears. All streaming and 15 GiB storage safeguards remain enabled.

Verification on the separate `results/launcher-smoke` lineage: visible start, graceful stop at step 63, automatic resume with the remembered target, and second graceful stop at step 72 (294,912 tokens). The final checkpoint includes populated AdamW state, all RNG states, and 16,578 buffered dataset tokens. A changed-target attempt was rejected and its checkpoint SHA-256 stayed unchanged. Production training was not started.

## BPE/512 ready for a fresh run

A custom 4,096-token byte-level BPE tokenizer and the 512-context configuration are now prepared. The longer run has **not** started. See [BPE measurements and exact launch/resume/inference commands](results/bpe512-report.md). `config/bpe512.json` uses a distinct checkpoint lineage; legacy byte checkpoints remain usable through the same generation CLI. The dataset viewer rate-limited the short resume check, so sustained streaming performance is not guaranteed; a data stop saves a checkpoint for manual resume after the limit clears.

## Measured language experiment (session 2)

Use the project interpreter for each command:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_verified.py --seconds 15
.\.venv\Scripts\python.exe scripts\experiment.py --tokens 2097152 --seconds 3600 --precision bf16
.\.venv\Scripts\python.exe scripts\experiment.py --tokens 2097152 --seconds 3600 --precision bf16 --resume
.\.venv\Scripts\python.exe scripts\generate.py results\first-run\checkpoints\step_00001024.pt
```

The experiment is a separate bounded entry point; legacy `adamlm.train` and original checkpoints remain available. A new run refuses to overwrite an existing run with checkpoints. Resume requires the same model, precision, batch, accumulation, learning rate, and validation snapshot. The token target and elapsed-time limit are cumulative across saved invocations. Sixty seconds are reserved for closing evaluation and checkpointing; network calls have a 20-second timeout. Use a distinct `--run-dir` for another experiment.

Validation uses 64 stories from the dataset's `validation` split: first 32 for tracking, next 32 only for final holdout evaluation. A small local snapshot and its SHA-256 hash make the evaluation repeatable. Exact story matches are filtered from training, but near-duplicate contamination is not audited. The remote viewer API is not pinned to an immutable dataset revision; a future dataset update can change rows beyond the saved cursor. This is a pilot data source, not yet a publication-quality corpus manifest.

The default experiment retains the byte tokenizer and 256-token context. Every loss is natural-log cross-entropy per byte; it must not be compared numerically to a future BPE-token loss. Samples reload the saved checkpoint and use temperature 0.8, top-k 40, and seed 42. Only load trusted local checkpoints because they include Python optimizer/RNG state.

Results are stored in `results/benchmark.json`, `results/first-run/metrics.jsonl`, and `results/first-run/summary.json`. CUDA timing synchronizes both before and after work. Compute benchmarks include backward, clipping, and AdamW; the experiment's end-to-end rate additionally includes network reads, tokenization, storage scans, validation, checkpoints, and final sampling. GPU allocation excludes driver/desktop memory, which is reported separately by `nvidia-smi`.

Checkpoint writes reserve space before serialization and keep three files. Drive free space is checked each step; project/cache sizes are scanned every 30 seconds and before checkpoint writes. If disk safety fails, the bounded experiment pauses by stopping and leaves a marker; free space and use `--resume` from the latest completed checkpoint. This can replay steps since that checkpoint. It never deletes baseline checkpoints or downloads the full corpus. BF16 needs no loss scaler; FP16 saves/restores its scaler. Gradient accumulation uses `--batch` as microbatch size and averages equal-sized microbatches.

AdamLM is a from-scratch, local language-model training baseline for Windows 11 and an RTX 3060 12 GB. The next serious run can use the preserved BPE/512 model on an explicitly selected local TinyStories, WikiText-103, FineWeb-Edu, or weighted-mixture stage; the legacy byte-tokenizer lineage remains available for comparison.

## Environment

This project is isolated in `.venv`; the repository does not install packages system-wide. Python 3.12 is used because it has broad support for current PyTorch CUDA wheels. Create and install it from PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e . pytest
```

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## Baseline checks

```powershell
.\.venv\Scripts\python.exe scripts\overfit_tiny.py
.\.venv\Scripts\python.exe scripts\stream_tinystories.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\benchmark.py --steps 20
```

The benchmark is intentionally short. It measures actual forward/backward/update throughput on the active GPU and reports peak allocated VRAM. It does not start a long training run.

## Resumable training

`python -m adamlm.train --steps 10` trains only ten steps by default. Each saved checkpoint contains model weights, AdamW state, step, Python/PyTorch/CUDA RNG states, and the TinyStories cursor plus buffered token IDs. Writes are atomic and only the configured number of recent checkpoints is retained. Use `--resume --steps N` to continue to step N.

`config/default.json` sets a 15 GiB project budget, a 2 GiB `.cache` limit, a 5 GiB minimum free-disk safety floor, and three retained checkpoints. The training loop checks these before every step. If a limit is crossed it writes `PAUSED_LOW_DISK` and waits until the condition is safe; it then resumes and removes the marker. There is no automatic Windows scheduler or service yet.

## Architecture recommendation

The byte/256 baseline is retained for checkpoint compatibility. Session 2 measured its language quality and recommends a training-only custom BPE vocabulary and longer context for the next serious run, with a new checkpoint lineage. See [the experiment report](results/experiment-report.md) for actual measurements, proposed settings, and limitations. The 15 GiB budget includes the virtual environment, baseline checkpoints, and new run artifacts.
