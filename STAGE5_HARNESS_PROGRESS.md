# Stage-5 / Long-Training Harness — Progress Record

Task: fix Stage-5 dataset exhaustion, long unattended training, research
evidence use, inference-harness audit. Continues work Codex left unfinished
at ~13:17 on 2026-09-18.

No training launched, no provider calls, no promotion, no pointer change,
no data deleted. No git repo (stays that way).

## Established by inspection (verified, not assumed)

### Root cause of the Stage-5 failure — CONFIRMED by arithmetic

Plan `auto-assistant-20260918-115609`, 10 stages.

- One SFT pass = 8,291,299 source tokens (manifest `curated-mix-20260918`:
  dd 5,258,299 + dolly 3,033,000).
- One optimizer step = 4,096 tokens (`batch_size * accumulation * block_size`,
  `config/sft-assistant.json`).
- Batchable per pass = floor(8,291,299 / 4,096) * 4,096 = **8,290,304**.
  Permanent, structural per-pass deficit = **995 tokens** (the tail cannot
  form a complete batch, so it is never trainable).
- The OLD planner set every stage target to the *source* count 8,291,299 and
  then **carried each stage's shortfall forward into the next stage's target**,
  treating a structural deficit as a recoverable tail.

Observed targets vs actual (from the plan file):

| stage | target    | actual    | cumulative shortfall |
|-------|-----------|-----------|----------------------|
| 1     | 8,291,299 | 8,290,304 |   995                |
| 2     | 8,292,294 | 8,290,304 | 1,990                |
| 3     | 8,293,289 | 8,290,304 | 2,985                |
| 4     | 8,294,284 | 8,290,304 | 3,980                |
| 5     | 8,295,279 | 8,290,304 | **4,975**            |

`_expected_full_pass_data_stop` accepts an end-of-data stop only when the
shortfall is **< one batch (4,096)**. The carry grows by 995 per stage, so it
crosses 4,096 at stage 5 — the failure was **arithmetically inevitable at
stage 5**, not caused by data, disk, or the cursor.

=> Raising MAX_EPOCHS, adding storage, or resetting the cursor would all have
been wrong, exactly as the task warned.

## Work already completed by Codex (verified present in code)

- `allocate()` rewritten: batch-aligned targets, `batchable_epoch_tokens` +
  `discarded_tail_tokens` reported, `pass_policy` (`unique_once` / `repeat`)
  and explicit `stage_count`; the 3-pass ceiling is gone; impossible plans
  raise in preflight (`auto_train.py:210-248`).
- Runtime shortfall carry-forward removed (no `target_tokens` mutation).
- Long-session states wired in `_run_stage`: `done` / `waiting`
  (disk_pause, interrupted) / `stopped` / `regression` / `time-exceeded` /
  `failed`, with `recovery_checkpoint` recorded (`auto_train.py:689-732`).
- Validation policy + best-candidate tracking: `stage_compare_every`,
  `regression_patience`, `best_candidate` (`auto_train.py:418, 739-830`).
- Checkpoint retention hooks: `checkpoint.preserve` (outside rolling
  retention) + approved/milestone exemption (`bpe_train.py:46`).
- Inference baseline captured: `results/harness-inference-before.json`.

## Fixes applied in this session

1. **`auto_train.py:923` — live hygiene bug.** The U+FFFD literal in
   `_hygiene_ok` had degraded to a plain ASCII space (`if " " in reply:`), so
   **every reply containing a space was flagged "replacement character"** and
   `compare_evals` could never promote or report real signal. The same literal
   is intact in `scripts/eval_assistant.py:92`, which proves corruption rather
   than intent. Rewritten as the escape `"�"` so an encoding round-trip
   cannot silently degrade it again. (4 failing tests.)

2. **`allocate()` — same over-claim bug surviving in the `repeat` path.**
   A request expressed in source-epoch units (N x 8,291,299) rounded *up* to a
   batch boundary, exceeding the N x 8,290,304 that N passes can actually
   supply, and spilled into a phantom extra stage (5 passes requested -> 6
   stages of 6,909,952). Now capped at real capacity:
   `passes_spanned * usable_epoch`. This also subsumes the old `unique_once`
   special case. (1 failing test.)

Suite: **143 passed** (was 5 failed / 138 passed).

## Part 4 finding — inference harness is mechanically CORRECT

`results/harness-inference-before.json` (checkpoint
`auto-assistant-20260917-231947-s3/checkpoints/step_00002024.pt`,
tokens=60 temp=0.8 top_k=40 seed=42):

- Template renders `User:\n<text>\n\nAssistant:\n` correctly.
- Reply extraction is exact (reply == raw_output minus prompt, no leakage).
- Truncation accounting present and honest (kept/total/trimmed).

Raw "before" output for "hi":
`the House, Yellowstone, Williamsburg, Texas, and Lake Megarte are the best
friends of all time\n\nIn April 2015, USADCE`

=> The word salad is a **model-capability limit** (8.87M params, ~8M training
tokens), **not** a harness defect. No hardcoded greeting may be used to hide
this. Any claim of conversational improvement must come from training, and
must be shown with raw outputs from the same checkpoint and settings.

## Remaining / NOT done

- [ ] Research does not use prior failed experiments when proposing the next
      one (`research.py` has only a comment at :1172; no `failed_experiments`
      / `already_tried` logic). Genuine gap — repeat-proposal risk is real.
- [ ] Reproduce Stage 5 end-to-end on a small synthetic dataset, incl. the
      pass-4 -> pass-5 transition.
- [ ] Tests for: exact interrupted-stage resume, completed-stage
      non-repetition, storage interruption, checkpoint preservation.
- [ ] Verify checkpoint retention policy actually runs and protects approved /
      completed-stage / milestone checkpoints.
- [ ] "After" inference capture + the required behavioral checks.

## Live run 2026-09-18 13:37 — `auto-assistant-20260918-133731`

Started at user request: 12 stages x 8,290,304 tokens (99.48M), pass_policy
`repeat`, 2h deadline, stage_compare_every 1, regression_patience 2, manual
promotion. Fixes visibly correct in the live plan: all 12 stage targets
identical (no carry-forward); `effective 99,483,648 = 12 * 8,290,304`
(requested 99,495,588 rounded DOWN, no phantom 13th stage); preflight refused
14 and 13 stages on the storage budget before starting.

**Outcome: auto-paused at `regression` after 204,800 tokens (~1 min).**
train_loss 0.265 vs validation_loss 3.873 (best 2.7339 at step 0) — the parent
has memorized this corpus. GPU claim released, trainer exited cleanly, best and
latest checkpoints preserved, nothing promoted, stages 2-12 left `pending`.
This is the early-stopping machinery working as designed, and it is hard
evidence that further repeat passes over the 8.29M mix cannot help.

Next productive step: process `data/extra/2023-04-12_oasst_ready.trees.jsonl.gz`
into the SFT manifest (splits, hashes, tokenizer verification) so there is
genuinely new unique data.

## Dashboard single-instance fix (`web.py`)

Real bug: `ThreadingHTTPServer` inherits `allow_reuse_address = 1`, so on
Windows SO_REUSEADDR let a second launch bind port 8765 while the first was
still serving — two dashboards, undefined routing, and two Auto Train
supervisor sets competing for the same plans and GPU claim.

Fixed with `_SingleInstanceServer` (`allow_reuse_address = False`) plus a
pre-bind probe of `/api/overview`; a duplicate launch now prints where the
running dashboard is and exits 0, so `AdamLM.cmd` still opens the browser to
the existing instance instead of starting a rival server. Bind-race fallback
catches EADDRINUSE/EACCES. Verified live against the running dashboard.
Note: `python.exe` appearing twice per dashboard is normal — the venv launcher
plus its redirector child. One listener = one instance.

Second half of the same requirement: the server guard stops a second *backend*,
but `AdamLM.cmd` always runs `start "" "http://127.0.0.1:8765/"`, so a repeat
launch still opened a second *browser tab* — two studios polling and driving
one backend. Fixed in the frontend with a BroadcastChannel owner election
(`adamlm.studio.v1`, `app.js`): a new tab asks `who`, and any live owner
answers `here` within 300 ms, which parks the new tab behind a `.tab-lock`
overlay with its polling timer never started. A `Use this tab instead` button
posts `takeover`, which makes the previous owner clear its interval and show
the overlay. No BroadcastChannel support degrades to a working studio rather
than a blank page. (Browsers cannot focus another tab programmatically, so
explicit takeover is the correct affordance.)

Assets are read per request with `Cache-Control: no-store`, so this went live
without restarting the backend — verified over HTTP against the running one.

## Recovery facts for the failed plan (do not auto-resume)

- Stages 1-4 completed and are intact; stage 5 trained 8,290,304 tokens and
  saved before stopping.
- Last recoverable checkpoint of the failed plan:
  `results/auto-assistant-20260918-115609-s5/checkpoints/step_00002024.pt`
  (step 2024, val loss 2.7339, holdout 2.9169).
- Approved pointer is UNCHANGED:
  `results/auto-assistant-20260917-231947-s3/checkpoints/step_00002024.pt`.
