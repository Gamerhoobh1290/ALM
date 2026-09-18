# AdamLM 2.0 Upgrade Plan

Status: **IMPLEMENTED 2026-09-18 (all phases; decisions applied: strict
promotion, archive-not-delete, text-only local Chat, manual promotion, no
git, bounded API default 25). Full suite 135 passed. Restart the dashboard
backend to serve the new code (a pre-upgrade backend was still running at
implementation time).** No training started, no provider requests sent, no
checkpoint promoted, no pointer changed, no data modified or deleted.
No git repository exists; none initialized.

## 0. Ground truth established by inspection (2026-09-18)

### 0.1 Approved model and lineage (confirmed)

- Approved pointer: `results/assistant_default.json` →
  `results/auto-assistant-20260917-231947-s3/checkpoints/step_00002024.pt`
  (step 2024, promoted 2026-09-17, plan `auto-assistant-20260917-231947`).
  Previous: `results/auto-assistant-20260917-230901-s1/.../step_00002024.pt`.
- Single-pointer rule is real: `src/adamlm/versions.py:255-274`
  (`resolve_approved_checkpoint`), Chat always uses it
  (`src/adamlm/web.py:929-956`, `_start_chat`). Good — preserve.
- Registry exists: `src/adamlm/versions.py:114-224` (`build_registry`,
  `results/versions.json`) with states approved / previous-approved /
  candidate / experimental / foundation / general / smoke, plus
  `promote_candidate` (requires reason, optional eval-win gate) and
  `rollback_to_previous` (pointer swap only). Good — preserve, declutter UI.
- Clutter source: `discover_runs` (`src/adamlm/gui_core.py:206-219`) indexes
  **every** `results/*/` dir with a launcher/status/checkpoint. `results/`
  currently holds ~14 `auto-assistant-*-sN` stage dirs, 6+ `*.auto.json`
  plans, 9+ `research-*` sessions, plus `web-stage-*`, `web-ui-*`,
  `gui-stage-*`, `launcher-smoke`, `*-smoke`, `*-benchmark` dirs. The registry
  sorts but does not hide/archive — hence the cluttered Versions page.

### 0.2 The 24.87M-token / 3×8.29M-token run (confirmed, with one new bug found)

- Latest plan `results/auto-assistant-20260918-012952.auto.json`: requested
  100M, effective 24.87M = 3 full passes of the 8.29M mixed file, stages
  s1/s2/s3 each 8,290,304 tokens, all `done`. Promotion held correctly:
  `instr 0.3127 → 0.3011` regressed (held), `conv 0.0233 → 0.1281` improved.
- Manual full comparison `results/evals/compare-489100e9ce12.json` confirms:
  suite accuracy `0.222 → 0.185` (Δ −0.037, beyond the 0.02 tolerance in
  `src/adamlm/eval_full.py:32`), grammar `0.75 →` lower, knowledge/reasoning
  still 0. Holding was the right call.
- Mechanism (all in `src/adamlm/auto_train.py`):
  - `allocate()` (`:164-229`): `MAX_EPOCHS = 3`, `BASE_DD_SHARE = 0.70`,
    epoch = `dd_tokens + dolly_tokens` from `data/sft_assistant/
    processed-manifest.json` (5,258,299 + 3,033,000 = 8,291,299; rev
    `curated-mix-20260918`, 48,401 train examples). Cap + repetition labeling
    are honest. **Genuine planner constraint:** fixed mix file, fixed 70/30,
    tilt at most −10pp/+5pp clamped to [50/50, 85/15], max 3 passes, one
    deadline for all stages.
  - `preflight()` (`:266-334`) is pure computation + honest dry-run via
    `gui_core.build_plan`. Good — keep, add preview confirm.
  - `Supervisor.stage_command()` (`:484-500`) hardcodes
    `dataset="dailydialog"`, `MIX_CONFIG`. Stage session label
    (`:566`) says `"dailydialog+dolly"` — the command and the label disagree
    in wording though both resolve to the same mix config in practice.
    Rename for clarity; verify, don't assume breakage.
  - **CONFIRMED BUG — skill tilt is dead code:** `read_skill_profile()`
    (`:150-161`) reads `evaluation["instr_f1"]` / `evaluation["conv_f1"]`,
    but every real pointer (`results/assistant_default.json`,
    `results/auto-assistant-*.auto.json` `eval` blocks) stores
    `instr_f1_old / instr_f1_new / conv_f1_old / conv_f1_new`. Lookup always
    misses → returns `None` → `allocate()` always takes the
    "no evaluated skill profile yet" branch (`:192-193`). The ±tilt has
    likely **never** engaged. Fix the key mapping before any further
    training (Phase 1).
- Data-repeat note: `Supervisor._run_stage` (`:605-631`) absorbs full-pass
  tail shortfalls (< one 4096 batch) into the next stage. Honest, but it
  means repeated identical passes look "productive" in token counts while
  eval shows regression. The plan must make repeat-pass cost explicit.

### 0.3 Research agent (confirmed)

- Loop: `src/adamlm/research.py` — `inspect_project()` (`:411-450`),
  `local_heuristic_proposal()` (`:517-563`), `ai_propose()` (`:579-638`),
  `validate_proposal()` (`:463-514`), session thread `start_session()`
  (`:1119-...`), `stop_session()` (`:1191-...`). State in
  `results/research/<id>.json` + `.report.json/.md`. Survives restarts. Good.
- **Genuine constraints, not just UI:**
  - Heuristic picks ≤1M-token single stages, DailyDialog-heavy if
    `conv + 0.05 < instr`, Dolly-heavy if `instr <= conv`, else stops. It
    reads the same dead `last_promotion_eval` keys as above, so with the
    skill-profile bug it falls back to defaults (instr 0.3 / conv 0.05 →
    always conversation-heavy). Two-branch logic, no dataset/stage choice.
  - Frontend (`src/adamlm/web/index.html:90-147`) exposes only max tokens,
    duration, max experiments (1–10), goal text, planning mode, promotion
    mode. **No dataset picker, no stage-count/budget-per-stage control, no
    plan preview, no per-experiment decision rationale surface** beyond the
    downloadable report. The `.report.md` for failed sessions is terse
    (e.g. `research-20260918-012927-1fd8.report.md` repeats one line ×3).
  - `ai_propose` sends only a compact summary (approved/head/last-eval/
    datasets/disk/budgets, ≤6000 chars) — data-leak posture is fine.
- Provider lock-in (confirmed, `src/adamlm/research.py:48-236`,
  `:246-376`): `config/research-provider.json` (`base_url` openrouter,
  `:free`-suffix gate `is_free_model_id`, ordered fallback walk,
  `max_api_cost_usd` forced 0, key in `config/.research-api-key` never sent
  to browser, `/models` probe in `test_provider_connection`). Chat path is
  clean: `_start_chat` never touches the provider. Training uses only
  `bpe_train` CLI. **No paid-request or Chat-substitution hole found** —
  keep these invariants while abstracting the interface (Phase 5).

### 0.4 Evaluation: previously repaired, now correct-but-split (confirmed)

- Current call shapes are **correct — do not "re-fix"**:
  - `scripts/evaluate.py:21-27` → `evaluate_checkpoints(checkpoints, suite,
    output, device, max_new_tokens)` matches `src/adamlm/evaluation.py:
    313-319` signature.
  - `src/adamlm/eval_full.py:124-140` passes a file path as `suite_path` to
    `ev.evaluate_checkpoints` — correct.
  - Suite `config/eval-suite.json` v1, 29 items (4 grammar + 3 coherence
    paired-choice, 2 generation samples unscored, 10 knowledge/Windows +
    6 reasoning/instruction generation, 4 unsupported/abstention). Greedy
    argmax + mean-logprob choice scoring, deterministic. Good.
- **The real gap is architectural, not a crash:**
  - Auto-promotion path (`Supervisor._finish`, `auto_train.py:655-698` via
    `web.py:529-548` `_AutoCtx.run_eval`) runs **only**
    `scripts/eval_assistant.py` (11 probes: greetings/instructions/
    follow-ups, tokens=60 temp=0.8 top_k=40 seed=42 CPU) + `compare_evals`
    (`auto_train.py:810-841`: hygiene + both-skill F1 win-or-tie,
    `PROMOTE_TOLERANCE = 0.0`). **Capability suite and validation loss play
    no part in auto-promotion.**
  - Manual compare (`eval_full.compare_checkpoints`, `eval_full.py:
    214-235`) runs probes + capability suite + scorecard with 0.02
    capability tolerance and `old/new_validation_loss` **parameters — but
    `compare_checkpoints` never fills them** (always `None`); no caller
    passes checkpoint `status.json` validation loss through. Hence
    "validation loss was not evaluated" — confirmed in code, not just UI.
  - Duplicated F1/hygiene logic in three places (`auto_train.py:64-77 +
    767-841`, `eval_full.py:35-104`) with slightly different bucketing
    (`_replies_by_kind` vs `_skill_split`). Unify behind one module.
  - Small-n warning: promotion hinges on instr_n=4, conv_n=2 F1 means.
    Treat as smoke signal, never as intelligence proof — the code already
    says this (`eval_full.py:143-193` note); the UI must say it too.

### 0.5 Chat (confirmed)

- `_start_chat` + `_compose_chat_prompt` (`web.py:881-956`): approved-only,
  server-side SFT template (`User:/Context:/Assistant:`), 512-token budget
  accounting (`CHAT_BLOCK_TOKENS`, margin, char-per-token estimate),
  honest kept/total/trimmed counts, seed handling, GPU-claimed generation
  workers (`_start_generation`, `:712-845`). Saved chats opt-in
  (`chatstore.py`), feedback→training only after explicit approval. Solid
  base — needs polish, not replacement.

### 0.6 Training / safety infra (confirmed — preserve)

- `src/adamlm/bpe_train.py` (444 lines), `gui_core.build_plan`,
  `gpu_session.py` (exclusive claim `results/.gpu-session.json`, non-blocking
  lock, age-based stale release — never PID-kill), `_require_gpu_free`
  (`web.py:207-217`), graceful stop via `STOP_REQUESTED`, atomic JSON
  writes, storage guards (25 GiB project budget per `config/sft-assistant.
  json`, 5 GiB free floor). Working — do not rewrite.
- Backend: stdlib `http.server` + vanilla JS (`app.js` 1989 lines,
  4 tabs: chat/research/training/versions). Prefer incremental improvement
  over a framework migration.

### 0.7 Suspected-but-NOT-confirmed (do not state as fact)

- Whether repeated 8.29M passes actively *harmed* weights vs. merely wasted
  power (only the eval deltas are confirmed).
- Whether the researcher "repeats stages without control" beyond the above:
  confirmed missing controls are dataset/stage/preview/rationale; anything
  about intent is not confirmed.
- Exact prior eval crash stack (mismatched args / suite-path): only the
  repaired state was inspected; history is unavailable (no git).

---

## 1. Proposed UX and key workflows

### 1.1 Chat — polished, honest, unchanged model authority

- Header: "AdamLM · Approved `auto-assistant-…-s3 · step 00002024`" version
  pill (from `versions.approved_summary`), GPU/activity status, context note
  "using X of Y turns" + trimmed indicator (already computed server-side).
- Controls: New / Save (opt-in) / Saved list / Delete, regenerate, seed
  lock in Advanced (default random), token slider (cap 4096 as today),
  temperature/top-k collapsed under Advanced.
- Honesty: capability footnote ("small local model; strong at short
  grammar/coherence, weak at factual/reasoning — verified 7/27 baseline,
  suite 0.22 approved"), abstention-friendly system behavior unchanged,
  errors name the cause + next action ("GPU busy (training: …) — Chat
  queued/refused, try after stop").
- No checkpoint picker in Chat. Ever. Lab/A-B stays in Versions.

### 1.2 Research — configurable, previewable, explainable, stoppable

- New session form: goal, planning mode (local/AI), **dataset multiselect
  (from SUPPORTED_DATASETS + mix weights when mixture), stage plan
  (1–3 stages, tokens per stage, default 1 stage ≤1M for diagnosis)**,
  token + time + experiment budgets, promotion mode (manual default).
- **Preflight preview (required):** parent checkpoint, diet mixture + basis
  string, effective-vs-requested tokens with cap explanation, stage table
  with epoch labels, storage + ETA, eval plan, explicit [Start] [Edit]
  [Cancel]. Nothing starts on form submit alone.
- Live view: per-experiment status, tokens/time consumed vs budget bars,
  current stage log tail, [Pause safely] [Resume] [Stop safely] — all mapped
  to existing `stop_session`/graceful-stop semantics (checkpoint-safe,
  resumable without replanning).
- Decision log per experiment: hypothesis, supervisor (local-heuristic +
  inputs OR provider model id + attempt trail), validated proposal summary,
  eval deltas, verdict + reason. Failed AI sessions show *which models were
  tried and why each failed* (attempts already exist in `ai_propose`
  envelope — surface them).

### 1.3 Evaluation — one gate, identical settings, honest numbers

- Single comparison report shape for auto AND manual paths: assistant
  probes (identical prompts/seeds/settings, CPU) + capability suite
  (greedy, same suite hash) + **recorded validation-loss pair** (reported,
  never a promotion criterion) + hygiene + 6 raw reply pairs.
- Verdict block: PROMOTABLE / HELD + bullet reasons (which skill regressed
  by how much on n=4/n=2, capability Δ vs 0.02 tolerance, hygiene list),
  plus the mandatory footnote: "n=4/n=2 unigram F1 is a smoke signal, not a
  measure of conversational intelligence; lower loss alone never promotes."
- History: `results/evals/` list with old→new, Δs, reason; link from each
  plan/session and each Versions candidate row.
- Progress-across-stages: per-stage eval rows appended to the plan file so
  s1→s2→s3 deltas are visible, not just head-vs-final.

### 1.4 Models & Versions — lineage at a glance

- Top: Approved card (version, step, tokens, promoted-at, reason, eval Δs,
  [Rollback to previous] when available).
- Second: Candidates awaiting decision (head vs approved Δs + [Compare]
  [Promote… reason required] [Hold]).
- Then: collapsible groups — Assistant lineage (experimental branches,
  newest first, with plan links), Foundations & general pretraining,
  Smoke/benchmark (collapsed by default, clearly badged), Recovery (any
  interrupted/stopped with resumable checkpoints — surfaced, never hidden).
- Each row: version id, step/tokens, dataset/stage, parent run + verified
  flag, plan/session link, state badge. Filters by group + text search.
  Optional archive flag (hide from default view, never delete).

### 1.5 Providers — seam now, new providers later (design only)

- `SupervisorProvider` interface in a new `src/adamlm/providers.py`:
  `id, display_name, is_free(model), list_models(), chat_complete(messages,
  caps) -> {text, attempts}`. OpenRouter moves behind it unchanged
  (behavior, `:free` gate, cost-0 enforcement, key handling identical).
- Hard rules preserved in code + UI copy: cloud supervises research
  planning only; Chat is always local approved; provider output is
  untrusted JSON validated by `validate_proposal`; no weights training on
  cloud outputs beyond the existing SFT datasets; GPU exclusivity unchanged.
- Ollama + other free APIs:228: **designed-for, not built** (config schema
  reserves `provider: openrouter | ollama | custom-openai`, base_url/model
  fields already exist). No implementation in this upgrade.

### 1.6 Interface — consistent, responsive, no new framework

- Keep stdlib backend + vanilla JS. Add: shared status/error/next-action
  component, tab-level loading/empty/error states, CSS breakpoints for
  1920×1080 and 1600×900 (panels wrap, scorecards scroll horizontally,
  no fixed-width overflow), keyboard-accessible tabs (already
  `role=tablist`), visible focus, reduced-motion respect.
- Every mutating action confirms + reports: Start (preview first), Stop
  (graceful, "checkpoint safe"), Promote (reason required + eval summary),
  Rollback (explicit confirm, pointer-only).

---

## 2. Backend changes (by file)

1. `src/adamlm/auto_train.py`
   - Fix `read_skill_profile()` key mapping: accept both legacy
     `instr_f1/conv_f1` and real `instr_f1_new/conv_f1_new`
     (prefer `_new`, fall back to `_old`, then legacy). Unit-test with the
     real `results/assistant_default.json` shape.
   - Parameterize `Supervisor.stage_command()` dataset (replace hardcoded
     `"dailydialog"` with plan-level diet; keep mix default), rename session
     label to match.
   - Default repeat policy: single-pass default in UI/preflight (cap stays
     MAX_EPOCHS=3 but UI warns past 1 pass; multi-pass requires explicit
     "repeat data" acknowledgment).
   - Record per-stage eval rows in plan file during `_run_stage`/`_finish`.
   - Route `_finish` eval through the unified gate (see `eval_full.py`).
2. `src/adamlm/eval_full.py` (single gate)
   - Merge `assistant_scores/compare_evals` here; `auto_train.compare_evals`
     becomes a thin wrapper (or vice versa — pick one, delete the other).
   - `compare_checkpoints()` gains `old/new_validation_loss` filled from
     each checkpoint dir's `status.json` (+ `metrics.jsonl` fallback);
     reported only.
   - Embed settings manifest (suite hash, probe settings, device, seeds)
     in every report; `save_report` links report name back into plan/session.
3. `src/adamlm/research.py`
   - `new_session()` accepts dataset allow-list + mixture + stage budget
     hint; `local_heuristic_proposal()` honors the allow-list and reads the
     fixed skill keys; `session_overview()` exposes budgets consumed,
     per-experiment status, supervisor attempts.
   - Extract provider calls behind `providers.py` interface (OpenRouter
     behavior byte-identical). No new providers.
4. `src/adamlm/web.py` (+ `gui_core.py` untouched unless needed)
   - New endpoints: `GET /api/auto/preview` (already preflight — expose
     diet/skill-basis fields), `POST /api/research/preview`,
     `GET /api/research/decision-log`, `GET /api/evals` (+ item),
     `GET /api/versions?group=&q=` (backed by `versions.build_registry`
     grouping + archive flags in `results/versions.json`).
   - Pass validation-loss pair + suite results through `_AutoCtx.run_eval`
     into plan `eval` block.
   - Keep `_require_gpu_free`, `_resolve_checkpoint` allow-list,
     key-never-to-browser exactly as-is.
5. `src/adamlm/versions.py`
   - Grouped registry output (approved/candidates/lineage/foundations/
     smoke/recovery) + `archived` flag persistence; ancestry unchanged.
6. `src/adamlm/providers.py` (new, ~150 lines)
   - Interface + OpenRouter adapter moved verbatim from `research.py`
     (`is_free_model_id`, `supervisor_models`, `_chat_completion*`,
     error classification). `research.py` imports it.
7. Frontend (`web/index.html`, `app.js`, `styles.css`)
   - Chat polish; Research form + preview + budgets + decision log;
     Training preflight diet/skill-basis display + repeat acknowledgment;
     Versions grouping/filter; shared status/error component; responsive CSS.

## 3. Frontend changes (minimal, same stack)

- No framework. Extend existing panels; add preview/decision/grouping
  renders to `app.js` with the same fetch patterns; add two breakpoints
  (`≤1700px`, `≤1200px`) to `styles.css`; reuse `.eval-scorecard`,
  `.plan-status`, `.action-note` conventions.

---

## 4. Prioritized phases (small, independently verifiable; stop after each)

### Phase 0 — Freeze & baseline (½ day, NO training, NO provider calls)

- Snapshot: copy `results/assistant_default.json` → timestamped backup
  (new file only, e.g. `results/assistant_default.20260918.bak.json`);
  record SHA-256 of approved + previous checkpoint files; `git`-absent
  noted, no init.
- Verify: load approved pointer read-only; `GET /api/overview`,
  `GET /api/assistant`, `GET /api/versions` return 200 with current data.
  (`GET /api/evals` exists but is NOT a baseline gate: it returns an empty
  list when no comparison reports exist yet, so requiring it before any
  evaluation has run would fail a healthy system. Check it opportunistically
  only.)
- **Stop point:** baseline note committed to plan file appendix. **Checks:**
  the three baseline endpoints return 200, pointer backup exists, no process
  started (status shows no active trainer, GPU session untouched).

### Phase 1 — Eval gate + skill-profile fix (highest value, before ANY training)

- Fix `read_skill_profile` keys; unify F1/hygiene into `eval_full`;
  auto `_finish` uses unified gate (probes + suite + reported val-loss);
  per-stage eval rows; UI verdict footnote.
- **Stop point:** gate returns identical verdict on the two held
  checkpoints as `compare-489100e9ce12.json` (HELD, instr reason +
  suite −0.037). **Checks:** `pytest tests/test_evaluation.py
  tests/test_web.py -q` green; manual `POST /api/eval/compare`
  old=approved new=held-candidate → HELD with reasons; no pointer change.
- **Do this before any further training** (§6).

### Phase 2 — Research control & transparency

- Session form gains dataset allow-list + stage budget; mandatory preview;
  heuristic honors allow-list + fixed skill keys; decision log incl.
  supervisor attempts; pause/resume/stop wired to existing session ops.
- **Stop point:** local-only 1-stage ≤1M-token preview→start→stop→resume
  cycle on an *existing* scratch dir completes without launching training
  (use dry-run/preview + a stopped session; training launch only if user
  explicitly approves later). **Checks:** preview shows diet/basis/cap;
  decision log renders; stop leaves checkpoint + resumable state.

### Phase 3 — Versions clarity

- Grouped registry + archive flag + filters + candidate Compare/Promote/Hold
  wiring to existing endpoints; smoke collapsed; recovery surfaced.
- **Stop point:** Versions page shows Approved → Candidates → Lineage →
  Foundations → Smoke with counts matching `build_registry`. **Checks:**
  manual click-through; archive hides without deleting (file still on disk).

### Phase 4 — Chat polish + responsive shell

- Header pill, context/trim note, save/regenerate/seed-lock, shared
  error/next-action component, 1920×1080 + 1600×900 CSS pass.
- **Stop point:** Chat against approved only; GPU-busy refusal message
  verified by holding a fake session claim in a temp dir (unit) — never by
  racing real training. **Checks:** `pytest -q`, manual resize test.

### Phase 5 — Provider seam (no new providers)

- New `providers.py` with OpenRouter moved verbatim; `research.py` imports;
  config reserves `provider` field (default `openrouter`, ignored values
  rejected); all guards (`:free`, cost 0, key handling) unchanged +
  covered by tests.
- **Stop point:** existing provider tests + `test_provider_connection`
  dry-path green; no network calls in tests. **Checks:** local-only session
  unaffected; AI path with disabled provider errors exactly as before.

---

## 5. Minimal tests / manual checks per phase

- Unit (fast, CPU, temp dirs only — follow `tests/test_web.py` `_FakeGpu`
  pattern; never touch `results/` or GPU):
  - Phase 1: skill-profile key mapping (new/old/legacy shapes); gate
    verdicts (hygiene fail / instr regress / conv regress / suite regress
    beyond 0.02 / promotable tie); report settings-manifest presence.
  - Phase 5: `:free` gate rejects paid/empty; cost-cap ≠ 0 rejected;
    fallback list dedup/normalize (existing behavior locked).
- Manual (localhost only): endpoint 200s; preview→confirm flows; eval
  compare HELD on the known pair; Versions grouping counts; resize
  1920×1080 + 1600×900; graceful-stop leaves checkpoint + resumable plan.
- Explicitly NOT required: full training runs, GPU eval sweeps, large suite
  runs, unrelated cleanup, framework migration.

## 6. Do BEFORE any further training (blocking list)

1. Phase 1 skill-profile fix (else every plan mis-allocates the diet tilt).
2. Unified gate with suite + reported val-loss live in both auto and manual
   paths (else repeats of the 24.87M regression can promote on F1 noise).
3. Mandatory plan preview with effective-vs-requested + repeat-data
   acknowledgment for >1 pass (else silent 3× repeats recur).
4. Single-pass default (≤1M diagnostic first); multi-pass only with explicit
   justification + fresh-diet requirement noted.
5. Confirm foundation `general-training-500m` completed + storage floor OK
   via existing preflight (already enforced — keep).

## 7. Risks to models, runs, compatibility

- Pointer safety: promotion/rollback are pointer swaps; code changes must
  never write weights. Mitigation: Phase 0 backup; `promote_candidate`
  keeps `previous`; tests assert pointer-only writes.
- Checkpoint compat: no architecture/tokenizer/dataset-manifest changes in
  this upgrade. `bpe_train` resume rules untouched.
- Disk: eval reports + per-stage rows are KBs; registry stays small. No
  retention-policy change (keep=3) in this upgrade.
- GPU: exclusivity module untouched except imports; generation/eval stay
  CPU for comparisons (as today) to avoid VRAM contention.
- No-git: no history to bisect; rely on timestamped pointer backups +
  immutable plan/session files. Do not delete `results/*.auto.json`,
  `results/research/*`, `checkpoints/`, `config/.research-api-key`,
  `results/conversations/`, `results/feedback/`.
- Scope: `config/research-provider.json` has `max_api_requests: 1.9e+21`
  (effectively unbounded count; cost cap 0 is the real guard). Phase 5
  should replace with a sane count (e.g. 25, the code default) — needs your
  decision (§8).

## 8. Decisions needed from you

1. Repeat policy: single-pass default with explicit multi-pass
   acknowledgment (recommended) vs hard cap at 1 pass?
2. Promotion tolerance: keep strict `0.0` F1 + 0.02 suite (recommended) vs
   relax? Keeping strict is what held the bad candidate.
3. Archive vs delete for Versions clutter: archive-hide only, never delete
   (recommended) — confirm.
4. Provider count cap: reset `max_api_requests` to 25 default
   (recommended) — confirm.
5. Chat scope: stay text-only local (recommended) vs any voice/image work
   — out of scope for 2.0?
6. Git: initialize a repo for the upgrade work (recommended for review) or
   continue without version control?

## 9. File reference index (implementer entry points)

- Pointer/lineage: `results/assistant_default.json`, `src/adamlm/versions.py`,
  `src/adamlm/gui_core.py:206-219,323-382`.
- Planner/executor/gate: `src/adamlm/auto_train.py:53-61,150-229,266-334,
  453-698,752-841`; mix `data/sft_assistant/processed-manifest.json`,
  `config/sft-assistant.json`.
- Research: `src/adamlm/research.py:48-236,377-450,463-638,1119-1230`;
  `config/research-provider.json`, `config/.research-api-key`.
- Eval: `src/adamlm/eval_full.py`, `src/adamlm/evaluation.py:260-361`,
  `scripts/eval_assistant.py`, `scripts/evaluate.py`,
  `config/eval-suite.json`, `results/evals/compare-489100e9ce12.json`.
- Web/Chat: `src/adamlm/web.py:207-217,529-548,712-956,963-1007`,
  `src/adamlm/web/{index.html,app.js,styles.css}`, `src/adamlm/chatstore.py`,
  `src/adamlm/gpu_session.py`, `src/adamlm/inference.py`.
- Evidence plans: `results/auto-assistant-20260918-012952.auto.json`,
  `results/research/research-20260918-012927-1fd8.{json,report.md}`.
