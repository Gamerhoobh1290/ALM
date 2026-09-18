"""Automatic assistant-training planner, executor, and promotion gate.

Auto Train keeps one general-purpose AdamLM Assistant improving without the
operator choosing runs, datasets, checkpoints, stage names, or mixtures.

Design notes (all state lives on disk so a backend restart never loses a plan):
- The main assistant lineage is identified by run-name prefix + SFT stage +
  real checkpoints — never by bare recency, and smoke/production runs are
  excluded from both candidacy and parenting.
- The 500M general model (``general-training-500m``) is the frozen
  foundation: it must exist completed, and Auto Train never writes into it
  or restarts it — every stage continues the assistant head instead.
- Training diet is the curated ``data/sft_assistant`` mix (DailyDialog for
  conversation, Dolly for instructions) with the unchanged
  response-only loss masking. Allocation is token-level and measured, with
  any budget cap or data repetition stated explicitly in preflight.
- Multi-stage plans only occur when the effective budget exceeds one full
  pass of the mixed file; stages run sequentially under ONE token budget
  and ONE time deadline.
- Promotion to the Playground default requires a held-out eval win (or tie)
  on both skills with all hygiene checks passing — never loss alone.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

from . import gui_core
from .gui_core import (
    ROOT,
    RunInfo,
    atomic_json,
    build_plan,
    checkpoint_category,
    discover_runs,
    format_duration,
    format_number,
    read_json,
    write_session,
)
from .sft_data import epoch_capacity, manifest_epoch_tokens, read_dataset_cursor

# Run-name prefixes that constitute the main assistant lineage. Anything else
# (including the paused DailyDialog-only experiment) is never auto-continued.
ASSISTANT_LINEAGE_PREFIXES = ("sft-assistant", "auto-assistant")
FOUNDATION_RUN = "general-training-500m"
MIX_MANIFEST = "data/sft_assistant/processed-manifest.json"
MIX_CONFIG = "config/sft-assistant.json"
# Base token split, measured — not arbitrary: at these shares the reading diet
# stays conversation-majority while Dolly's ~3.3x response-token density keeps
# the actual response supervision roughly balanced between the skills.
BASE_DD_SHARE = 0.70
# Conservative per-stage disk estimate slack beyond kept checkpoints.
STAGE_SLACK_BYTES = 64 * 2**20
# Promotion tolerance: strict bar, ties keep the lineage moving forward.
PROMOTE_TOLERANCE = 0.0
ASSISTANT_KINDS = {"greeting", "follow-up"}  # conversation skill bucket


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def unigram_f1(reply: str, reference: str) -> float:
    """Overlap F1 over lowercased word multisets; 0 when either side is empty."""
    hyp, ref = Counter(_words(reply)), Counter(_words(reference))
    if not hyp or not ref:
        return 0.0
    overlap = sum(min(hyp[w], ref[w]) for w in hyp.keys() & ref.keys())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(hyp.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def assistant_runs(runs: list[RunInfo] | None = None, root: Path = ROOT) -> list[RunInfo]:
    """Candidate heads of the main assistant lineage (excludes smoke/production)."""
    found = []
    for run in runs if runs is not None else discover_runs(root):
        if not run.name.startswith(ASSISTANT_LINEAGE_PREFIXES):
            continue
        if checkpoint_category(run) == "smoke-test":
            continue
        stage = run.launcher.get("stage") or run.status.get("stage")
        if stage != "sft":
            continue
        if not run.checkpoint or not run.checkpoint.is_file():
            continue
        found.append(run)
    return found


def find_assistant_head(runs: list[RunInfo] | None = None, root: Path = ROOT) -> RunInfo:
    """Most-trained assistant head: most tokens, newest run on ties.

    Deliberately not "newest checkpoint overall": smoke tests, production
    snapshots, and unrelated experiments can never become the head.
    """
    candidates = assistant_runs(runs, root)
    if not candidates:
        raise ValueError("No assistant lineage found (runs named sft-assistant*/auto-assistant* with SFT checkpoints). Train one manually first.")
    return max(candidates, key=lambda r: (r.tokens, r.directory.stat().st_mtime))


def check_foundation(runs: list[RunInfo] | None = None, root: Path = ROOT) -> RunInfo:
    """The completed 500M general model must exist; it is never trained into."""
    for run in runs if runs is not None else discover_runs(root):
        if run.name == FOUNDATION_RUN:
            if (run.status.get("state") == "target_reached" and run.checkpoint
                    and run.checkpoint.is_file()):
                return run
            raise ValueError(f"Foundation run '{FOUNDATION_RUN}' exists but is not a completed model; will not train without it.")
    raise ValueError(f"Foundation run '{FOUNDATION_RUN}' is missing; will not train without it.")


def read_mix_stats(root: Path = ROOT) -> dict:
    """Measured token availability from the curated mix manifest (no guessing)."""
    manifest_path = root / MIX_MANIFEST
    manifest = read_json(manifest_path, {})
    if manifest.get("format") != "adamlm-response-sft-v1" or "sources" not in manifest:
        raise ValueError(f"Assistant mix manifest is missing or unsupported: {MIX_MANIFEST}")
    sources = manifest["sources"]
    try:
        dd_tokens = int(sources["dailydialog"]["tokens"])
        dolly_tokens = int(sources["dolly"]["tokens"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Assistant mix manifest lacks measured token counts: {MIX_MANIFEST}") from None
    train_file = root / manifest["files"]["train"]["path"]
    if not train_file.is_file():
        raise ValueError(f"Assistant mixed training file is missing: {train_file}")
    # Same helper the trainer sizes a pass with, so neither can drift.
    epoch_tokens = manifest_epoch_tokens(manifest_path) or 0
    if epoch_tokens <= 0:
        raise ValueError("Assistant mixed diet reports zero tokens")
    return {
        "manifest": MIX_MANIFEST,
        "revision": manifest.get("revision", "?"),
        "dd_rows": int(sources["dailydialog"]["rows"]),
        "dolly_rows": int(sources["dolly"]["rows"]),
        "dd_tokens": dd_tokens,
        "dolly_tokens": dolly_tokens,
        "epoch_tokens": epoch_tokens,
        "train_examples": int(manifest["files"]["train"]["examples"]),
    }


def read_skill_profile(root: Path = ROOT) -> dict | None:
    """Last promotion eval for the current default assistant, if any.

    Returns {"instr_f1": x, "conv_f1": y} or None when no evaluated default
    exists yet — in which case the planner says so and uses the base mix.

    Real pointers store per-side keys (instr_f1_new/instr_f1_old,
    conv_f1_new/conv_f1_old); the current head measurement (_new) is
    preferred, then the previous side (_old), then the legacy flat keys.
    """
    pointer = read_json(root / "results" / "assistant_default.json", {})
    evaluation = pointer.get("eval") or {}

    def _number(*keys: str) -> float | None:
        for key in keys:
            try:
                return float(evaluation[key])
            except (KeyError, TypeError, ValueError):
                continue
        return None

    instr = _number("instr_f1", "instr_f1_new", "instr_f1_old")
    conv = _number("conv_f1", "conv_f1_new", "conv_f1_old")
    if instr is None or conv is None:
        return None
    return {"instr_f1": instr, "conv_f1": conv}


def allocate(requested_tokens: int, epoch_tokens: int, skill: dict | None = None, *,
             tokens_per_step: int = 1, pass_policy: str = "unique_once",
             stage_count: int | None = None, consumed_tokens: int = 0,
             epoch_index: int = 0) -> dict:
    """Split a token budget into explicit, honestly-labeled stage chunks.

    - Base 70/30 DailyDialog/Dolly token split (measured basis), tilted at
      most ±10pp toward the weaker skill when a prior evaluated profile
      exists, clamped to [50/50, 85/15].
    - Finite data capacity is rounded down to complete optimizer batches.
    - Repetition is unlimited only when explicitly requested and is never
      counted as unique data.
    - Capacity is measured from where the lineage's cursor actually stands
      (``consumed_tokens`` within pass ``epoch_index``), not from the start
      of the corpus, so a plan never schedules rows that are already used.
      The first stage continues that pass; later stages are repeats and are
      labeled as repeated data.
    """
    if requested_tokens <= 0:
        raise ValueError("Additional tokens must be positive")
    dd_share = BASE_DD_SHARE
    basis = (f"base {BASE_DD_SHARE:.0%} DailyDialog / {1 - BASE_DD_SHARE:.0%} Dolly by sampled "
             "training tokens: conversation-majority reading diet with roughly balanced "
             "response supervision (Dolly rows carry ~3.3x the response tokens)")
    if skill is not None:
        gap = skill["conv_f1"] - skill["instr_f1"]
        if gap > 0.05:
            dd_share = max(0.50, dd_share - 0.10)
            basis += (f"; tilted −10pp toward instructions on measured skill gap "
                      f"(instructions F1 {skill['instr_f1']:.3f} < conversation F1 {skill['conv_f1']:.3f})")
        elif gap < -0.05:
            dd_share = min(0.85, dd_share + 0.05)
            basis += (f"; tilted +5pp toward conversation on measured skill gap "
                      f"(conversation F1 {skill['conv_f1']:.3f} < instructions F1 {skill['instr_f1']:.3f})")
        else:
            basis += (f"; skills within noise (instructions F1 {skill['instr_f1']:.3f}, "
                      f"conversation F1 {skill['conv_f1']:.3f}), no tilt applied")
    else:
        basis += "; no evaluated skill profile yet, no tilt applied"
    if tokens_per_step <= 0:
        raise ValueError("Training batch size must be positive")
    if pass_policy not in {"unique_once", "repeat"}:
        raise ValueError("Dataset/pass policy must be 'unique_once' or 'repeat'")
    # One shared capacity calculation with the trainer: same batch rounding,
    # same discarded sub-batch tail, same reading of how much of this pass
    # the lineage has already consumed.
    cap = epoch_capacity(epoch_tokens, tokens_per_step, consumed_tokens)
    usable_epoch = cap["batchable_tokens"]
    tail_tokens = cap["discarded_tail_tokens"]
    remaining_epoch = cap["remaining_tokens"]
    if usable_epoch <= 0:
        raise ValueError(
            f"The dataset has {format_number(epoch_tokens)} tokens, fewer than one complete "
            f"training batch of {format_number(tokens_per_step)} tokens.")
    if pass_policy == "unique_once" and requested_tokens > remaining_epoch:
        raise ValueError(
            f"Requested {format_number(requested_tokens)} tokens exceeds the "
            f"{format_number(remaining_epoch)} unused tokens left in pass {epoch_index + 1} "
            f"({format_number(consumed_tokens)} of {format_number(usable_epoch)} batchable "
            f"tokens already trained). Choose the repeat policy explicitly or add unique data.")

    # Stage 1 finishes the pass the lineage is standing in; every later stage
    # is a full repeat of the corpus. A pass supplies only its batchable
    # tokens, so capping here is what stops a request expressed in
    # source-epoch units from claiming tail tokens that do not exist.
    first_stage_repeats = remaining_epoch <= 0
    first_capacity = usable_epoch if first_stage_repeats else remaining_epoch
    rounded = ((requested_tokens + tokens_per_step - 1) // tokens_per_step) * tokens_per_step
    # A pass supplies only its batchable tokens, so a request expressed in
    # source-epoch units must not claim the sub-batch tails that do not
    # exist, nor spill that phantom debt into an extra stage.
    source_left_now = epoch_tokens if first_stage_repeats else epoch_tokens - consumed_tokens
    passes_spanned = 1
    if requested_tokens > source_left_now:
        passes_spanned += -(-(requested_tokens - source_left_now) // epoch_tokens)
    if stage_count is None:
        scheduled = min(rounded, first_capacity + (passes_spanned - 1) * usable_epoch)
    else:
        try:
            requested_stages = int(stage_count)
        except (TypeError, ValueError):
            raise ValueError("Stage count must be a positive whole number") from None
        if requested_stages <= 0:
            raise ValueError("Stage count must be a positive whole number")
        scheduled = min(rounded, first_capacity + (requested_stages - 1) * usable_epoch)
    minimum_stages = 1
    while first_capacity + (minimum_stages - 1) * usable_epoch < scheduled:
        minimum_stages += 1
    if stage_count is None:
        stage_count = minimum_stages
    try:
        stage_count = int(stage_count)
    except (TypeError, ValueError):
        raise ValueError("Stage count must be a positive whole number") from None
    if stage_count <= 0:
        raise ValueError("Stage count must be a positive whole number")
    if stage_count < minimum_stages:
        raise ValueError(
            f"{format_number(scheduled)} scheduled tokens need at least {minimum_stages} stages "
            f"because one finite pass supplies at most {format_number(usable_epoch)} batchable tokens.")
    total_batches = scheduled // tokens_per_step
    if stage_count > total_batches:
        raise ValueError(
            f"Stage count {stage_count} exceeds the {format_number(total_batches)} complete batches "
            "in this plan; every stage must perform at least one optimizer update.")

    # Spread the work as evenly as the data allows, but never give a stage
    # more than its pass can supply: the first stage is limited to what is
    # left of the current pass, later stages to one full repeat pass each.
    caps = [(first_capacity if index == 0 else usable_epoch) // tokens_per_step
            for index in range(stage_count)]
    takes = [0] * stage_count
    left = total_batches
    while left > 0:
        open_stages = [index for index in range(stage_count) if takes[index] < caps[index]]
        if not open_stages:
            raise ValueError(
                f"{format_number(scheduled)} scheduled tokens do not fit in {stage_count} stages "
                f"bounded by the data available to each; raise the stage count or lower the target.")
        share = max(1, left // len(open_stages))
        for index in open_stages:
            if left <= 0:
                break
            give = min(share, caps[index] - takes[index], left)
            takes[index] += give
            left -= give
    chunks = []
    for index in range(stage_count):
        take = takes[index] * tokens_per_step
        # Pass numbering continues the lineage instead of restarting at 1.
        pass_number = epoch_index + 1 + index + (1 if first_stage_repeats else 0)
        repeats = index > 0 or first_stage_repeats
        span = usable_epoch if repeats else remaining_epoch
        label = f"pass {pass_number}"
        if take != span:
            label += f" (partial, {take / span:.1%} of the tokens available to this stage)"
        if repeats:
            label += "; repeated data — rows already trained in an earlier pass"
        elif consumed_tokens:
            label += (f"; continues the lineage cursor at {format_number(consumed_tokens)} of "
                      f"{format_number(usable_epoch)} batchable tokens")
        chunks.append({"tokens": take, "epoch_label": label, "pass_number": pass_number,
                       "repeated_data": repeats,
                       "source_tokens": epoch_tokens, "batchable_tokens": usable_epoch,
                       "remaining_before": remaining_epoch if index == 0 else usable_epoch,
                       "discarded_tail_tokens": tail_tokens})
    repeated = any(chunk["repeated_data"] for chunk in chunks)
    unique_upper_bound = sum(chunk["tokens"] for chunk in chunks if not chunk["repeated_data"])
    explanation = None
    if scheduled != requested_tokens:
        direction = "down" if scheduled < requested_tokens else "up"
        explanation = (f"Requested target rounded {direction} to {format_number(scheduled)} tokens "
                       f"on {format_number(tokens_per_step)}-token batch boundaries.")
    return {
        "dd_share": dd_share,
        "dolly_share": 1.0 - dd_share,
        "basis": basis,
        "requested_tokens": requested_tokens,
        "effective_tokens": scheduled,
        "scheduled_tokens": scheduled,
        "source_epoch_tokens": epoch_tokens,
        "batchable_epoch_tokens": usable_epoch,
        "discarded_tail_tokens": tail_tokens,
        "epoch_index": epoch_index,
        "consumed_tokens": consumed_tokens,
        "remaining_epoch_tokens": remaining_epoch,
        "first_stage_repeats": first_stage_repeats,
        "pass_policy": pass_policy,
        "stage_count": stage_count,
        "repeated_data": repeated,
        "unique_tokens_upper_bound": unique_upper_bound,
        "capped": scheduled < requested_tokens,
        "cap_explanation": explanation,
        "stage_chunks": chunks,
    }


def plan_id_for(root: Path = ROOT, now: float | None = None) -> str:
    """Readable, unique plan id; numeric suffix on collision (never overwrite)."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now if now is not None else time.time()))
    candidate = f"auto-assistant-{stamp}"
    index = 2
    taken = {p.name for p in (root / "results").glob("auto-assistant-*")} if (root / "results").exists() else set()
    while candidate in taken or (root / "results" / f"{candidate}.auto.json").exists():
        candidate = f"auto-assistant-{stamp}-{index}"
        index += 1
    return candidate


def estimate_storage(root: Path = ROOT, stages: int = 1, checkpoint_bytes: int = 0) -> dict:
    """Conservative disk estimate for the planned stages (never bypassed)."""
    from .storage import StorageBudget
    config = read_json(root / MIX_CONFIG, {})
    storage_cfg = config.get("storage", {})
    budget_gb = float(storage_cfg.get("project_budget_gb", 15))
    floor_gb = float(storage_cfg.get("minimum_free_disk_gb", 5))
    guard = StorageBudget(root, budget_gb, float(storage_cfg.get("cache_limit_gb", 2)), floor_gb)
    snapshot = guard.snapshot()
    est_new = stages * (checkpoint_bytes * int(storage_cfg.get("checkpoint_keep", 3)) + STAGE_SLACK_BYTES)
    return {
        "project_gib": snapshot["project_bytes"] / 2**30,
        "est_new_gib": est_new / 2**30,
        "budget_gib": budget_gb,
        "free_gib": snapshot["disk_free_bytes"] / 2**30,
        "floor_gib": floor_gb,
        "fits_budget": snapshot["project_bytes"] + est_new <= guard.budget_bytes,
        "floor_ok": snapshot["disk_free_bytes"] - est_new >= guard.minimum_free_bytes,
        "config": MIX_CONFIG,
    }


def preflight(requested_tokens: int, max_seconds: float, root: Path = ROOT,
              runs: list[RunInfo] | None = None, *, pass_policy: str = "unique_once",
              stage_count: int | None = None) -> dict:
    """Concise, fully-measured preflight summary. Pure computation: starts nothing."""
    if max_seconds is None or not max_seconds > 0:
        raise ValueError("Maximum session duration must be positive")
    known = runs if runs is not None else discover_runs(root)
    foundation = check_foundation(known, root)
    head = find_assistant_head(known, root)
    if head.name == FOUNDATION_RUN:
        raise ValueError("Assistant head must not be the foundation model itself")
    stats = read_mix_stats(root)
    tps = _tokens_per_step(root)
    if not tps:
        raise ValueError(f"Cannot determine optimizer batch size from {MIX_CONFIG}")
    # Where the lineage's cursor actually stands, read from the checkpoint
    # itself — the trainer re-reads the same cursor from the same file, so a
    # plan that passes preflight is the plan the trainer executes.
    position = head_cursor(head, root)
    allocation = allocate(requested_tokens, stats["epoch_tokens"], read_skill_profile(root),
                          tokens_per_step=tps, pass_policy=pass_policy,
                          stage_count=stage_count, consumed_tokens=position["epoch_tokens"],
                          epoch_index=position["epoch"])
    plan_id = plan_id_for(root)
    stages = []
    for index, chunk in enumerate(allocation["stage_chunks"], start=1):
        stages.append({
            "index": index,
            "run_dir": f"results/{plan_id}-s{index}",
            "target_tokens": chunk["tokens"],
            "epoch_label": chunk["epoch_label"],
            "pass_number": chunk["pass_number"],
            "repeat_epoch": chunk["repeated_data"],
            "source_tokens": chunk["source_tokens"],
            "batchable_tokens": chunk["batchable_tokens"],
            "discarded_tail_tokens": chunk["discarded_tail_tokens"],
            "status": "pending",
        })
    # Dry-run the first stage through the real planner for honest throughput,
    # checkpoint-size, and ETA figures (nothing is launched or created).
    dry = build_plan(mode="continuation", dataset="dailydialog",
                     additional_tokens=stages[0]["target_tokens"],
                     duration_seconds=max_seconds,
                     parent_checkpoint=head.checkpoint, run_dir=root / stages[0]["run_dir"],
                     config_path=root / MIX_CONFIG, root=root)
    storage = estimate_storage(root, stages=len(stages), checkpoint_bytes=dry.checkpoint_bytes)
    if not storage["fits_budget"]:
        raise ValueError(
            f"Planned stages need ~{storage['est_new_gib']:.2f} GiB on top of a "
            f"{storage['project_gib']:.2f} GiB project against a {storage['budget_gib']:.0f} GiB "
            f"budget ({storage['config']}). Free disk is {storage['free_gib']:.1f} GiB — "
            f"raise the configured budget through the resume-compatible storage settings "
            f"or free project space before starting.")
    if not storage["floor_ok"]:
        raise ValueError(
            f"Planned stages would breach the {storage['floor_gib']:.0f} GiB Windows free-space "
            f"safeguard (free: {storage['free_gib']:.1f} GiB). Refusing to start.")
    return {
        "plan_id": plan_id,
        "head": {"run": head.name, "checkpoint": str(head.checkpoint),
                 "step": head.checkpoint and _checkpoint_step(head), "tokens": head.tokens},
        "foundation": {"run": foundation.name, "state": foundation.status.get("state"),
                       "checkpoint": str(foundation.checkpoint)},
        "diet": {"manifest": stats["manifest"], "revision": stats["revision"],
                 "masking": "response-only (unchanged SFT template)",
                 "dailydialog": {"rows": stats["dd_rows"], "tokens": stats["dd_tokens"]},
                 "dolly": {"rows": stats["dolly_rows"], "tokens": stats["dolly_tokens"]}},
        "allocation": allocation,
        "requested_tokens": allocation["requested_tokens"],
        "effective_tokens": allocation["effective_tokens"],
        "budget": {
            "requested_tokens": allocation["requested_tokens"],
            "effective_tokens": allocation["effective_tokens"],
            "source_epoch_tokens": stats["epoch_tokens"],
            "batchable_epoch_tokens": allocation["batchable_epoch_tokens"],
            "discarded_tail_tokens": allocation["discarded_tail_tokens"],
            "pass_policy": allocation["pass_policy"],
            "stage_count": allocation["stage_count"],
            "capped": allocation["capped"],
            "cap_explanation": allocation["cap_explanation"],
            "repeated_data": allocation["repeated_data"],
            "unique_tokens_upper_bound": allocation["unique_tokens_upper_bound"],
            "note": (f"Each fresh pass has {format_number(stats['epoch_tokens'])} source tokens; "
                     f"{format_number(allocation['batchable_epoch_tokens'])} fit complete batches "
                     f"and {format_number(allocation['discarded_tail_tokens'])} tail tokens are not trained. "
                     + ("Repeated passes are explicitly enabled; repeated tokens are not unique."
                        if allocation["repeated_data"] else "No rows are intentionally repeated.")),
        },
        "stages": stages,
        "storage": storage,
        "time": {"max_seconds": max_seconds, "max_duration": format_duration(max_seconds),
                 "est_seconds": dry.estimated_seconds,
                 "est_duration": format_duration(dry.estimated_seconds)},
        "stopping": [
            f"overall time limit of {format_duration(max_seconds)} (stages share one deadline)",
            "Stop button: graceful stop with checkpoint, resumable without replanning",
            "disk pause: stages halt safely on storage-guard trips, checkpoints intact",
        ],
        "validation_policy": {"stage_compare_every": 1, "regression_patience": 2,
                              "manual_promotion": True},
        "eval": ("Every completed stage is compared with its parent. Sustained regression pauses "
                 "the plan. Final evaluation can nominate a candidate, but promotion is manual."),
        "warnings": ([allocation["cap_explanation"]] if allocation["cap_explanation"] else []),
    }


def head_cursor(head: RunInfo, root: Path = ROOT) -> dict:
    """Corpus position the assistant head stands at, from its checkpoint.

    A head trained before the position was tracked, or one whose parent was
    a pretrain run, reports the start of pass 1 — which is exactly where a
    continuation from it should begin. A head whose checkpoint cannot be
    read is refused rather than assumed to be at the start: guessing there
    is what silently replays rows.
    """
    blank = {"epoch": 0, "epoch_tokens": 0, "byte_offset": 0, "tracked": False}
    if not head.checkpoint:
        return blank
    try:
        cursor = read_dataset_cursor(head.checkpoint)
    except Exception as exc:
        raise ValueError(
            f"Cannot read the dataset cursor from the assistant head checkpoint "
            f"({head.checkpoint}): {exc}. Refusing to plan, because assuming a position "
            "would silently retrain rows this lineage has already used.") from None
    if not cursor or "sft_source" not in cursor:
        return blank
    stats = cursor.get("stats") or {}
    consumed = cursor.get("epoch_tokens", stats.get("tokens_consumed", 0))
    return {"epoch": int(cursor.get("epoch", 0)), "epoch_tokens": int(consumed or 0),
            "byte_offset": int(cursor.get("byte_offset", 0)),
            "tracked": "epoch_tokens" in cursor}


def _checkpoint_step(run: RunInfo) -> int | None:
    step = run.status.get("checkpoint_step") or run.status.get("step")
    return int(step) if isinstance(step, int) else None


PLAN_SUFFIX = ".auto.json"


def plan_path(plan_id: str, root: Path = ROOT) -> Path:
    return Path(root) / "results" / f"{plan_id}{PLAN_SUFFIX}"


def write_plan(plan: dict, root: Path = ROOT) -> Path:
    path = plan_path(plan["plan_id"], root)
    atomic_json(path, plan)
    return path


def read_plan(plan_id: str, root: Path = ROOT) -> dict:
    plan = read_json(plan_path(plan_id, root), {})
    if not plan or plan.get("plan_id") != plan_id:
        raise ValueError(f"Auto Train plan '{plan_id}' not found")
    return plan


def latest_plan(root: Path = ROOT) -> dict | None:
    """Most recently updated auto plan, if any (drives the dashboard status)."""
    candidates = []
    results = Path(root) / "results"
    if results.exists():
        for path in results.glob(f"*{PLAN_SUFFIX}"):
            plan = read_json(path, {})
            if plan.get("plan_id"):
                candidates.append(plan)
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.get("updated_at", 0), p.get("created_at", 0)))


def touch(plan: dict, root: Path = ROOT) -> dict:
    plan["updated_at"] = time.time()
    write_plan(plan, root)
    return plan


# ---------------------------------------------------------------------------
# Executor: sequential stages under one token budget and one deadline.
# ---------------------------------------------------------------------------

RESUMABLE_STAGE = {"pending", "running", "stopped", "waiting", "time-exceeded"}
TERMINAL_PLAN = {"stopped", "waiting", "regression", "time-exceeded", "failed", "done"}


def _tokens_per_step(root: Path = ROOT) -> int | None:
    """One optimizer step in tokens (batch * accumulation * block)."""
    try:
        cfg = read_json(Path(root) / MIX_CONFIG, {})
        return int(cfg["batch_size"]) * int(cfg.get("accumulation", 1)) * int(cfg["model"]["block_size"])
    except (KeyError, TypeError, ValueError):
        return None


def _is_full_pass_stage(stage: dict) -> bool:
    """Only stages whose label is a whole pass (no 'partial') end at a data boundary.

    Partial stages must never exhaust: the mixed file holds far more tokens
    than their target, so any end-of-data there is unexpected (truncation,
    corruption, manifest drift) and must stay FAILED.
    """
    label = str(stage.get("epoch_label") or "")
    return bool(label) and ("partial" not in label.lower())


def _expected_full_pass_data_stop(plan: dict, stage: dict, status: dict,
                                  checkpoint, root: Path = ROOT) -> str | None:
    """Explain why an end-of-data stop is an expected pass boundary, else None.

    Expected only when ALL hold (anything else stays FAILED):
    - trainer state is 'data_stop: ...'
    - latest complete checkpoint file exists (never claim unsaved work)
    - actual tokens are within one batch short of the stage target
      (the file tail cannot form a complete batch)
    - the stage is a planned full pass (label has no 'partial')
    """
    state = str(status.get("state") or "")
    if not state.startswith("data_stop"):
        return None
    if checkpoint is None:
        return None
    try:
        if not Path(checkpoint).is_file():
            return None
    except (OSError, ValueError):
        return None
    try:
        tokens = int(status.get("tokens"))
        target = int(stage.get("target_tokens"))
    except (TypeError, ValueError):
        return None
    if tokens <= 0:
        return None
    shortfall = target - tokens
    if shortfall < 0:
        return None
    tps = _tokens_per_step(root)
    if tps is None or tps <= 0:
        return None
    if shortfall >= tps:
        return None
    if not _is_full_pass_stage(stage):
        return None
    name = Path(checkpoint).name
    return (f"full-pass boundary: data exhausted {shortfall} tokens short of target "
            f"(< one batch of {tps}); advancing with actual {tokens} tokens from {name}")


class Supervisor:
    """Runs plan stages sequentially in a background thread.

    The context object wires the supervisor to the host (real backend or
    tests) without import cycles:
      - active_trainer() -> RunInfo | None
      - launch(command, cwd, log_path) -> handle with .poll()/.pid
      - read_stage_status(run_dir) -> dict (status.json content)
      - latest_checkpoint(run_dir) -> Path | None
      - request_stop(run_dir) -> None
      - run_eval(old_checkpoint, new_checkpoint, out_path, timeout) -> parsed JSON
      - sleep(seconds), now() -> float
    """

    def __init__(self, plan_id, root=ROOT, ctx=None):
        self.plan_id = plan_id
        self.root = Path(root)
        self.ctx = ctx

    def _load(self):
        return read_plan(self.plan_id, self.root)

    def _save(self, plan):
        return touch(plan, self.root)

    def _deadline(self, plan):
        return (plan.get("started_at") or 0) + plan["time"]["max_seconds"]

    def _remaining(self, plan):
        return max(0.0, self._deadline(plan) - self.ctx.now())

    def stage_command(self, plan, stage):
        """(Re)build the stage launch command with the current remaining time."""
        parent = stage.get("parent_checkpoint")
        if stage["index"] > 1 or plan["stages"][stage["index"] - 1].get("status") == "done":
            prev = plan["stages"][stage["index"] - 2] if stage["index"] > 1 else None
            if prev is not None:
                parent = str(self.ctx.latest_checkpoint(self.root / prev["run_dir"]) or parent)
        remaining = self._remaining(plan)
        if remaining <= 0:
            raise ValueError("Overall Auto Train time limit is exhausted")
        plan_obj = build_plan(mode="continuation", dataset="dailydialog",
                              additional_tokens=stage["target_tokens"],
                              duration_seconds=remaining,
                              parent_checkpoint=Path(parent),
                              run_dir=self.root / stage["run_dir"],
                              config_path=self.root / MIX_CONFIG, root=self.root)
        command = list(plan_obj.command)
        # Repeats were approved when the plan was accepted; the trainer will
        # not rewind the corpus without being told to, so carry that decision
        # through instead of letting it re-derive one.
        if stage.get("repeat_epoch"):
            command.append("--repeat-epoch")
        return command

    def run(self):
        """Execute until the plan reaches a terminal state. Never raises out."""
        try:
            self._run_inner()
        except Exception as exc:  # plan file must always explain itself
            try:
                plan = self._load()
                if plan.get("status") not in TERMINAL_PLAN:
                    plan["status"] = "failed"
                    plan["failure"] = f"supervisor error: {exc}"
                    self._save(plan)
            except Exception:
                pass

    def _run_inner(self):
        plan = self._load()
        if plan.get("status") not in ("running",):
            return
        if self.ctx.active_trainer() is not None:
            plan["status"] = "failed"
            plan["failure"] = "refusing to start: another trainer is already active"
            self._save(plan)
            return
        if plan.get("started_at") is None:
            plan["started_at"] = self.ctx.now()
            self._save(plan)
        for position in range(len(plan["stages"])):
            plan = self._load()
            if plan["stages"][position].get("status") == "done":
                continue
            if plan.get("status") != "running":
                break
            self._run_stage(plan, position)
            plan = self._load()
        plan = self._load()
        if plan.get("status") != "running":
            return
        if all(s.get("status") == "done" for s in plan["stages"]):
            self._finish(plan)
        elif self._remaining(plan) <= 0:
            plan["status"] = "time-exceeded"
            plan["stop_reason"] = "overall time limit reached with tokens remaining"
            self._save(plan)

    def _run_stage(self, plan, position):
        # NOTE: plan is reloaded from disk inside the wait loop, so the stage
        # is always re-resolved by position — never hold a dict reference
        # across a load, or mutations land on a stale object.
        stage = plan["stages"][position]
        run_dir = self.root / stage["run_dir"]
        try:
            command = self.stage_command(plan, stage)
        except ValueError as exc:
            plan["status"] = "time-exceeded"
            plan["stop_reason"] = str(exc)
            self._save(plan)
            return
        stage["status"] = "running"
        stage["command"] = [str(c) for c in command]
        self._save(plan)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_session(run_dir, {
            "run_name": run_dir.name,
            "mode": "continuation",
            "dataset": "dailydialog+dolly",
            "started_at": self.ctx.now(),
            "max_seconds": self._remaining(plan),
            "target_tokens": stage["target_tokens"],
            "auto_plan": plan["plan_id"],
        })
        log_path = run_dir / "web-training.log"
        handle = self.ctx.launch(command, self.root, log_path)
        stop_sent = False
        while True:
            self.ctx.sleep(5)
            plan = self._load()
            stage = plan["stages"][position]
            if plan.get("status") != "running":
                # User asked to stop (or backend is adopting a stale plan):
                # ask once for a graceful save, then wait for the exit.
                if not stop_sent:
                    try:
                        self.ctx.request_stop(run_dir)
                    except OSError:
                        pass
                    stop_sent = True
            if self._remaining(plan) <= 0 and not stop_sent:
                try:
                    self.ctx.request_stop(run_dir)
                except OSError:
                    pass
                stop_sent = True
                plan["note"] = "overall time limit reached; waiting for a graceful save"
                self._save(plan)
            if handle.poll() is not None:
                break
        status = self.ctx.read_stage_status(run_dir)
        state = str(status.get("state") or "unknown")
        checkpoint = self.ctx.latest_checkpoint(run_dir)
        stage["checkpoint"] = str(checkpoint) if checkpoint else None
        # Actual processed tokens only — never the planned target, so unsaved
        # work is never claimed as preserved.
        stage["tokens"] = status.get("tokens")
        if state == "target_reached":
            stage["status"] = "done"
        elif state.startswith("data_stop"):
            boundary = _expected_full_pass_data_stop(plan, stage, status, checkpoint, self.root)
            if boundary is not None:
                stage["status"] = "done"
                stage["note"] = boundary
            else:
                stage["status"] = "failed"
                if plan.get("status") == "running":
                    plan["status"] = "failed"
                    plan["failure"] = (f"stage {stage['index']} ended as '{state}': "
                                       f"{status.get('detail') or 'see web-training.log'} "
                                       f"(unexpected exhaustion: not a planned full-pass boundary or shortfall exceeds one batch)")
        elif state == "stopped" or (plan.get("status") == "stopped"):
            stage["status"] = "stopped"
            plan["status"] = "stopped"
            plan["stop_reason"] = "stopped by operator; resume continues without replanning"
        elif state in {"disk_pause", "interrupted"}:
            stage["status"] = "waiting"
            plan["status"] = "waiting"
            plan["stop_reason"] = (f"stage {stage['index']} is waiting after {state}; resume uses "
                                   "the last intact checkpoint and exact saved cursor")
            plan["recovery_checkpoint"] = str(checkpoint) if checkpoint else None
        elif state == "regression_pause":
            stage["status"] = "regression"
            plan["status"] = "regression"
            plan["stop_reason"] = (status.get("detail") or
                                   "validation stopped improving; review before creating a different plan")
        elif state == "session_limit" or (stop_sent and self._remaining(plan) <= 0):
            stage["status"] = "time-exceeded"
            plan["status"] = "time-exceeded"
            plan["stop_reason"] = "overall time limit reached; resume re-arms a fresh allowance"
        else:
            stage["status"] = "failed"
            if plan.get("status") == "running":
                plan["status"] = "failed"
                plan["failure"] = (f"stage {stage['index']} ended as '{state}': "
                                   f"{status.get('detail') or 'see web-training.log'}")
        if stage.get("status") == "done" and plan.get("status") == "running":
            self._compare_stage_checkpoint(plan, position)
        self._save(plan)

    def _compare_stage_checkpoint(self, plan, position):
        """Bounded conversational comparison used to pause sustained regression."""
        policy = plan.get("validation_policy") or {}
        every = max(0, int(policy.get("stage_compare_every", 0)))
        if not every or (position + 1) % every:
            return
        stage = plan["stages"][position]
        new_checkpoint = stage.get("checkpoint")
        old_checkpoint = (plan["head"]["checkpoint"] if position == 0 else
                          plan["stages"][position - 1].get("checkpoint"))
        if not old_checkpoint or not new_checkpoint:
            stage["evaluation"] = {"status": "incomplete",
                                   "reason": "parent or candidate checkpoint is missing"}
            return
        report = self.root / stage["run_dir"] / "stage-comparison.json"
        try:
            old_rows, new_rows = self.ctx.run_eval(old_checkpoint, new_checkpoint, report)
            verdict = compare_evals(old_rows, new_rows)
        except Exception as exc:
            stage["evaluation"] = {"status": "incomplete", "reason": str(exc),
                                   "report": str(report)}
            return
        stage["evaluation"] = {"status": "improved-or-tied" if verdict["promote"] else "regressed",
                               "report": str(report), **verdict["scores"]}
        if verdict["promote"]:
            plan["regression_streak"] = 0
            score = (float(verdict["scores"].get("instr_f1_new") or 0) +
                     float(verdict["scores"].get("conv_f1_new") or 0))
            if score >= float((plan.get("best_candidate") or {}).get("score", -1)):
                plan["best_candidate"] = {"checkpoint": new_checkpoint,
                                          "run": Path(stage["run_dir"]).name,
                                          "score": score, "eval": verdict["scores"],
                                          "report": str(report), "status": "unapproved"}
            return
        streak = int(plan.get("regression_streak", 0)) + 1
        plan["regression_streak"] = streak
        patience = max(1, int(policy.get("regression_patience", 2)))
        if streak >= patience:
            plan["status"] = "regression"
            plan["stop_reason"] = (f"{streak} consecutive stage comparisons regressed on held-out "
                                   "conversation/instruction probes; candidate checkpoints are preserved "
                                   "and the approved model is unchanged")

    def _finish(self, plan):
        plan = self._load()
        if plan.get("status") != "running":
            return  # stopped while the last stage saved; leave the plan stopped
        plan["status"] = "evaluating"
        self._save(plan)
        head_ckpt = plan["head"]["checkpoint"]
        new_ckpt = plan["stages"][-1].get("checkpoint")
        if not new_ckpt:
            plan["status"] = "failed"
            plan["failure"] = "final stage completed with no checkpoint to evaluate"
            self._save(plan)
            return
        plan["produced_checkpoint"] = new_ckpt
        eval_path = self.root / plan["stages"][-1]["run_dir"] / "auto_eval.json"
        try:
            old_eval, new_eval = self.ctx.run_eval(head_ckpt, new_ckpt, eval_path)
        except Exception as exc:
            plan["status"] = "done"
            plan["promotion"] = {"promoted": False, "status": "hold",
                                 "reason": f"evaluation incomplete ({exc}); candidate held and default unchanged"}
            self._save(plan)
            return
        from . import eval_full
        try:
            old_suite = eval_full.run_capability_suite(head_ckpt, root=self.root, device="cpu")
            new_suite = eval_full.run_capability_suite(new_ckpt, root=self.root, device="cpu")
            suite_note = None
        except Exception as exc:
            old_suite, new_suite = None, None
            suite_note = f"capability suite unavailable ({exc}); evaluation is incomplete"
        old_vl = eval_full.read_validation_loss(head_ckpt, root=self.root)
        new_vl = eval_full.read_validation_loss(new_ckpt, root=self.root)
        verdict = compare_evals(old_eval, new_eval, old_suite=old_suite, new_suite=new_suite,
                                old_validation_loss=old_vl, new_validation_loss=new_vl)
        plan = self._load()
        if plan.get("status") != "evaluating":
            return  # operator stopped during eval; never promote over a stop
        plan["produced_checkpoint"] = new_ckpt  # re-assert: reload drops unsaved fields
        plan["eval"] = {"old_checkpoint": head_ckpt, "new_checkpoint": new_ckpt,
                        "report": str(eval_path), **verdict["scores"]}
        if suite_note:
            plan["eval"]["suite_note"] = suite_note
        if verdict["promote"] and suite_note is None:
            plan["best_candidate"] = {"checkpoint": new_ckpt,
                                      "run": plan["stages"][-1]["run_dir"].split("/")[-1],
                                      "eval": verdict["scores"], "report": str(eval_path)}
            plan["promotion"] = {"promoted": False, "status": "hold",
                                 "reason": verdict["reason"] +
                                 "; candidate passed automated checks and awaits manual promotion"}
        else:
            reason = verdict["reason"]
            if suite_note:
                reason += "; incomplete evaluation must HOLD"
            plan["promotion"] = {"promoted": False, "status": "hold", "reason": reason}
        plan["status"] = "done"
        self._save(plan)


def request_plan_stop(plan_id: str, root: Path = ROOT, ctx=None) -> dict:
    """Operator stop: persist first so a concurrent supervisor halts gracefully."""
    plan = read_plan(plan_id, root)
    if plan.get("status") not in ("running", "evaluating"):
        return plan
    plan["status"] = "stopped"
    plan["stop_reason"] = "stopped by operator; resume continues without replanning"
    touch(plan, root)
    if ctx is not None:
        for stage in plan["stages"]:
            if stage.get("status") == "running":
                try:
                    ctx.request_stop(root / stage["run_dir"])
                except OSError:
                    pass
    return read_plan(plan_id, root)


def plan_overview(root: Path = ROOT) -> dict:
    """Compact latest-plan status for /api/overview (never raises)."""
    try:
        plan = latest_plan(root)
    except Exception:
        return {"status": "none"}
    if plan is None:
        return {"status": "none"}
    stages = [{"run_dir": s["run_dir"], "status": s.get("status"),
               "target_tokens": s.get("target_tokens"),
               "tokens": s.get("tokens")} for s in plan.get("stages", [])]
    done_tokens = sum(s.get("tokens") or 0 for s in plan.get("stages", [])
                      if s.get("status") == "done")
    eval_report = plan.get("eval")
    if isinstance(eval_report, dict):
        problems = eval_report.get("hygiene_problems") or []
        eval_report = {**eval_report, "hygiene_problems": list(problems)[:8],
                       "hygiene_problem_count": len(problems)}
    return {"plan_id": plan["plan_id"], "status": plan.get("status"),
            "requested_tokens": plan.get("requested_tokens"),
            "effective_tokens": plan.get("effective_tokens"),
            "done_tokens": done_tokens,
            "stages": stages,
            "current_stage": next((s["run_dir"] for s in plan.get("stages", [])
                                   if s.get("status") == "running"), None),
            "produced_checkpoint": plan.get("produced_checkpoint"),
            "promotion": plan.get("promotion"),
            "eval": eval_report,
            "stop_reason": plan.get("stop_reason"),
            "failure": plan.get("failure"),
            "warnings": plan.get("warnings", [])}


def read_assistant_default(root: Path = ROOT) -> dict | None:
    """Promoted default assistant pointer, validated — None keeps legacy default."""
    pointer = read_json(Path(root) / "results" / "assistant_default.json", {})
    checkpoint = pointer.get("checkpoint")
    if not pointer or not checkpoint:
        return None
    if not (root / checkpoint).is_file() and not Path(checkpoint).is_file():
        return None
    return pointer


# ---------------------------------------------------------------------------
# Promotion gate: held-out eval comparison, never loss alone.
# ---------------------------------------------------------------------------

def _replies_by_kind(rows: list[dict]) -> dict:
    """Split replies into conversation vs instruction buckets (references kept)."""
    buckets = {"conv": [], "instr": []}
    for row in rows or []:
        if "turns" in row:
            for turn in row["turns"]:
                buckets["conv"].append({"reply": turn.get("reply", ""), "reference": None,
                                       "checks": (row.get("checks") or [{}])[0] if isinstance(row.get("checks"), list) else {}})
            continue
        bucket = "instr" if row.get("kind") == "instruction" else "conv"
        checks = row.get("checks") or {}
        buckets[bucket].append({"reply": row.get("reply", ""), "reference": row.get("reference"),
                                "checks": checks if isinstance(checks, dict) else {}})
    return buckets


def _hygiene_ok(entries: list[dict]) -> tuple[bool, list[str]]:
    problems = []
    for i, entry in enumerate(entries):
        checks = entry.get("checks") or {}
        reply = entry.get("reply", "")
        if not reply or not reply.strip():
            problems.append(f"reply {i}: empty")
        # Escape form on purpose: a literal U+FFFD here does not survive a
        # non-UTF-8 rewrite of this file, and silently degrades to " ", which
        # flags every reply that contains a space.
        if "�" in reply:
            problems.append(f"reply {i}: replacement character")
        if "User:" in reply:
            problems.append(f"reply {i}: prompt echo")
        for key in ("non_empty", "no_replacement_char", "no_prompt_echo", "no_runaway_repeat"):
            if key in checks and not checks[key]:
                problems.append(f"reply {i}: failed {key}")
        ratio = checks.get("printable_ratio")
        if isinstance(ratio, (int, float)) and ratio < 0.95:
            problems.append(f"reply {i}: printable ratio {ratio}")
    return (not problems, problems)


def _mean_f1(entries: list[dict]) -> tuple[float, int]:
    scores = [unigram_f1(e["reply"], e["reference"]) for e in entries if e.get("reference")]
    if not scores:
        return (0.0, 0)
    return (sum(scores) / len(scores), len(scores))


def compare_evals(old_rows: list[dict], new_rows: list[dict], *,
                  old_suite: dict | None = None, new_suite: dict | None = None,
                  old_validation_loss=None, new_validation_loss=None) -> dict:
    """Promotion verdict from two consistent held-out eval runs.

    Same decision criteria as the manual comparison (eval_full.scorecard):
    hygiene + both-skill F1 win-or-tie + capability-suite tolerance.
    Recorded validation losses are reported for context only — a lower loss
    alone never promotes. Suite arguments are optional so recorded probe-only
    evals (and existing unit fixtures) keep working; without a suite pair the
    gate is probes-only and says so in the scorecard note.
    """
    from .eval_full import CAPABILITY_TOLERANCE
    old, new = _replies_by_kind(old_rows), _replies_by_kind(new_rows)
    new_entries = new["conv"] + new["instr"]
    hygienic, problems = _hygiene_ok(new_entries)
    old_instr, old_instr_n = _mean_f1(old["instr"])
    new_instr, new_instr_n = _mean_f1(new["instr"])
    old_conv, old_conv_n = _mean_f1(old["conv"])
    new_conv, new_conv_n = _mean_f1(new["conv"])
    suite_delta = None
    if old_suite is not None and new_suite is not None:
        try:
            old_acc = float((old_suite or {}).get("accuracy") or 0)
            new_acc = float((new_suite or {}).get("accuracy") or 0)
            suite_delta = round(new_acc - old_acc, 4)
        except (TypeError, ValueError):
            suite_delta = None
    scores = {"instr_f1_old": round(old_instr, 4), "instr_f1_new": round(new_instr, 4),
              "instr_n": new_instr_n, "conv_f1_old": round(old_conv, 4),
              "conv_f1_new": round(new_conv, 4), "conv_n": new_conv_n,
              "hygiene_problems": problems,
              "suite_accuracy_old": (old_suite or {}).get("accuracy") if old_suite else None,
              "suite_accuracy_new": (new_suite or {}).get("accuracy") if new_suite else None,
              "suite_delta": suite_delta,
              "old_validation_loss": old_validation_loss,
              "new_validation_loss": new_validation_loss}
    if not hygienic:
        return {"promote": False, "scores": scores,
                "reason": f"held: {len(problems)} hygiene problem(s) in new replies ({problems[0]})"}
    if new_instr + PROMOTE_TOLERANCE < old_instr:
        return {"promote": False, "scores": scores,
                "reason": (f"held: instructions regressed "
                           f"(new F1 {new_instr:.3f} < old F1 {old_instr:.3f}); default unchanged")}
    if new_conv + PROMOTE_TOLERANCE < old_conv:
        return {"promote": False, "scores": scores,
                "reason": (f"held: conversation regressed "
                           f"(new F1 {new_conv:.3f} < old F1 {old_conv:.3f}); default unchanged")}
    if suite_delta is not None:
        old_acc = float((old_suite or {}).get("accuracy") or 0)
        new_acc = float((new_suite or {}).get("accuracy") or 0)
        if new_acc + CAPABILITY_TOLERANCE < old_acc:
            return {"promote": False, "scores": scores,
                    "reason": (f"held: general capability regressed "
                               f"({old_acc:.3f} → {new_acc:.3f}); default unchanged")}
    return {"promote": True, "scores": scores,
            "reason": (f"promoted: instructions F1 {old_instr:.3f}->{new_instr:.3f}, conversation F1 "
                       f"{old_conv:.3f}->{new_conv:.3f}, all hygiene checks passing")}
