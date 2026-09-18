"""Local AdamLM web dashboard.

The server deliberately uses only the Python standard library. It exposes the
existing planner and training entry point through a small, localhost-only API;
the browser never supplies a shell command.
"""
from __future__ import annotations

import errno
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import auto_train
from . import chatstore
from . import eval_full
from . import gpu_session
from . import research
from . import versions
from .gui_core import (
    ROOT,
    RunInfo,
    active_runs,
    build_plan,
    checkpoint_category,
    config_for_run,
    discover_runs,
    disk_telemetry,
    display_state,
    file_mtime,
    format_duration,
    format_number,
    gpu_telemetry,
    latest_checkpoint,
    load_metrics,
    read_json,
    request_stop,
    session_max_seconds,
    session_started_at,
    stop_pending,
    trainer_active,
    unique_stage_dir,
    validate_mixture,
    write_session,
)


WEB_ROOT = Path(__file__).with_name("web")
ALLOWED_DATASETS = {"tinystories", "wikitext103", "fineweb_edu", "dolly", "dailydialog", "mixture"}
SMOKE_MARKERS = ("smoke", "benchmark", "test", "web-ui", "web-stage", "gui-", "launcher-")
PRODUCTION_RUN = "bpe512-local-run"
MAX_BODY_BYTES = 64 * 1024

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Recent (timestamp, tokens) samples per run for a measured live throughput.
# In-memory only: after a backend restart there are no samples yet, so the
# dashboard reports throughput as unavailable until new samples arrive rather
# than inventing a number.
_rate_samples: dict[str, deque] = {}
_rate_lock = threading.Lock()


def _live_tokens_per_second(run: RunInfo) -> float | None:
    now = time.time()
    with _rate_lock:
        samples = _rate_samples.setdefault(run.name, deque(maxlen=8))
        samples.append((now, run.tokens))
        if len(samples) < 2:
            return None
        first_time, first_tokens = samples[0]
        elapsed = now - first_time
        if elapsed < 1.5:
            return None
        rate = (run.tokens - first_tokens) / elapsed
        return float(rate) if rate >= 0 else None


def _is_smoke(run: RunInfo) -> bool:
    name = run.name.lower()
    return any(marker in name for marker in SMOKE_MARKERS)


def _checkpoint_step(run: RunInfo) -> int:
    value = run.status.get("checkpoint_step") or run.status.get("step")
    if isinstance(value, int):
        return value
    if run.checkpoint:
        match = re.search(r"step_(\d+)", run.checkpoint.name)
        if match:
            return int(match.group(1))
    return 0


def _ordered_runs(runs: list[RunInfo]) -> list[RunInfo]:
    def key(run: RunInfo):
        return (
            0 if run.active else 1,
            1 if _is_smoke(run) else 0,
            -_checkpoint_step(run),
            -run.directory.stat().st_mtime,
        )

    return sorted(runs, key=key)


def _preferred_run(runs: list[RunInfo]) -> RunInfo | None:
    ordered = _ordered_runs(runs)
    production = [run for run in ordered if run.checkpoint and not _is_smoke(run)]
    return production[0] if production else next((run for run in ordered if run.checkpoint), None)


def _path_text(path: Path | None) -> str | None:
    return str(path) if path else None


def _run_json(run: RunInfo) -> dict:
    status = read_json(run.directory / "status.json", {})
    summary = read_json(run.directory / "summary.json", {})
    launcher = run.launcher
    target = run.target
    tokens = run.tokens
    progress = min(100, (tokens / target) * 100) if target else 0
    checkpoint_time = file_mtime(run.checkpoint) if run.checkpoint and run.checkpoint.is_file() else None
    status_time = file_mtime(run.directory / "status.json")
    historic = summary.get("end_to_end_tokens_per_second")
    historic_rate = float(historic) if isinstance(historic, (int, float)) and historic > 0 else None
    live_rate = _live_tokens_per_second(run) if run.active else None
    rate = live_rate if live_rate is not None else historic_rate
    return {
        "name": run.name,
        "directory": str(run.directory),
        "state": status.get("state", "unknown"),
        "display_state": display_state(run),
        "stage": launcher.get("stage") or status.get("stage") or "pretrain",
        "dataset": launcher.get("dataset") or status.get("dataset") or "tinystories",
        "mixture": launcher.get("mixture"),
        "parent_checkpoint": launcher.get("parent_checkpoint"),
        "step": status.get("step"),
        "tokens": tokens,
        "target_tokens": target,
        "remaining_tokens": max(0, target - tokens),
        "progress": round(progress, 1),
        "active": run.active,
        "stop_pending": stop_pending(run.directory),
        "is_smoke": _is_smoke(run),
        "is_production": run.name == PRODUCTION_RUN,
        "category": checkpoint_category(run, production_name=PRODUCTION_RUN, smoke_markers=SMOKE_MARKERS),
        "checkpoint": _path_text(run.checkpoint),
        "checkpoint_step": _checkpoint_step(run),
        "checkpoint_exists": bool(run.checkpoint and run.checkpoint.is_file()),
        "checkpoint_time": checkpoint_time,
        "status_time": status_time,
        "updated_at": status.get("updated_at"),
        "train_loss": status.get("train_loss"),
        "validation_loss": status.get("validation_loss"),
        "learning_rate": status.get("learning_rate"),
        "throughput": historic_rate,
        "live_throughput": live_rate,
        "eta": (max(0, target - tokens) / rate if rate and target else None),
        "session_max_seconds": session_max_seconds(run.directory),
        "session_started_at": session_started_at(run.directory),
    }


def _known_runs() -> list[RunInfo]:
    return _ordered_runs(discover_runs())


def _run_map() -> dict[str, RunInfo]:
    return {run.name: run for run in _known_runs()}


def _active_run(runs: list[RunInfo] | None = None) -> RunInfo | None:
    """The currently training run, rediscovered from the lock on every call.

    Independent of the dashboard's selected run, so a backend restart (which
    clears in-memory process handles) still finds a trainer launched earlier.
    """
    found = active_runs(runs)
    return found[0] if found else None


def _reconcile_gpu_session() -> None:
    """Drop a dead GPU claim (crash leftovers) without touching live owners.

    Stale means: training with no trainer lock behind it past the startup
    grace window, or inference/eval past its age lease. Live claims are
    never disturbed, so this is safe to call on every status poll.
    """
    try:
        gpu_session.force_release_stale(ROOT)
    except Exception:
        pass


def _require_gpu_free(action: str) -> None:
    _reconcile_gpu_session()
    live = _active_run()
    if live is not None:
        raise RuntimeError(
            f"Trainer '{live.name}' is already active. Stop it before starting another session.")
    owner = gpu_session.current(ROOT)
    if owner is not None:
        raise RuntimeError(
            f"GPU is busy ({owner.get('kind')}: {owner.get('label')}). "
            f"{action} refused so training and inference never compete for GPU memory.")


def _choose_stop_target(requested_name: str | None, runs: list[RunInfo]) -> tuple[RunInfo | None, RunInfo | None, bool]:
    """Return (target, active, already_stopped) for a graceful-stop request.

    The stop always targets the actual active trainer, even when the dashboard
    selection points at a different (parent or completed) run. ``already_stopped``
    is True when no trainer holds any lock, so callers can report the idle
    state accurately instead of a misleading error.
    """
    by_name = {run.name: run for run in runs}
    requested = by_name.get((requested_name or "").strip()) if requested_name else None
    active = _active_run(runs)
    if requested is not None and requested.active:
        return requested, active, False
    if active is not None:
        return active, active, False
    return None, None, True


def _session_json(run: RunInfo) -> dict:
    """Prominent active-session details with real metrics only.

    Any value that cannot be determined from local files is None so the UI
    renders it as unavailable instead of inventing it.
    """
    info = _run_json(run)
    now = time.time()
    started = info["session_started_at"]
    limit = info["session_max_seconds"]
    elapsed = (now - started) if started else None
    return {
        "run_name": run.name,
        "directory": info["directory"],
        "display_state": info["display_state"],
        "stage": info["stage"],
        "dataset": info["dataset"],
        "mixture": info["mixture"],
        "parent_checkpoint": info["parent_checkpoint"],
        "step": info["step"],
        "tokens": info["tokens"],
        "target_tokens": info["target_tokens"],
        "remaining_tokens": info["remaining_tokens"],
        "progress": info["progress"],
        "elapsed_seconds": elapsed,
        "session_max_seconds": limit,
        "session_remaining_seconds": (max(0, limit - elapsed) if (elapsed is not None and limit) else None),
        "live_throughput": info["live_throughput"],
        "throughput": info["throughput"],
        "eta": info["eta"],
        "train_loss": info["train_loss"],
        "validation_loss": info["validation_loss"],
        "learning_rate": info["learning_rate"],
        "checkpoint": info["checkpoint"],
        "checkpoint_step": info["checkpoint_step"],
        "checkpoint_exists": info["checkpoint_exists"],
        "checkpoint_time": info["checkpoint_time"],
        "updated_at": info["updated_at"],
        "stop_pending": info["stop_pending"],
    }


def _known_checkpoints(runs: list[RunInfo] | None = None) -> dict[str, Path]:
    return {run.name: run.checkpoint for run in (runs or _known_runs()) if run.checkpoint and run.checkpoint.is_file()}


def _resolve_run(name: str | None) -> RunInfo:
    run = _run_map().get((name or "").strip())
    if not run:
        raise ValueError("Select an existing run")
    return run


def _resolve_checkpoint(value: str | None) -> Path:
    requested = Path((value or "").strip()).resolve()
    known = {path.resolve() for path in _known_checkpoints().values()}
    if requested not in known or not requested.is_file():
        raise ValueError("Select a saved checkpoint from the dashboard")
    return requested


def _duration(payload: dict) -> float | None:
    if bool(payload.get("until_stopped")):
        return None
    try:
        value = float(payload.get("duration"))
    except (TypeError, ValueError):
        raise ValueError("Session duration must be a positive number") from None
    if value <= 0:
        raise ValueError("Session duration must be positive")
    unit = payload.get("duration_unit", "minutes")
    if unit not in {"minutes", "hours"}:
        raise ValueError("Choose minutes or hours")
    return value * (3600 if unit == "hours" else 60)


def _plan_from_payload(payload: dict):
    mode = payload.get("mode")
    if mode not in {"resume", "continuation"}:
        raise ValueError("Choose resume or continuation")
    dataset = payload.get("dataset")
    if dataset not in ALLOWED_DATASETS:
        raise ValueError("Choose a supported dataset")
    try:
        additional = int(str(payload.get("additional_tokens", "")).replace(",", ""))
    except (TypeError, ValueError):
        raise ValueError("Additional tokens must be a positive whole number") from None
    if additional <= 0:
        raise ValueError("Additional tokens must be positive")

    selected = _resolve_run(payload.get("run_name")) if mode == "resume" else None
    checkpoint = _resolve_checkpoint(payload.get("parent_checkpoint")) if mode == "continuation" else None
    stage_name = str(payload.get("stage_name", "")).strip()
    if mode == "continuation":
        if not stage_name:
            stage_name = unique_stage_dir(checkpoint.parent.parent.name).name
        if Path(stage_name).name != stage_name or stage_name in {".", ".."}:
            raise ValueError("Stage name must be one folder name")
        run_dir = ROOT / "results" / stage_name
    else:
        run_dir = None
    mixture = str(payload.get("mixture", "")).strip() or None
    if dataset == "mixture":
        if not mixture:
            raise ValueError("Enter mixture weights such as tinystories=0.2,wikitext103=0.8")
        validate_mixture(mixture)
    return build_plan(
        mode=mode,
        dataset=dataset,
        additional_tokens=additional,
        duration_seconds=_duration(payload),
        selected_run=selected,
        parent_checkpoint=checkpoint,
        mixture=mixture,
        run_dir=run_dir,
    )


def _plan_json(plan) -> dict:
    return {
        "mode": plan.mode,
        "run_dir": str(plan.run_dir),
        "dataset": plan.dataset,
        "checkpoint": _path_text(plan.checkpoint),
        "additional_tokens": plan.additional_tokens,
        "cumulative_tokens": plan.cumulative_tokens,
        "remaining_tokens": plan.remaining_tokens,
        "effective_target": plan.effective_target,
        "updates": plan.updates,
        "total_updates": plan.total_updates,
        "learning_rate": plan.learning_rate,
        "minimum_learning_rate": plan.minimum_learning_rate,
        "warmup_steps": plan.warmup_steps,
        "throughput": plan.throughput,
        "estimated_seconds": plan.estimated_seconds,
        "estimated_duration": format_duration(plan.estimated_seconds),
        "params": plan.params,
    }


def _payload(handler: BaseHTTPRequestHandler) -> dict:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        raise ValueError("Invalid request body") from None
    if length <= 0 or length > MAX_BODY_BYTES:
        raise ValueError("Request body is missing or too large")
    try:
        return json.loads(handler.rfile.read(length))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("Request body must be JSON") from None


def _session_limit_from_command(command: list[str]) -> float | None:
    if "--run-until-stopped" in command:
        return None
    try:
        index = command.index("--max-seconds")
        value = float(command[index + 1])
        return value if value > 0 else None
    except (ValueError, IndexError):
        return None


def _start_training(payload: dict) -> dict:
    plan = _plan_from_payload(payload)
    _require_gpu_free("Training start")
    if plan.mode == "continuation" and plan.run_dir.exists() and any(plan.run_dir.joinpath("checkpoints").glob("step_*.pt")):
        raise RuntimeError("That stage already has checkpoints. Choose Resume or a new stage name.")
    if plan.mode == "resume" and trainer_active(plan.run_dir):
        raise RuntimeError(f"Trainer '{plan.run_dir.name}' is already active. Stop it before starting another session.")
    claim = gpu_session.try_acquire(ROOT, kind="training", label=f"training:{plan.run_dir.name}",
                                      run=plan.run_dir.name)
    if not claim:
        live = _active_run()
        detail = f"Trainer '{live.name}' is already active. " if live else ""
        raise RuntimeError(f"{detail}GPU is busy. Stop the current session before starting another.")
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    log = (plan.run_dir / "web-training.log").open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            plan.command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        log.close()
        gpu_session.release(ROOT, kind="training", label=f"training:{plan.run_dir.name}")
        raise
    # Keep the handle so an operator can see whether this backend-launched process exited.
    with _jobs_lock:
        _jobs[f"trainer:{plan.run_dir.name}"] = {"process": process, "log": log}
    write_session(plan.run_dir, {
        "run_name": plan.run_dir.name,
        "mode": plan.mode,
        "dataset": plan.dataset,
        "started_at": time.time(),
        "max_seconds": _session_limit_from_command(plan.command),
        "target_tokens": plan.effective_target,
    })
    return {
        "plan": _plan_json(plan),
        "run_name": plan.run_dir.name,
        "run_dir": str(plan.run_dir),
        "message": f"Training started in {plan.run_dir.name}; watching that stage now. The backend keeps it running if the browser closes.",
    }


def _stop_training(payload: dict) -> dict:
    runs = _known_runs()
    requested_name = str(payload.get("run_name") or "").strip() or None
    target, active, already_stopped = _choose_stop_target(requested_name, runs)
    if already_stopped or target is None:
        if requested_name:
            try:
                run = _resolve_run(requested_name)
                info = _run_json(run)
                return {
                    "already_stopped": True,
                    "run_name": run.name,
                    "display_state": info["display_state"],
                    "checkpoint": info["checkpoint"],
                    "message": f"No active trainer. '{run.name}' is {info['display_state']}; checkpoints unchanged.",
                }
            except ValueError:
                pass
        return {
            "already_stopped": True,
            "run_name": None,
            "display_state": "idle",
            "checkpoint": None,
            "message": "No active trainer. Nothing was stopped; checkpoints unchanged.",
        }
    request_stop(target.directory)
    redirected = requested_name is not None and requested_name != target.name
    detail = (f" Note: selected run '{requested_name}' is not active; stopping the active session '{target.name}' instead."
              if redirected else "")
    return {
        "already_stopped": False,
        "run_name": target.name,
        "run_dir": str(target.directory),
        "display_state": "stopping",
        "message": f"Stopping — saving checkpoint for '{target.name}'. The trainer finishes its safe save before exiting.{detail}",
    }


_auto_threads: dict[str, threading.Thread] = {}


class _AutoCtx:
    """Wires the Auto Train supervisor to this backend process."""

    def active_trainer(self):
        return _active_run()

    def launch(self, command, cwd, log_path):
        run_name = Path(log_path).parent.name
        _reconcile_gpu_session()
        claim = gpu_session.try_acquire(ROOT, kind="training", label=f"training:auto:{run_name}",
                                        run=run_name)
        if not claim:
            owner = gpu_session.current(ROOT)
            detail = f" ({owner.get('kind')}: {owner.get('label')})" if owner else ""
            raise RuntimeError(f"GPU is busy{detail}; stage '{run_name}' not launched.")
        log = open(log_path, "a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [str(c) for c in command],
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            log.close()
            raise
        with _jobs_lock:
            _jobs[f"auto:{Path(log_path).parent.name}"] = {"process": process, "log": log}
        return process

    def read_stage_status(self, run_dir):
        return read_json(Path(run_dir) / "status.json", {})

    def latest_checkpoint(self, run_dir):
        return latest_checkpoint(Path(run_dir))

    def request_stop(self, run_dir):
        request_stop(Path(run_dir))

    def run_eval(self, old_checkpoint, new_checkpoint, out_path, timeout=1800):
        out_path = Path(out_path)
        command = [sys.executable, "scripts/eval_assistant.py",
                   f"old={old_checkpoint}", f"new={new_checkpoint}",
                   f"--out={out_path}"]
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"held-out evaluation timed out after {timeout}s") from exc
        if result.returncode != 0:
            raise RuntimeError(f"held-out evaluation failed: {(result.stderr or result.stdout).strip()[:300]}")
        try:
            entries = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"held-out evaluation wrote no usable report: {exc}") from exc
        by_label = {entry.get("label"): entry.get("rows", []) for entry in entries}
        if "old" not in by_label or "new" not in by_label:
            raise RuntimeError("held-out evaluation report is missing old/new results")
        return by_label["old"], by_label["new"]

    def sleep(self, seconds):
        time.sleep(seconds)

    def now(self):
        return time.time()


def _spawn_supervisor(plan_id: str) -> None:
    with _jobs_lock:
        thread = _auto_threads.get(plan_id)
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(target=auto_train.Supervisor(plan_id, ROOT, _AutoCtx()).run,
                                  daemon=True, name=f"auto-{plan_id}")
        _auto_threads[plan_id] = thread
        thread.start()


def _auto_inputs(payload: dict) -> tuple[int, float, str, int | None]:
    try:
        tokens = int(str(payload.get("additional_tokens", "")).replace(",", ""))
    except (TypeError, ValueError):
        raise ValueError("Additional tokens must be a positive whole number") from None
    if tokens <= 0:
        raise ValueError("Additional tokens must be positive")
    try:
        max_seconds = float(payload.get("max_seconds"))
    except (TypeError, ValueError):
        raise ValueError("Maximum session duration must be a positive number of seconds") from None
    if not max_seconds > 0:
        raise ValueError("Maximum session duration must be positive")
    pass_policy = str(payload.get("pass_policy") or "unique_once").strip().lower()
    if pass_policy not in {"unique_once", "repeat"}:
        raise ValueError("Dataset/pass policy must be unique_once or repeat")
    raw_stages = payload.get("stage_count")
    stage_count = None
    if raw_stages not in (None, "", "auto"):
        try:
            stage_count = int(str(raw_stages).replace(",", ""))
        except (TypeError, ValueError):
            raise ValueError("Stage count must be a positive whole number or blank for automatic") from None
        if stage_count <= 0:
            raise ValueError("Stage count must be positive")
    return tokens, max_seconds, pass_policy, stage_count


def _auto_plan(payload: dict) -> dict:
    tokens, max_seconds, pass_policy, stage_count = _auto_inputs(payload)
    return auto_train.preflight(tokens, max_seconds, ROOT, pass_policy=pass_policy,
                                stage_count=stage_count)


def _auto_start(payload: dict) -> dict:
    tokens, max_seconds, pass_policy, stage_count = _auto_inputs(payload)
    _require_gpu_free("Auto Train start")
    existing = auto_train.latest_plan(ROOT)
    if existing is not None and existing.get("status") == "running":
        raise RuntimeError(f"Auto Train plan '{existing['plan_id']}' is already running. Stop it first.")
    summary = auto_train.preflight(tokens, max_seconds, ROOT, pass_policy=pass_policy,
                                   stage_count=stage_count)
    if summary["budget"]["repeated_data"] and not bool(payload.get("repeat_confirm")):
        raise ValueError("Repeated data requires explicit confirmation before training starts")
    plan = {**summary, "status": "running", "created_at": time.time(),
            "started_at": None, "updated_at": time.time(),
            "stop_reason": None, "failure": None, "produced_checkpoint": None,
            "promotion": None, "eval": None}
    plan["stages"][0]["parent_checkpoint"] = plan["head"]["checkpoint"]
    for stage in plan["stages"][1:]:
        stage["parent_stage"] = stage["index"] - 1
    auto_train.write_plan(plan, ROOT)
    _spawn_supervisor(plan["plan_id"])
    return {"plan_id": plan["plan_id"], "plan": auto_train.plan_overview(ROOT),
            "message": f"Auto Train plan '{plan['plan_id']}' started; stages run sequentially under one budget and deadline."}


def _auto_stop() -> dict:
    plan = auto_train.latest_plan(ROOT)
    if plan is None or plan.get("status") not in ("running", "evaluating"):
        return {"already_stopped": True, "plan_id": plan["plan_id"] if plan else None,
                "message": "No Auto Train plan is running."}
    plan = auto_train.request_plan_stop(plan["plan_id"], ROOT, _AutoCtx())
    return {"already_stopped": False, "plan_id": plan["plan_id"],
            "plan": auto_train.plan_overview(ROOT),
            "message": f"Auto Train plan '{plan['plan_id']}' stopping — current stage saves gracefully."}


def _auto_resume(payload: dict) -> dict:
    plan = auto_train.latest_plan(ROOT)
    if plan is None:
        raise ValueError("No resumable Auto Train plan found.")
    if plan.get("status") in ("done", "failed", "regression"):
        checkpoint = plan.get("recovery_checkpoint") or next(
            (stage.get("checkpoint") for stage in reversed(plan.get("stages", []))
             if stage.get("checkpoint")), None)
        suffix = f" Last intact checkpoint: {checkpoint}." if checkpoint else ""
        raise ValueError(
            f"Auto Train plan '{plan['plan_id']}' is {plan.get('status')} and cannot resume in place."
            f"{suffix} Create a new continuation plan after addressing the recorded cause.")
    if plan.get("status") == "running":
        if _active_run() is not None:
            raise RuntimeError("An Auto Train plan is already running.")
        _spawn_supervisor(plan["plan_id"])  # adopt stale plan (e.g. after backend restart)
        return {"plan_id": plan["plan_id"], "plan": auto_train.plan_overview(ROOT),
                "message": f"Auto Train plan '{plan['plan_id']}' adopted and running."}
    if plan.get("status") not in ("stopped", "waiting", "time-exceeded"):
        raise ValueError(f"Auto Train plan '{plan['plan_id']}' is {plan.get('status')} and cannot resume.")
    if _active_run() is not None:
        live = _active_run()
        raise RuntimeError(f"Trainer '{live.name}' is already active. Stop it before resuming Auto Train.")
    raw_max = payload.get("max_seconds")
    max_seconds = plan["time"]["max_seconds"]
    if raw_max is not None:
        try:
            max_seconds = float(raw_max)
        except (TypeError, ValueError):
            raise ValueError("Maximum session duration must be a positive number of seconds") from None
        if not max_seconds > 0:
            raise ValueError("Maximum session duration must be positive")
    plan["status"] = "running"
    plan["started_at"] = time.time()  # fresh allowance, same token plan
    plan["time"]["max_seconds"] = max_seconds
    plan["time"]["max_duration"] = format_duration(max_seconds)
    plan["stop_reason"] = None
    plan["failure"] = None
    auto_train.touch(plan, ROOT)
    _spawn_supervisor(plan["plan_id"])
    return {"plan_id": plan["plan_id"], "plan": auto_train.plan_overview(ROOT),
            "message": f"Auto Train plan '{plan['plan_id']}' resumed without replanning."}


def _parse_playground_seed(value) -> int:
    """Playground-only seed: random by default, explicit when locked.

    Training and evaluation seeds are intentionally untouched — this helper
    applies only to normal Playground generation. ``None``/empty/``random``/
    ``auto`` means a fresh random seed; otherwise a whole number in
    ``[0, 2**32-1]`` is required so ``torch.Generator.manual_seed`` stays
    portable. Fixed seeds remain available for reproducible tests and fair
    checkpoint comparisons (e.g. Compare mode sends one explicit seed to
    both sides).
    """
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "random", "auto")):
        return secrets.randbits(31)
    try:
        seed = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("Seed must be a whole number in [0, 4294967295] or 'random'") from None
    if not 0 <= seed <= 4294967295:
        raise ValueError("Seed must be a whole number in [0, 4294967295] or 'random'")
    return seed


def _assistant_overview(runs: list[RunInfo]) -> dict:
    """Lineage-based assistant status for the Playground (never raises).

    The newest model is NOT picked by bare step number: step counters restart
    in every new stage and unrelated experiments share the same filename
    pattern. Instead the saved lineage (``sft-assistant*``/``auto-assistant*``
    SFT runs with real checkpoints) and promotion status decide:
    - ``default``: approved pointer from ``results/assistant_default.json`` (None = legacy ``sft-assistant`` fallback).
    - ``head``: most-trained lineage head by tokens (see ``auto_train.find_assistant_head``).
    - ``lineage``: all lineage candidates ordered by tokens desc for honest labeling.
    An unapproved head (head != default) must be labeled Experimental in the UI.
    """
    try:
        default = auto_train.read_assistant_default(ROOT)
    except Exception:
        default = None
    try:
        candidates = auto_train.assistant_runs(runs, ROOT)
    except Exception:
        candidates = []
    def _entry(run: RunInfo) -> dict:
        step = run.status.get("checkpoint_step") or run.status.get("step")
        if not isinstance(step, int) and run.checkpoint:
            match = re.search(r"step_(\d+)", run.checkpoint.name)
            step = int(match.group(1)) if match else 0
        return {"run": run.name, "checkpoint": str(run.checkpoint) if run.checkpoint else None,
                "tokens": run.tokens, "step": step or 0}
    lineage = sorted((_entry(r) for r in candidates), key=lambda e: (e["tokens"] or 0), reverse=True)
    head = lineage[0] if lineage else None
    return {"default": default, "head": head, "lineage": lineage}


def _start_generation(payload: dict, *, checkpoint_override: str | None = None,
                      gpu_kind: str = "inference") -> dict:
    _reconcile_gpu_session()
    owner = gpu_session.current(ROOT)
    if owner is not None:
        raise RuntimeError(
            f"GPU is busy ({owner.get('kind')}: {owner.get('label')}). Inference is paused while "
            "training/evaluation runs so the two never compete for GPU memory. Stop training or wait, then generate again.")
    live = _active_run()
    if live is not None:
        raise RuntimeError(
            f"Trainer '{live.name}' is active. Inference is paused while training runs "
            "so the two never compete for GPU memory. Stop training or wait for it to finish, then generate again.")
    checkpoint = _resolve_checkpoint(checkpoint_override if checkpoint_override else payload.get("checkpoint"))
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Enter a prompt")
    try:
        temperature = float(payload.get("temperature", 0.8))
        top_k = int(payload.get("top_k", 40))
        tokens = int(payload.get("tokens", 160))
    except (TypeError, ValueError):
        raise ValueError("Generation settings must be numeric") from None
    if temperature <= 0 or top_k <= 0 or tokens <= 0 or tokens > 4096:
        raise ValueError("Generation settings must be positive; new tokens must be at most 4096")
    seed = _parse_playground_seed(payload.get("seed", "random"))
    run_dir = checkpoint.parent.parent
    launcher = read_json(run_dir / "launcher-config.json", {})
    chat = bool(payload.get("chat"))
    if chat and launcher.get("stage") != "sft":
        raise ValueError("Conversation mode requires a checkpoint from an instruction-tuned SFT stage")
    job_id = uuid.uuid4().hex
    command = [
        sys.executable,
        "scripts/generate.py",
        str(checkpoint),
        "--prompt",
        prompt,
        "--tokens",
        str(tokens),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
        "--stream",
    ]
    if chat:
        command.append("--chat")

    gpu_label = f"{gpu_kind}:{job_id[:8]}"
    claim = gpu_session.try_acquire(ROOT, kind=gpu_kind, label=gpu_label)
    if not claim:
        owner = gpu_session.current(ROOT)
        detail = f" ({owner.get('kind')}: {owner.get('label')})" if owner else ""
        raise RuntimeError(f"GPU is busy{detail}; generation refused so workloads never overlap.")

    def _release_gpu_claim():
        try:
            gpu_session.release(ROOT, kind=gpu_kind, label=gpu_label)
        except Exception:
            pass

    def worker():
        proc = None
        try:
            proc = subprocess.Popen(
                [str(c) for c in command],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            with _jobs_lock:
                _jobs[job_id]["process"] = proc
            # Incremental stdout: generate.py --stream emits one JSON
            # {"text": ...} line per update; the last line wins. Anything
            # else (tracebacks on stderr path) is ignored until exit.
            latest, raw_tail = "", ""
            try:
                for line in proc.stdout:
                    raw_tail = (raw_tail + line)[-2000:]
                    try:
                        latest = json.loads(line).get("text", latest)
                    except (json.JSONDecodeError, AttributeError):
                        continue
                    with _jobs_lock:
                        if _jobs.get(job_id, {}).get("state") == "running":
                            _jobs[job_id]["output"] = latest
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                with _jobs_lock:
                    _jobs[job_id].update(state="error", output="Generation timed out after 600s.")
                _release_gpu_claim()
                return
            with _jobs_lock:
                job = _jobs.get(job_id, {})
                if job.get("state") == "cancelled":
                    _release_gpu_claim()
                    return
                if proc.returncode == 0:
                    _jobs[job_id].update(state="done", output=latest.strip() or (raw_tail.strip() or "No output returned."))
                else:
                    err = ""
                    try:
                        err = (proc.stderr.read() or "").strip()
                    except Exception:
                        err = ""
                    _jobs[job_id].update(state="error", output=(err or raw_tail).strip()[-2000:] or "Generation failed.")
            _release_gpu_claim()
        except Exception as exc:
            with _jobs_lock:
                if _jobs.get(job_id, {}).get("state") == "running":
                    _jobs[job_id].update(state="error", output=str(exc))
            _release_gpu_claim()

    with _jobs_lock:
        _jobs[job_id] = {"state": "running", "output": "", "seed": seed,
                         "checkpoint": str(checkpoint), "process": None,
                         "gpu_kind": gpu_kind, "gpu_label": gpu_label,
                         "started_at": time.time()}
    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "seed": seed}


def _cancel_generation(payload: dict) -> dict:
    job_id = str(payload.get("job_id", "")).strip()
    if not job_id:
        raise ValueError("Missing job_id")
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return {"job_id": job_id, "already_done": True,
                    "message": "Unknown generation job; nothing to cancel."}
        if job.get("state") != "running":
            return {"job_id": job_id, "already_done": True,
                    "message": f"Job is already {job.get('state')}."}
        job["state"] = "cancelled"
        proc = job.get("process")
        gpu_kind = job.get("gpu_kind", "inference")
        gpu_label = job.get("gpu_label")
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass
    if gpu_label:
        try:
            gpu_session.release(ROOT, kind=gpu_kind, label=gpu_label)
        except Exception:
            pass
    return {"job_id": job_id, "already_done": False,
            "message": "Generation cancelled. Conversation history unchanged."}


CHAT_BLOCK_TOKENS = 512
CHAT_MARGIN_TOKENS = 32
CHAT_CHARS_PER_TOKEN = 2


@lru_cache(maxsize=8)
def _chat_protocol(checkpoint: str | None):
    """Resolve the exact checkpoint tokenizer/context once per backend process."""
    from .bpe import BPETokenizer
    if checkpoint:
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        tokenizer_path = Path(state["extra"]["tokenizer_path"])
        block_size = int(state["model_config"]["block_size"])
        expected_hash = state["extra"].get("tokenizer_sha256")
        del state
    else:
        config = read_json(ROOT / "config" / "sft-assistant.json", {})
        tokenizer_path = Path(config["tokenizer"])
        block_size = int(config["model"]["block_size"])
        expected_hash = None
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path
    tokenizer = BPETokenizer(tokenizer_path)
    if expected_hash and tokenizer.sha256 != expected_hash:
        raise ValueError("Tokenizer hash does not match the approved checkpoint")
    return tokenizer, block_size


def _compose_chat_prompt(messages: list[dict], new_tokens: int,
                         checkpoint: str | Path | None = None) -> dict:
    from .inference import build_chat_prompt
    tokenizer, block_size = _chat_protocol(str(checkpoint) if checkpoint else None)
    return build_chat_prompt(messages, tokenizer, block_size, int(new_tokens))


def _start_chat(payload: dict) -> dict:
    """Chat with the ONE approved AdamLM. The caller never picks a checkpoint."""
    try:
        approved = versions.resolve_approved_checkpoint(ROOT)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Chat needs a non-empty messages array.")
    try:
        tokens = int(payload.get("tokens", 60))
    except (TypeError, ValueError):
        raise ValueError("Generation settings must be numeric") from None
    tokens = min(4096, max(1, tokens))
    composed = _compose_chat_prompt(messages, tokens, approved)
    job = _start_generation({
        "checkpoint": str(approved),
        "prompt": composed["prompt"],
        "chat": False,
        "temperature": payload.get("temperature", 0.8),
        "top_k": payload.get("top_k", 40),
        "tokens": tokens,
        "seed": payload.get("seed", "random"),
    }, checkpoint_override=str(approved))
    job.update({"kept_turns": composed["kept_turns"],
                "total_turns": composed["total_turns"],
                "trimmed": composed["trimmed"]})
    return job


_evals: dict[str, dict] = {}
_evals_lock = threading.Lock()


def _start_eval_compare(payload: dict) -> dict:
    old = str(payload.get("old") or payload.get("old_checkpoint") or "").strip()
    new = str(payload.get("new") or payload.get("new_checkpoint") or "").strip()
    if not old or not new:
        raise ValueError("Provide both 'old' and 'new' checkpoint paths.")
    for value in (old, new):
        _resolve_checkpoint(value)
    include_suite = bool(payload.get("include_suite", True))
    eval_id = uuid.uuid4().hex[:12]
    with _evals_lock:
        _evals[eval_id] = {"state": "running", "started_at": time.time(),
                           "old": old, "new": new, "report": None, "error": None}

    def worker():
        try:
            report = eval_full.compare_checkpoints(old, new, root=ROOT, device="cpu",
                                                   include_suite=include_suite)
            path = eval_full.save_report(report, root=ROOT, name=f"compare-{eval_id}.json")
            with _evals_lock:
                _evals[eval_id].update(state="done", report=str(path),
                                       scorecard=report["scorecard"],
                                       examples=report.get("examples", [])[:6])
        except Exception as exc:
            with _evals_lock:
                _evals[eval_id].update(state="error", error=str(exc))

    threading.Thread(target=worker, daemon=True).start()
    return {"eval_id": eval_id, "state": "running"}


def _promote_candidate(payload: dict) -> dict:
    checkpoint = str(payload.get("checkpoint") or "").strip()
    if not checkpoint:
        raise ValueError("Provide the candidate 'checkpoint' path.")
    _resolve_checkpoint(checkpoint)
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        raise ValueError("A promotion reason is required.")
    eval_info = payload.get("eval") if isinstance(payload.get("eval"), dict) else {}
    pointer = versions.promote_candidate(
        ROOT, checkpoint=checkpoint, eval_info=eval_info, reason=reason,
        plan_id=str(payload.get("plan_id") or "") or None,
        require_eval_win=bool(payload.get("require_eval_win", True)))
    versions.write_registry(ROOT)
    return {"promoted": True, "pointer": pointer}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "AdamLMWeb/1.0"

    def log_message(self, _format, *_args):
        return

    def _json(self, value: dict, status: int = 200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc: Exception, status: int = 400):
        self._json({"error": str(exc)}, status)

    def _local_origin(self) -> bool:
        origin = self.headers.get("Origin") or self.headers.get("Referer")
        if not origin:
            return True
        parsed = urlparse(origin)
        return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port == 8765

    def _file(self, path: Path, content_type: str):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self._file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            elif parsed.path in {"/styles.css", "/app.js"}:
                suffix = parsed.path.lstrip("/")
                self._file(WEB_ROOT / suffix, "text/css; charset=utf-8" if suffix.endswith("css") else "text/javascript; charset=utf-8")
            elif parsed.path == "/api/overview":
                runs = _known_runs()
                query = parse_qs(parsed.query)
                selected_name = query.get("run", [""])[0]
                selected = _run_map().get(selected_name) if selected_name else _preferred_run(runs)
                active = _active_run(runs)
                self._json({
                    "runs": [_run_json(run) for run in runs],
                    "selected_run": selected.name if selected else None,
                    "preferred_run": _preferred_run(runs).name if _preferred_run(runs) else None,
                    "active_run": active.name if active else None,
                    "active_session": _session_json(active) if active else None,
                    "checkpoints": [{"run": run.name, "path": str(run.checkpoint), "step": _checkpoint_step(run), "tokens": run.tokens, "is_smoke": _is_smoke(run), "is_production": run.name == PRODUCTION_RUN, "category": checkpoint_category(run, production_name=PRODUCTION_RUN, smoke_markers=SMOKE_MARKERS), "stage": run.launcher.get("stage") or run.status.get("stage") or "pretrain", "dataset": run.launcher.get("dataset") or run.status.get("dataset") or "tinystories"} for run in runs if run.checkpoint and run.checkpoint.is_file()],
                    "gpu": gpu_telemetry(),
                    "gpu_session": gpu_session.describe(ROOT),
                    "disk": disk_telemetry(),
                    "auto": auto_train.plan_overview(ROOT),
                    "assistant_default": auto_train.read_assistant_default(ROOT),
                    "assistant": _assistant_overview(runs),
                    "assistant_chat": versions.approved_summary(ROOT),
                    "research": research.session_overview(research.latest_session(ROOT)),
                    "provider": research.public_provider_config(ROOT),
                })
            elif parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif parsed.path == "/api/metrics":
                name = parse_qs(parsed.query).get("run", [""])[0]
                self._json({"run": name, "metrics": load_metrics(_resolve_run(name).directory)})
            elif parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                with _jobs_lock:
                    job = dict(_jobs.get(job_id, {}))
                job.pop("process", None)
                job.pop("log", None)
                self._json(job)
            elif parsed.path == "/api/assistant":
                runs = _known_runs()
                summary = versions.approved_summary(ROOT)
                summary["gpu_session"] = gpu_session.describe(ROOT)
                summary["active_run"] = (_active_run(runs).name if _active_run(runs) else None)
                self._json(summary)
            elif parsed.path == "/api/versions":
                self._json(versions.read_registry(ROOT))
            elif parsed.path == "/api/conversations":
                query = parse_qs(parsed.query)
                self._json({"conversations": chatstore.list_conversations(
                    ROOT, limit=int(query.get("limit", ["20"])[0] or 20))})
            elif parsed.path.startswith("/api/conversations/"):
                conversation_id = parsed.path.rsplit("/", 1)[-1]
                self._json(chatstore.load_conversation(ROOT, conversation_id))
            elif parsed.path == "/api/feedback":
                query = parse_qs(parsed.query)
                status = (query.get("status", [""])[0] or "").strip() or None
                self._json({"feedback": chatstore.list_feedback(ROOT, status=status)})
            elif parsed.path == "/api/evals":
                self._json({"reports": eval_full.list_reports(root=ROOT)})
            elif parsed.path.startswith("/api/eval/"):
                eval_id = parsed.path.rsplit("/", 1)[-1]
                with _evals_lock:
                    entry = dict(_evals.get(eval_id, {}))
                if not entry:
                    report_path = ROOT / "results" / "evals" / f"compare-{eval_id}.json"
                    if report_path.is_file():
                        report = eval_full.load_report(f"compare-{eval_id}.json", root=ROOT)
                        entry = {"state": "done", "report": str(report_path),
                                 "scorecard": report.get("scorecard"),
                                 "examples": report.get("examples", [])[:6]}
                    else:
                        raise ValueError(f"Evaluation '{eval_id}' was not found.")
                self._json(entry)
            elif parsed.path == "/api/research/inspect":
                self._json(research.inspect_project(ROOT))
            elif parsed.path == "/api/research/provider":
                self._json(research.public_provider_config(ROOT))
            elif parsed.path == "/api/research/latest":
                self._json(research.session_overview(research.latest_session(ROOT)))
            elif parsed.path.startswith("/api/research/report/"):
                session_id = parsed.path.rsplit("/", 1)[-1]
                session = research.read_session(session_id, ROOT)
                report_path = ROOT / "results" / "research" / f"{session_id}.report.md"
                try:
                    text = report_path.read_text(encoding="utf-8")
                except OSError:
                    text = ""
                self._json({"session_id": session_id, "report": text,
                            "experiments": session.get("experiments", [])})
            elif parsed.path.startswith("/api/research/"):
                session_id = parsed.path.rsplit("/", 1)[-1]
                self._json(research.session_overview(research.read_session(session_id, ROOT)))
            else:
                self._error(ValueError("Not found"), 404)
        except Exception as exc:
            self._error(exc, 500)

    def do_POST(self):
        if not self._local_origin():
            self._error(ValueError("State-changing requests must originate from the local dashboard"), 403)
            return
        parsed = urlparse(self.path)
        try:
            payload = _payload(self)
            if parsed.path == "/api/plan":
                self._json({"plan": _plan_json(_plan_from_payload(payload))})
            elif parsed.path == "/api/training/start":
                self._json(_start_training(payload), 202)
            elif parsed.path == "/api/training/stop":
                self._json(_stop_training(payload))
            elif parsed.path == "/api/generate":
                self._json(_start_generation(payload), 202)
            elif parsed.path == "/api/generate/cancel":
                self._json(_cancel_generation(payload))
            elif parsed.path == "/api/chat":
                self._json(_start_chat(payload), 202)
            elif parsed.path == "/api/assistant/rollback":
                self._json(versions.rollback_to_previous(
                    ROOT, confirm=bool(payload.get("confirm"))))
            elif parsed.path == "/api/assistant/promote":
                self._json(_promote_candidate(payload))
            elif parsed.path == "/api/versions/archive":
                self._json(versions.set_archived(
                    ROOT, version_id=str(payload.get("version_id") or ""),
                    archived=bool(payload.get("archived", True))))
            elif parsed.path == "/api/conversations":
                self._json(chatstore.save_conversation(
                    ROOT, messages=payload.get("messages"),
                    title=str(payload.get("title") or ""),
                    approved_version=str(payload.get("approved_version") or "") or None,
                    conversation_id=str(payload.get("conversation_id") or "") or None))
            elif parsed.path == "/api/conversations/delete":
                self._json(chatstore.delete_conversation(
                    ROOT, str(payload.get("conversation_id") or "")))
            elif parsed.path == "/api/feedback":
                self._json(chatstore.submit_feedback(
                    ROOT, conversation_id=str(payload.get("conversation_id") or ""),
                    assistant_text=str(payload.get("assistant_text") or ""),
                    corrected_text=str(payload.get("corrected_text") or ""),
                    note=str(payload.get("note") or "")))
            elif parsed.path == "/api/feedback/review":
                self._json(chatstore.review_feedback(
                    ROOT, feedback_id=str(payload.get("feedback_id") or ""),
                    approve=bool(payload.get("approve"))))
            elif parsed.path == "/api/eval/compare":
                self._json(_start_eval_compare(payload), 202)
            elif parsed.path == "/api/research/sessions":
                try:
                    max_tokens = int(str(payload.get("max_tokens", "")).replace(",", ""))
                except (TypeError, ValueError):
                    raise ValueError("Maximum total training tokens must be a positive whole number") from None
                try:
                    max_seconds = float(payload.get("max_seconds", 0))
                except (TypeError, ValueError):
                    raise ValueError("Maximum overall session duration must be a positive number of seconds") from None
                raw_datasets = payload.get("allowed_datasets")
                if isinstance(raw_datasets, str):
                    raw_datasets = [p.strip() for p in raw_datasets.replace(",", " ").split()]
                try:
                    stage_cap = int(str(payload.get("stage_token_cap", "1000000") or "1000000").replace(",", ""))
                except (TypeError, ValueError):
                    raise ValueError("Per-stage token cap must be a positive whole number") from None
                session = research.new_session(
                    ROOT, max_tokens=max_tokens, max_seconds=max_seconds,
                    max_experiments=int(payload.get("max_experiments", 3) or 3),
                    max_api_requests=int(payload.get("max_api_requests", 25) or 25),
                    max_api_cost_usd=float(payload.get("max_api_cost_usd", 0.0) or 0.0),
                    max_repetition=int(payload.get("max_repetition", 3) or 3),
                    allow_external_eval=bool(payload.get("allow_external_eval", False)),
                    require_manual_promotion=bool(payload.get("require_manual_promotion", True)),
                    allowed_ops=payload.get("allowed_ops") or ["train", "evaluate", "compare"],
                    use_ai=bool(payload.get("use_ai", False)),
                    goal=str(payload.get("goal") or ""),
                    allowed_datasets=raw_datasets,
                    stage_token_cap=stage_cap,
                    allow_repetition=bool(payload.get("allow_repetition", False)),
                    mixture=str(payload.get("mixture") or ""))
                self._json({"session_id": session["session_id"],
                            "session": research.session_overview(session)})
            elif parsed.path == "/api/research/preview":
                try:
                    max_tokens = int(str(payload.get("max_tokens", "")).replace(",", ""))
                except (TypeError, ValueError):
                    raise ValueError("Maximum total training tokens must be a positive whole number") from None
                try:
                    max_seconds = float(payload.get("max_seconds", 0))
                except (TypeError, ValueError):
                    raise ValueError("Maximum overall session duration must be a positive number of seconds") from None
                raw_datasets = payload.get("allowed_datasets")
                if isinstance(raw_datasets, str):
                    raw_datasets = [p.strip() for p in raw_datasets.replace(",", " ").split()]
                try:
                    stage_cap = int(str(payload.get("stage_token_cap", "1000000") or "1000000").replace(",", ""))
                except (TypeError, ValueError):
                    raise ValueError("Per-stage token cap must be a positive whole number") from None
                # Pure preflight: creates no session, contacts no provider, starts nothing.
                self._json({"preview": research.preview_research(
                    ROOT, max_tokens=max_tokens, max_seconds=max_seconds,
                    max_experiments=int(payload.get("max_experiments", 3) or 3),
                    allowed_datasets=raw_datasets, stage_token_cap=stage_cap,
                    allow_repetition=bool(payload.get("allow_repetition", False)),
                    mixture=str(payload.get("mixture") or ""),
                    use_ai=bool(payload.get("use_ai", False)),
                    goal=str(payload.get("goal") or ""))})
            elif parsed.path == "/api/research/start":
                self._json(research.start_session(str(payload.get("session_id") or ""), ROOT))
            elif parsed.path == "/api/research/stop":
                self._json(research.stop_session(str(payload.get("session_id") or ""), ROOT))
            elif parsed.path == "/api/research/provider":
                self._json(research.write_provider_config(ROOT, payload))
            elif parsed.path == "/api/research/provider/key":
                self._json(research.store_api_key(ROOT, str(payload.get("key") or "")))
            elif parsed.path == "/api/research/provider/key/clear":
                self._json(research.clear_api_key(ROOT))
            elif parsed.path == "/api/research/provider/test":
                self._json(research.test_provider_connection(ROOT))
            elif parsed.path == "/api/auto/plan":
                self._json({"preflight": _auto_plan(payload)})
            elif parsed.path == "/api/auto/start":
                self._json(_auto_start(payload), 202)
            elif parsed.path == "/api/auto/stop":
                self._json(_auto_stop())
            elif parsed.path == "/api/auto/resume":
                self._json(_auto_resume(payload), 202)
            else:
                self._error(ValueError("Not found"), 404)
        except RuntimeError as exc:
            self._error(exc, 409)
        except ValueError as exc:
            self._error(exc, 400)
        except Exception as exc:
            self._error(exc, 500)


HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}/"


class _SingleInstanceServer(ThreadingHTTPServer):
    """One dashboard per machine.

    ThreadingHTTPServer inherits allow_reuse_address = 1, which on Windows sets
    SO_REUSEADDR and lets a second process bind a port that is already serving.
    The result is two dashboards on 8765 with undefined request routing -- and,
    because each holds its own Auto Train supervisor threads, two schedulers
    competing for the same plans and the same GPU claim. Refusing the reuse
    turns that into a clean, catchable bind error.
    """

    allow_reuse_address = False


def _dashboard_already_running() -> bool:
    """True when this project's dashboard already answers on the port."""
    try:
        with socket.create_connection((HOST, PORT), timeout=0.5):
            pass
    except OSError:
        return False
    # The port is taken; confirm it is our dashboard and not another service.
    try:
        with urllib.request.urlopen(f"{URL}api/overview", timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _report_existing() -> None:
    print(f"AdamLM dashboard is already running at {URL}", flush=True)
    print("Not starting a second one -- open that address to use it.", flush=True)
    print("To stop it: close its dashboard window, or end the 'adamlm.web' process.", flush=True)
    print("Training is separate: stop-training.cmd stops a run, the dashboard can stay open.",
          flush=True)


def serve() -> None:
    if _dashboard_already_running():
        _report_existing()
        return
    try:
        server = _SingleInstanceServer((HOST, PORT), DashboardHandler)
    except OSError as exc:
        # Lost the race against another launch, or something else holds the port.
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            _report_existing()
            return
        raise
    print(f"AdamLM dashboard running at {URL}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAdamLM dashboard stopped. Any trainer subprocess may continue; inspect status.cmd.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
