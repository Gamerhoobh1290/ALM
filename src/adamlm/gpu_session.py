"""Shared exclusive GPU-session ownership for training, inference, and eval.

All local GPU users — manual training, Auto Train, the Research Agent, the
Chat/Playground inference jobs, and automated evaluation — coordinate through
one small session file so two GPU workloads can never run at once.

Design rules (learned the hard way on Windows):
- Only non-blocking primitives: ``FileLock(..., timeout=0)`` single-attempt
  probes plus plain file reads/writes. This module never waits on a lock and
  never calls ``os.kill`` (repeated liveness probes of dead PIDs wedge this
  machine's process-handle path and hang the caller inside the C call).
- Training liveness comes from the trainer's own run lock
  (``gui_core.trainer_active`` — the established mechanism), not from PIDs.
- Inference jobs cannot outlive their hard timeout (600s), so an inference
  claim carries an age cap instead of a heartbeat.
- A short startup-grace window covers the backend -> trainer handoff: the
  backend pre-claims, the trainer adopts within two minutes.

Claim file ``results/.gpu-session.json``: ``{kind, label, run, started_at}``.
Lock file ``results/.gpu-session.lock`` guards the check-and-claim sequence
(single attempt; a failed attempt means busy, never a wait).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from filelock import FileLock, Timeout

SESSION_FILE_NAME = ".gpu-session.json"
LOCK_FILE_NAME = ".gpu-session.lock"

VALID_KINDS = ("training", "inference", "eval")

# Backend pre-claim -> trainer adopt window (trainer rewrites on startup).
TRAINING_GRACE_SECONDS = 120.0
# Generation workers are killed after 600s, so an older inference claim is
# definitionally dead — no heartbeat or PID check required.
INFERENCE_LEASE_SECONDS = 660.0
# Eval claims use the same age-cap reasoning with a longer bound.
EVAL_LEASE_SECONDS = 4 * 3600.0


def _paths(root) -> tuple[Path, Path]:
    root = Path(root)
    return root / "results" / SESSION_FILE_NAME, root / "results" / LOCK_FILE_NAME


def read_session(root) -> dict | None:
    session_path, _ = _paths(root)
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("kind") not in VALID_KINDS:
        return None
    try:
        started = float(data.get("started_at") or 0)
    except (TypeError, ValueError):
        return None
    if started <= 0:
        return None
    return data


def _training_run_active(root, session: dict) -> bool:
    """Whether the claimed training run still holds its trainer lock."""
    from .gui_core import trainer_active

    run = session.get("run")
    if not run:
        return False
    try:
        return bool(trainer_active(Path(root) / "results" / str(run)))
    except OSError:
        return False


def _claim_alive(root, session: dict | None, *, now: float | None = None) -> bool:
    """Conservative liveness without any PID or blocking syscall."""
    if not session:
        return False
    now = time.time() if now is None else now
    try:
        age = now - float(session.get("started_at") or 0)
    except (TypeError, ValueError):
        return False
    kind = session.get("kind")
    if kind == "training":
        if _training_run_active(root, session):
            return True
        # Startup grace: backend pre-claimed, trainer adopting right now.
        return age < TRAINING_GRACE_SECONDS
    if kind == "inference":
        return age < INFERENCE_LEASE_SECONDS
    if kind == "eval":
        return age < EVAL_LEASE_SECONDS
    return False


def current(root) -> dict | None:
    """Live owner, or None when the GPU is free (stale claims are ignored)."""
    try:
        session = read_session(root)
    except Exception:
        return None
    try:
        if session and _claim_alive(root, session):
            return session
    except Exception:
        return None
    return None


def is_busy(root) -> bool:
    return current(root) is not None


def _write_claim(session_path: Path, claim: dict) -> None:
    tmp = session_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(claim, indent=2), encoding="utf-8")
    os.replace(tmp, session_path)


def try_acquire(root, *, kind: str, label: str, run: str | None = None) -> dict | None:
    """Single-attempt claim. Returns the claim, or None when busy.

    The check and the write happen inside one non-blocking lock attempt, so
    concurrent claimants cannot both win — and a contender never waits.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"GPU session kind must be one of {VALID_KINDS}")
    root = Path(root)
    session_path, lock_path = _paths(root)
    session_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock_path), timeout=0):
            existing = read_session(root)
            if existing and _claim_alive(root, existing):
                return None
            claim = {
                "kind": kind,
                "label": str(label or kind),
                "run": str(run) if run else None,
                "started_at": time.time(),
            }
            _write_claim(session_path, claim)
            return claim
    except Timeout:
        return None


def adopt(root, *, kind: str, label: str, run: str | None = None) -> dict | None:
    """Trainer startup: take over a matching pre-claim or claim when free.

    Adoption succeeds when there is no live foreign claim: a stale/absent
    claim, the grace window, or a claim already naming this run/label.
    A live claim for a different run/label refuses (returns None) so two
    trainers can never share the GPU.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"GPU session kind must be one of {VALID_KINDS}")
    root = Path(root)
    session_path, lock_path = _paths(root)
    session_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock_path), timeout=0):
            existing = read_session(root)
            if existing and _claim_alive(root, existing):
                same_run = bool(run) and existing.get("run") == str(run)
                same_label = bool(label) and existing.get("label") == str(label)
                if not (same_run or same_label):
                    return None
            claim = {
                "kind": kind,
                "label": str(label),
                "run": str(run) if run else (existing or {}).get("run"),
                "started_at": time.time(),
            }
            _write_claim(session_path, claim)
            return claim
    except Timeout:
        return None


def release(root, *, kind: str | None = None, label: str | None = None,
            run: str | None = None) -> bool:
    """Release a claim we hold (matched by label/run when given)."""
    root = Path(root)
    session_path, lock_path = _paths(root)
    try:
        with FileLock(str(lock_path), timeout=0):
            existing = read_session(root)
            if not existing:
                return True
            if kind and existing.get("kind") != kind:
                return False
            if label and existing.get("label") != label:
                return False
            if run and existing.get("run") != str(run):
                return False
            try:
                session_path.unlink()
            except OSError:
                return False
            return True
    except Timeout:
        return False


def force_release_stale(root) -> bool:
    """Drop the claim file when it is not live. Never blocks, never waits."""
    root = Path(root)
    session_path, lock_path = _paths(root)
    try:
        with FileLock(str(lock_path), timeout=0):
            existing = read_session(root)
            if existing and _claim_alive(root, existing):
                return False
            try:
                session_path.unlink()
            except OSError:
                pass
            return True
    except Timeout:
        return False


def describe(root) -> dict:
    """Small status object for /api/overview (never raises, never blocks)."""
    try:
        owner = current(root)
    except Exception:
        owner = None
    if not owner:
        return {"busy": False, "kind": None, "label": None, "started_at": None}
    return {
        "busy": True,
        "kind": owner.get("kind"),
        "label": owner.get("label"),
        "started_at": owner.get("started_at"),
    }
