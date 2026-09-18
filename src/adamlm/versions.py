"""ONE AdamLM version registry: approved pointer, lineage, and history.

Only ONE checkpoint represents AdamLM in ordinary Chat at a time — the
approved pointer in ``results/assistant_default.json``. Everything else
(history, candidates, experiments, foundation, smoke tests) is indexed
non-destructively in ``results/versions.json`` so the UI can show an honest
timeline without guessing ancestry.

Approval states:
- ``approved``: the current Chat model (exactly one).
- ``previous-approved``: last approved model, kept for rollback.
- ``candidate``: produced by a recent plan/session, awaiting a decision.
- ``experimental``: any other assistant-lineage checkpoint.
- ``foundation``: the frozen general-pretraining base (never Chat-selected).
- ``general``: other general-pretraining checkpoints.
- ``smoke``: smoke-test / benchmark artifacts.

Ancestry is recorded from launcher configs, checkpoint
``extra.source_transition`` (when already cached — never by loading large
torch files during a scan), and auto/research plan files. When ancestry
cannot be verified it is reported as ``unknown``, never guessed.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from . import gui_core
from .gui_core import ROOT, RunInfo, checkpoint_category, discover_runs, read_json

REGISTRY_NAME = "versions.json"
POINTER_NAME = "assistant_default.json"
FOUNDATION_RUN = "general-training-500m"
ASSISTANT_PREFIXES = ("sft-assistant", "auto-assistant")
SMOKE_MARKERS = ("smoke", "benchmark", "test", "web-ui", "web-stage", "gui-", "launcher-")

_STEP_RE = re.compile(r"step_(\d+)")


def _step_of(path: Path | None, status: dict) -> int:
    value = (status or {}).get("checkpoint_step")
    if isinstance(value, int):
        return value
    value = (status or {}).get("step")
    if isinstance(value, int):
        return value
    if path:
        match = _STEP_RE.search(path.name)
        if match:
            return int(match.group(1))
    return 0


def _friendly_version_id(run_name: str, step: int) -> str:
    return f"{run_name}@step-{step:08d}"


def _parent_of_run(run: RunInfo) -> dict:
    """Best-effort parent reference from on-disk metadata (no torch loads)."""
    launcher_parent = (run.launcher or {}).get("parent_checkpoint")
    parent_run = None
    if launcher_parent:
        try:
            text = str(launcher_parent).replace("\\", "/")
            parts = text.split("/")
            if "results" in parts:
                idx = parts.index("results")
                if idx + 1 < len(parts):
                    parent_run = parts[idx + 1]
        except Exception:
            parent_run = None
    return {
        "parent_checkpoint": str(launcher_parent) if launcher_parent else None,
        "parent_run": parent_run,
        "verified": bool(launcher_parent),
    }


def _plan_parent_map(root: Path) -> dict[str, dict]:
    """Map produced run_dir -> {plan_id, parent info} from auto/research plans."""
    mapping: dict[str, dict] = {}
    results = Path(root) / "results"
    if not results.exists():
        return mapping
    for path in results.glob("*.auto.json"):
        plan = read_json(path, {})
        for stage in plan.get("stages", []) or []:
            run_dir = str(stage.get("run_dir") or "")
            name = run_dir.split("/")[-1].split("\\")[-1]
            if name:
                mapping[name] = {
                    "plan_id": plan.get("plan_id"),
                    "parent_checkpoint": stage.get("parent_checkpoint"),
                    "parent_stage": stage.get("parent_stage"),
                }
    research_dir = results / "research"
    if research_dir.exists():
        for path in research_dir.glob("*.json"):
            if path.name.endswith(".report.json"):
                continue
            session = read_json(path, {})
            for exp in session.get("experiments", []) or []:
                name = str(exp.get("run_dir") or "").split("/")[-1].split("\\")[-1]
                if name:
                    mapping[name] = {
                        "plan_id": session.get("session_id"),
                        "parent_checkpoint": exp.get("parent_checkpoint"),
                        "parent_stage": None,
                    }
    return mapping


def build_registry(root: Path = ROOT) -> dict:
    """Scan all runs non-destructively and return the registry object."""
    root = Path(root)
    runs = discover_runs(root)
    try:
        from . import auto_train

        default = auto_train.read_assistant_default(root)
    except Exception:
        default = read_json(root / "results" / POINTER_NAME, {}) or None
        if default and not default.get("checkpoint"):
            default = None
    approved_path = str((default or {}).get("checkpoint") or "")
    previous_path = str((((default or {}).get("previous") or {}).get("checkpoint")) or "")
    plan_parents = _plan_parent_map(root)

    # Newest experimental head by tokens (assistant lineage only).
    head_path = None
    try:
        from . import auto_train as _at

        head = _at.find_assistant_head(runs, root)
        head_path = str(head.checkpoint) if head.checkpoint else None
    except Exception:
        head_path = None

    versions: list[dict] = []
    for run in runs:
        if not run.checkpoint:
            continue
        step = _step_of(run.checkpoint, run.status)
        category = checkpoint_category(run)
        parent = _parent_of_run(run)
        plan_info = plan_parents.get(run.name, {})
        if plan_info.get("parent_checkpoint") and not parent["parent_checkpoint"]:
            parent = {
                "parent_checkpoint": plan_info.get("parent_checkpoint"),
                "parent_run": None,
                "verified": False,
            }
        ancestry = []
        if parent.get("parent_run"):
            ancestry = [parent["parent_run"]]
        checkpoint_str = str(run.checkpoint)
        if checkpoint_str == approved_path:
            approval = "approved"
        elif checkpoint_str == previous_path:
            approval = "previous-approved"
        elif run.name == FOUNDATION_RUN:
            approval = "foundation"
        elif category == "smoke-test":
            approval = "smoke"
        elif run.name.startswith(ASSISTANT_PREFIXES) and checkpoint_str == head_path:
            approval = "candidate" if approved_path and head_path != approved_path else "approved"
        elif run.name.startswith(ASSISTANT_PREFIXES):
            approval = "experimental"
        elif category == "general-pretraining":
            approval = "general"
        else:
            approval = "experimental"
        try:
            stat = run.checkpoint.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            size, mtime = None, None
        versions.append(
            {
                "version_id": _friendly_version_id(run.name, step),
                "label": _label_for(run.name, approval, step),
                "run": run.name,
                "checkpoint": checkpoint_str,
                "step": step,
                "tokens": run.tokens,
                "target_tokens": run.target,
                "stage": run.launcher.get("stage") or run.status.get("stage") or "pretrain",
                "dataset": run.launcher.get("dataset") or run.status.get("dataset") or "tinystories",
                "mixture": run.launcher.get("mixture"),
                "category": category,
                "approval": approval,
                "parent_checkpoint": parent["parent_checkpoint"],
                "parent_run": parent["parent_run"],
                "ancestry": ancestry or (["unknown"] if approval in ("experimental", "candidate", "general") else []),
                "ancestry_verified": bool(parent["verified"]),
                "plan_id": plan_info.get("plan_id"),
                "active": bool(run.active),
                "display_state": gui_core.display_state(run),
                "checkpoint_bytes": size,
                "created_at": mtime,
                "archived": False,  # reconciled with stored flags by write/read_registry
            }
        )

    # Order: approved first, then previous, candidate, lineage, foundation, rest.
    order = {"approved": 0, "previous-approved": 1, "candidate": 2, "experimental": 3,
             "general": 4, "foundation": 5, "smoke": 6}
    versions.sort(key=lambda v: (order.get(v["approval"], 9), -(v["tokens"] or 0)))

    # Assistant timeline: approved ancestry chain first, no false straight lines.
    timeline = [v for v in versions if v["approval"] in ("approved", "previous-approved", "candidate", "experimental")]
    branches = [v for v in versions if v["approval"] in ("general", "foundation")]
    smoke = [v for v in versions if v["approval"] == "smoke"]
    registry = {
        "format_version": 1,
        "updated_at": time.time(),
        "approved": default,
        "count": len(versions),
        "versions": versions,
        "timeline": [v["version_id"] for v in timeline],
        "branches": [v["version_id"] for v in branches],
        "smoke": [v["version_id"] for v in smoke],
        "archived": [],
    }
    return registry


def _label_for(run_name: str, approval: str, step: int) -> str:
    base = f"{run_name} · step {step:08d}"
    tags = {
        "approved": "AdamLM · Approved",
        "previous-approved": "AdamLM · Previous approved",
        "candidate": "Candidate · awaiting decision",
        "experimental": "Experimental branch",
        "foundation": "General-language foundation",
        "general": "General pretraining",
        "smoke": "Smoke test",
    }
    return f"{tags.get(approval, approval)} — {base}"


def _read_archived(root: Path) -> set[str]:
    """Version IDs hidden from the default view. Hide-only: files stay on disk."""
    data = read_json(Path(root) / "results" / REGISTRY_NAME, {})
    stored = (data or {}).get("archived") or []
    return {str(v) for v in stored if str(v or "").strip()}


def set_archived(root: Path = ROOT, *, version_id: str, archived: bool = True) -> dict:
    """Archive (hide) or unarchive a version. Never deletes checkpoints."""
    root = Path(root)
    version_id = str(version_id or "").strip()
    if not version_id:
        raise ValueError("Provide a version_id to archive.")
    registry = build_registry(root)
    known = {v["version_id"] for v in registry.get("versions", [])}
    if version_id not in known:
        raise ValueError(f"Unknown version '{version_id}'; nothing archived.")
    # Never hide the approved or previous-approved pointer targets.
    entry = next(v for v in registry["versions"] if v["version_id"] == version_id)
    if entry.get("approval") in ("approved", "previous-approved"):
        raise ValueError(f"Refusing to archive the {entry['approval']} version; it stays visible.")
    flags = _read_archived(root)
    if archived:
        flags.add(version_id)
    else:
        flags.discard(version_id)
    registry["archived"] = sorted(flags)
    for version in registry["versions"]:
        version["archived"] = version["version_id"] in flags
    gui_core.atomic_json(root / "results" / REGISTRY_NAME, registry)
    return {"version_id": version_id, "archived": version_id in flags}


def write_registry(root: Path = ROOT) -> dict:
    registry = build_registry(root)
    flags = _read_archived(root)
    # Drop flags for versions that no longer exist; keep everything else.
    known = {v["version_id"] for v in registry.get("versions", [])}
    flags &= known
    registry["archived"] = sorted(flags)
    for version in registry["versions"]:
        version["archived"] = version["version_id"] in flags
    path = Path(root) / "results" / REGISTRY_NAME
    gui_core.atomic_json(path, registry)
    return registry


def read_registry(root: Path = ROOT) -> dict:
    data = read_json(Path(root) / "results" / REGISTRY_NAME, {})
    if data and data.get("format_version") == 1:
        flags = {str(v) for v in (data.get("archived") or []) if str(v or "").strip()}
        data["archived"] = sorted(flags)
        for version in data.get("versions", []) or []:
            version["archived"] = version.get("version_id") in flags
        return data
    # No usable cache: build in memory without writing. (When the file is
    # missing there are no stored archive flags to preserve, so the build
    # is exact.) Explicit writes happen only via write_registry / set_archived.
    return build_registry(root)


def resolve_approved_checkpoint(root: Path = ROOT) -> Path:
    """The single checkpoint Chat must load. Raises with a clear message."""
    root = Path(root)
    try:
        from . import auto_train

        pointer = auto_train.read_assistant_default(root)
    except Exception:
        pointer = None
    if not pointer:
        pointer = read_json(root / "results" / POINTER_NAME, {})
    checkpoint = (pointer or {}).get("checkpoint")
    if not checkpoint:
        raise ValueError("No approved AdamLM version is set (results/assistant_default.json is missing).")
    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        candidate = root / checkpoint
    if not candidate.is_file():
        raise ValueError(f"Approved AdamLM checkpoint is missing: {checkpoint}. Previous versions remain under results/ for rollback.")
    return candidate.resolve()


def approved_summary(root: Path = ROOT) -> dict:
    """Small Chat header model: name, version label, checkpoint identity."""
    root = Path(root)
    try:
        pointer = resolve_approved_checkpoint(root)
    except ValueError as exc:
        return {"name": "AdamLM", "version": "unavailable", "approved": False, "error": str(exc)}
    registry = read_registry(root)
    entry = next((v for v in registry.get("versions", []) if Path(v["checkpoint"]).resolve() == pointer), None)
    version = (entry or {}).get("version_id") or pointer.parent.parent.name
    return {
        "name": "AdamLM",
        "version": version,
        "approved": True,
        "run": (entry or {}).get("run"),
        "step": (entry or {}).get("step"),
        "checkpoint": str(pointer),
        "promoted_at": ((registry.get("approved") or {}).get("promoted_at")),
    }


def rollback_to_previous(root: Path = ROOT, *, confirm: bool = False) -> dict:
    """Point Chat back at the previous approved checkpoint (pointer swap only)."""
    if not confirm:
        raise ValueError("Rollback requires explicit confirmation.")
    root = Path(root)
    pointer_path = root / "results" / POINTER_NAME
    pointer = read_json(pointer_path, {})
    previous = (pointer or {}).get("previous") or {}
    prev_checkpoint = previous.get("checkpoint")
    if not prev_checkpoint:
        raise ValueError("No previous approved version is recorded; nothing to roll back to.")
    candidate = Path(prev_checkpoint)
    if not candidate.is_absolute():
        candidate = root / prev_checkpoint
    if not candidate.is_file():
        raise ValueError(f"Previous approved checkpoint is missing: {prev_checkpoint}. Weights and history were not changed.")
    new_pointer = {
        "run": previous.get("run"),
        "checkpoint": str(candidate.resolve()) if candidate.is_absolute() else prev_checkpoint,
        "promoted_at": time.time(),
        "plan_id": f"rollback-{time.strftime('%Y%m%d-%H%M%S')}",
        "eval": {"rollback": True},
        "previous": {"run": pointer.get("run"), "checkpoint": pointer.get("checkpoint")},
        "rollback_from": pointer.get("checkpoint"),
        "reason": "operator rollback to previous approved version",
    }
    gui_core.atomic_json(pointer_path, new_pointer)
    return new_pointer


def promote_candidate(root: Path = ROOT, *, checkpoint: str, eval_info: dict | None = None,
                      reason: str = '', plan_id: str | None = None,
                      require_eval_win: bool = True) -> dict:
    '''Promote a candidate to the approved AdamLM pointer.

    Pointer swap only -- no weights are modified. The previous approved entry
    is retained for rollback. Promotion requires evidence: a recorded eval
    whose scores show no regression, unless the operator explicitly passes
    require_eval_win=False with a written reason.
    '''
    root = Path(root)
    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        candidate = root / checkpoint
    if not candidate.is_file():
        raise ValueError(f'Candidate checkpoint is missing: {checkpoint}.')
    if not reason.strip():
        raise ValueError('A promotion reason is required.')
    if require_eval_win:
        if not isinstance(eval_info, dict):
            raise ValueError('Promotion needs a complete eval comparison (missing evaluation).')
        for key in ('instr_f1_old', 'instr_f1_new', 'conv_f1_old', 'conv_f1_new'):
            if key not in eval_info:
                raise ValueError('Promotion needs a complete eval comparison (missing scores).')
        if float(eval_info['instr_f1_new']) < float(eval_info['instr_f1_old']):
            raise ValueError('Promotion refused: instructions regressed.')
        if float(eval_info['conv_f1_new']) < float(eval_info['conv_f1_old']):
            raise ValueError('Promotion refused: conversation regressed.')
        if eval_info.get('hygiene_problems'):
            raise ValueError('Promotion refused: hygiene problems in candidate replies.')
    pointer_path = root / 'results' / POINTER_NAME
    current = read_json(pointer_path, {})
    try:
        run_name = candidate.parent.parent.name
    except Exception:
        run_name = 'unknown'
    pointer = {
        'run': run_name,
        'checkpoint': str(candidate.resolve()),
        'promoted_at': time.time(),
        'plan_id': plan_id,
        'eval': eval_info or {},
        'previous': {'run': current.get('run'), 'checkpoint': current.get('checkpoint')} if current.get('checkpoint') else None,
        'reason': reason.strip()[:2000],
    }
    gui_core.atomic_json(pointer_path, pointer)
    return pointer
