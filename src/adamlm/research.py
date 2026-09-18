"""Optional AI Research Agent: bounded inspect → propose → train → test → decide.

This agent is SEPARATE from AdamLM. AdamLM's training and inference stay
local; the external provider (e.g. an OpenRouter free model over an
OpenAI-compatible API) only acts as a decision-making assistant that
proposes ONE focused experiment at a time. Its responses are untrusted
proposals validated by the local controller — never shell commands.

Local-only mode runs the same loop with the deterministic heuristic
planner and no external requests at all. Chat and manual training work
without any provider configured.

Session state lives at ``results/research/<session_id>.json`` (+
``.report.md`` / ``.report.json``) so browser closure, backend restarts,
provider outages, trainer crashes, and graceful stops all resume safely
without duplicating completed stages.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from difflib import SequenceMatcher
from pathlib import Path

from . import gpu_session
from .gui_core import ROOT, atomic_json, build_plan, discover_runs, format_duration, format_number, read_json

SESSIONS_DIR = "results/research"
SUPPORTED_DATASETS = ("tinystories", "wikitext103", "fineweb_edu", "mixture", "dolly", "dailydialog")
STOP_REASONS = ("token-budget", "time-budget", "api-cap", "no-data", "no-improvement",
                "repeated-failures", "storage", "incompatible", "trainer-failure", "user-stop")

_threads: dict[str, threading.Thread] = {}


# --------------------------------------------------------------------------
# Supervisor providers live in providers.py (OpenRouter only in AdamLM 2.0;
# Groq/UnoRouter/APInex/xKiro/Ollama are reserved for later). Names are
# re-exported here so existing callers keep working; the Research Agent
# uses them read-only for planning decisions. Training and Chat stay local.
# --------------------------------------------------------------------------

from .providers import (  # noqa: E402
    PROVIDER_CONFIG,
    PROVIDER_KEY_FILE,
    ProviderError,
    ProviderExhausted,
    _chat_completion_with_fallback,
    _classify_http_error,
    clear_api_key,
    default_provider_config,
    get_provider,
    is_free_model_id,
    public_provider_config,
    read_provider_config,
    store_api_key,
    supervisor_models,
    test_provider_connection,
    write_provider_config,
)


# --------------------------------------------------------------------------
# Project inspection: compact structured summaries (never whole dumps).
# --------------------------------------------------------------------------

def dataset_summaries(root: Path = ROOT) -> dict:
    """Measured usable tokens per prepared source (manifests, not file size)."""
    root = Path(root)
    summaries: dict[str, dict] = {}
    for name in ("dolly", "dailydialog", "sft_assistant"):
        manifest_path = root / "data" / name / "processed-manifest.json"
        manifest = read_json(manifest_path, {})
        if not manifest:
            continue
        files = manifest.get("files", {}) or {}
        train = files.get("train", {}) or {}
        summaries[name] = {
            "manifest": f"data/{name}/processed-manifest.json",
            "format": manifest.get("format"),
            "train_examples": train.get("examples"),
            "train_tokens": (manifest.get("sources", {}) or {}).get("tokens")
            if "tokens" in (manifest.get("sources", {}) or {}) else None,
            "sources": {k: {"rows": v.get("rows"), "tokens": v.get("tokens")}
                        for k, v in ((manifest.get("sources") or {}).items()) if isinstance(v, dict)},
        }
    extra = read_json(root / "config" / "extra-datasets.json", {})
    if extra:
        summaries["_extra_note"] = {"config": "config/extra-datasets.json",
                                    "datasets": list((extra.get("datasets") or {}).keys())}
    try:
        from . import auto_train

        stats = auto_train.read_mix_stats(root)
        summaries["assistant_mix_measured"] = stats
    except Exception as exc:
        summaries["assistant_mix_measured"] = {"error": str(exc)}
    return summaries


def _normalized_hypothesis(value) -> str:
    """Compare hypotheses by substance, ignoring casing and punctuation."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _normalized_mixture(value) -> tuple:
    """Make equivalent mixture spelling/order compare as one configuration."""
    parts = []
    for item in str(value or "").split(","):
        if "=" not in item:
            continue
        name, weight = item.split("=", 1)
        try:
            parts.append((name.strip().lower(), round(float(weight), 12)))
        except ValueError:
            parts.append((name.strip().lower(), weight.strip().lower()))
    return tuple(sorted(parts))


def _resolved_checkpoint(value) -> str:
    try:
        return str(Path(str(value)).resolve())
    except OSError:
        return str(value)


def research_outcomes(root: Path = ROOT, *, limit: int = 20) -> list[dict]:
    """Saved failed/rejected research decisions, newest first, safe for planners.

    Session files are the durable audit trail.  Only outcomes that say a
    configuration did not work are exposed here; successful candidates are
    not treated as a reason to suppress a later, independently justified run.
    """
    directory = Path(root) / SESSIONS_DIR
    if not directory.exists():
        return []
    outcomes = []
    paths = sorted(directory.glob("research-*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        session = read_json(path, {})
        session_id = str(session.get("session_id") or path.stem)
        for experiment in session.get("experiments", []) or []:
            if experiment.get("status") not in ("failed", "rejected"):
                continue
            plan = experiment.get("plan") or {}
            if not isinstance(plan, dict) or not plan.get("parent_checkpoint"):
                continue
            outcomes.append({
                "session_id": session_id,
                "index": experiment.get("index"),
                "status": experiment.get("status"),
                "parent_checkpoint": _resolved_checkpoint(plan.get("parent_checkpoint")),
                "dataset": plan.get("dataset"),
                "mixture": plan.get("mixture"),
                "target_tokens": plan.get("target_tokens"),
                "hypothesis": str(plan.get("hypothesis") or "")[:500],
                "reason": str(experiment.get("failure") or experiment.get("decision") or "")[:500],
                "eval": experiment.get("eval") if isinstance(experiment.get("eval"), dict) else None,
            })
            if len(outcomes) >= limit:
                return outcomes
    return outcomes


def _meaningful_proposal_changes(proposal: dict, prior: dict) -> list[str]:
    """List material changes from a failed/rejected proposal on one parent."""
    changes = []
    if _resolved_checkpoint(proposal.get("parent_checkpoint")) != _resolved_checkpoint(prior.get("parent_checkpoint")):
        changes.append("parent checkpoint")
    if str(proposal.get("dataset") or "") != str(prior.get("dataset") or ""):
        changes.append("dataset")
    if _normalized_mixture(proposal.get("mixture")) != _normalized_mixture(prior.get("mixture")):
        changes.append("mixture")
    try:
        before, after = int(prior.get("target_tokens")), int(proposal.get("target_tokens"))
        if abs(after - before) >= max(1, int(before * 0.10)):
            changes.append("token target")
    except (TypeError, ValueError):
        if proposal.get("target_tokens") != prior.get("target_tokens"):
            changes.append("token target")
    before_hypothesis = _normalized_hypothesis(prior.get("hypothesis"))
    after_hypothesis = _normalized_hypothesis(proposal.get("hypothesis"))
    if (before_hypothesis != after_hypothesis
            and SequenceMatcher(None, before_hypothesis, after_hypothesis).ratio() < 0.85):
        changes.append("hypothesis")
    return changes


def _proposal_reuse_reason(proposal: dict, outcomes: list[dict]) -> str | None:
    """Return a precise stop reason when a known failed configuration recurs."""
    parent = _resolved_checkpoint(proposal.get("parent_checkpoint"))
    for prior in outcomes:
        if parent != _resolved_checkpoint(prior.get("parent_checkpoint")):
            continue
        if _meaningful_proposal_changes(proposal, prior):
            continue
        return ("This repeats failed/rejected experiment "
                f"{prior.get('session_id')}#{prior.get('index')} from the same parent "
                "without a meaningful change to dataset, mixture, token target, or hypothesis. "
                f"Previous outcome: {prior.get('status')}"
                + (f" ({prior.get('reason')})." if prior.get("reason") else "."))
    return None


def inspect_project(root: Path = ROOT) -> dict:
    """Compact real-state summary for planning (approved, lineage, budgets...)."""
    root = Path(root)
    from . import auto_train, versions

    runs = discover_runs(root)
    approved = versions.approved_summary(root)
    try:
        head = auto_train.find_assistant_head(runs, root)
        head_info = {"run": head.name, "checkpoint": str(head.checkpoint),
                     "tokens": head.tokens,
                     "step": head.status.get("checkpoint_step") or head.status.get("step")}
    except Exception as exc:
        head_info = {"error": str(exc)}
    active = [r.name for r in runs if r.active]
    recent_plans = []
    for path in sorted((root / "results").glob("*.auto.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
        plan = read_json(path, {})
        recent_plans.append({"plan_id": plan.get("plan_id"), "status": plan.get("status"),
                             "promotion": (plan.get("promotion") or {}).get("promoted"),
                             "reason": (plan.get("promotion") or {}).get("reason")})
    last_eval = None
    pointer = read_json(root / "results" / "assistant_default.json", {})
    if isinstance(pointer.get("eval"), dict):
        last_eval = pointer["eval"]
    from .gui_core import disk_telemetry

    disk = disk_telemetry(root)
    return {
        "approved": approved,
        "assistant_head": head_info,
        "active_runs": active,
        "gpu": gpu_session.describe(root),
        "recent_plans": recent_plans,
        "last_promotion_eval": last_eval,
        "research_outcomes": research_outcomes(root),
        "datasets": dataset_summaries(root),
        "disk_free_gib": round(disk.get("free", 0) / 2**30, 2),
        "eval_reports": __import__("adamlm.eval_full", fromlist=["list_reports"]).list_reports(root=root, limit=5),
    }


# --------------------------------------------------------------------------
# Experiment proposals: schema + validation (untrusted input boundary).
# --------------------------------------------------------------------------

PROPOSAL_SCHEMA = {
    "hypothesis": str, "parent_checkpoint": str, "dataset": str,
    "target_tokens": int, "max_seconds": (int, float), "eval_plan": str,
}


def validate_proposal(proposal: dict, *, summary: dict, limits: dict,
                      root: Path = ROOT) -> dict:
    """Validate an experiment proposal; raises on any violation. No shell."""
    if not isinstance(proposal, dict):
        raise ValueError("Proposal must be a JSON object.")
    for field, types in PROPOSAL_SCHEMA.items():
        if field not in proposal:
            raise ValueError(f"Proposal is missing required field '{field}'.")
        if not isinstance(proposal[field], types) or (
                isinstance(proposal[field], str) and not proposal[field].strip()):
            raise ValueError(f"Proposal field '{field}' has an unsupported value.")
    dataset = proposal["dataset"]
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Allowed: {', '.join(SUPPORTED_DATASETS)}.")
    allowed = limits.get("allowed_datasets")
    if allowed is not None and dataset not in list(allowed):
        raise ValueError(f"Dataset '{dataset}' is not selected for this session "
                         f"(allowed: {', '.join(list(allowed))}).")
    if dataset == "mixture" and not str(proposal.get("mixture") or "").strip():
        raise ValueError("Mixture experiments must include explicit mixture weights.")
    parent = Path(str(proposal["parent_checkpoint"]))
    if not parent.is_file():
        raise ValueError(f"Parent checkpoint does not exist: {proposal['parent_checkpoint']}.")
    # Parent must be a known saved checkpoint (never an arbitrary path).
    known = {str(r.checkpoint) for r in discover_runs(Path(root)) if r.checkpoint}
    try:
        resolved = str(parent.resolve())
    except OSError:
        resolved = str(parent)
    if resolved not in {str(Path(k).resolve()) for k in known if Path(k).exists()}:
        raise ValueError("Parent checkpoint is not a known saved checkpoint; refusing.")
    prior_outcomes = summary.get("research_outcomes")
    if not isinstance(prior_outcomes, list):
        prior_outcomes = research_outcomes(root)
    reuse_reason = _proposal_reuse_reason({**proposal, "parent_checkpoint": resolved}, prior_outcomes)
    if reuse_reason:
        raise ValueError(reuse_reason)
    tokens = int(proposal["target_tokens"])
    if tokens <= 0 or tokens > int(limits.get("max_tokens", 0)):
        raise ValueError("Proposed token budget is outside the authorized session budget.")
    used = int(limits.get("_tokens_used", 0))
    if used + tokens > int(limits.get("max_tokens", 0)):
        raise ValueError(f"Proposal needs {format_number(tokens)} tokens but only "
                         f"{format_number(int(limits.get('max_tokens', 0)) - used)} remain in budget.")
    seconds = float(proposal["max_seconds"])
    if seconds <= 0 or seconds > float(limits.get("max_seconds", 0)):
        raise ValueError("Proposed time budget exceeds the authorized session limit.")
    stage_cap = limits.get("stage_token_cap")
    if stage_cap is not None and tokens > int(stage_cap):
        raise ValueError(f"Proposed {format_number(tokens)} exceeds this session's per-stage "
                         f"cap of {format_number(int(stage_cap))}; stages stay diagnostic by default.")
    single_pass = _single_pass_tokens(dataset, root)
    if (single_pass is not None and tokens > single_pass
            and not limits.get("allow_repetition", False)):
        raise ValueError(f"Proposed {format_number(tokens)} repeats the {dataset} diet "
                         f"(one pass is {format_number(single_pass)}). Confirm repetition "
                         "for this session before training on repeated rows.")
    allowed_ops = limits.get("allowed_ops") or ["train", "evaluate", "compare"]
    if "train" not in allowed_ops:
        raise ValueError("Training operations are not allowed in this session.")
    return {
        "hypothesis": str(proposal["hypothesis"]).strip()[:2000],
        "parent_checkpoint": resolved,
        "dataset": dataset,
        "mixture": str(proposal.get("mixture") or "").strip() or None,
        "target_tokens": tokens,
        "max_seconds": seconds,
        "eval_plan": str(proposal["eval_plan"]).strip()[:2000],
        "expected_risks": str(proposal.get("expected_risks") or "")[:2000],
        "stop_conditions": str(proposal.get("stop_conditions") or "")[:1000],
        "config": str(proposal.get("config") or "")[:300],
    }


def local_heuristic_proposal(summary: dict, *, limits: dict) -> dict:
    """Deterministic local planner: weakest measured skill -> focused diet.

    Never assumes more tokens or more datasets are better: picks exactly one
    small diagnostic experiment, or declines with stop=true when the measured
    state gives no justified next step.
    """
    approved = summary.get("approved", {}) or {}
    head = summary.get("assistant_head", {}) or {}
    last = summary.get("last_promotion_eval", {}) or {}
    parent = approved.get("checkpoint") or head.get("checkpoint") or ""
    budget_tokens = int(limits.get("max_tokens", 0)) - int(limits.get("_tokens_used", 0))
    budget_seconds = float(limits.get("max_seconds", 0))
    if not parent:
        return {"stop": True, "reason": "No approved assistant or lineage head found; nothing safe to continue from."}
    if budget_tokens < 200_000:
        return {"stop": True, "reason": f"Only {format_number(budget_tokens)} tokens remain — too little for a meaningful stage."}
    try:
        instr = float(last.get("instr_f1_new", last.get("instr_f1", 0.3)))
        conv = float(last.get("conv_f1_new", last.get("conv_f1", 0.05)))
    except (TypeError, ValueError):
        instr, conv = 0.3, 0.05
    try:
        stage_cap = int(limits.get("stage_token_cap") or 1_000_000)
    except (TypeError, ValueError):
        stage_cap = 1_000_000
    allowed = limits.get("allowed_datasets")
    allowed = list(allowed) if allowed is not None else list(SUPPORTED_DATASETS)
    tokens = min(budget_tokens, stage_cap, 1_000_000)
    seconds = min(budget_seconds, 3600.0)
    preference = "dailydialog" if conv + 0.05 < instr else ("dolly" if instr <= conv else None)
    if preference is None:
        return {"stop": True, "reason": "No clear weakest skill in the measured profile; declining to train without a hypothesis."}
    dataset = preference
    fallback_note = ""
    if dataset not in allowed:
        fallback = "dolly" if dataset == "dailydialog" else "dailydialog"
        if fallback in allowed:
            dataset = fallback
            fallback_note = (f" Preferred diet '{preference}' is not selected for this session, "
                             "so the other focused SFT diet is used instead.")
        else:
            return {"stop": True,
                    "reason": (f"The heuristic's focused SFT diets (dolly, dailydialog) are not "
                               f"selected for this session (allowed: {', '.join(allowed)}); "
                               "declining rather than training on an unselected diet.")}
    if dataset == "dailydialog":
        candidate = {
            "hypothesis": ("Conversation skill lags instructions "
                           f"(conv F1 {conv:.3f} < instr F1 {instr:.3f}). A small DailyDialog-heavy "
                           "SFT stage should improve greeting/follow-up relevance without touching general weights."
                           + fallback_note),
            "parent_checkpoint": parent, "dataset": "dailydialog",
            "target_tokens": tokens, "max_seconds": seconds,
            "eval_plan": "assistant probes (greetings + follow-ups primary) + hygiene + capability regression check",
            "expected_risks": "possible instruction-skill dip; promotion gate requires no regression",
            "stop_conditions": "hygiene failure or capability regression beyond tolerance",
        }
    else:
        candidate = {
            "hypothesis": ("Instruction-following is the weaker skill "
                           f"(instr F1 {instr:.3f} <= conv F1 {conv:.3f}). A small Dolly-heavy SFT "
                           "stage should improve short-instruction compliance."
                           + fallback_note),
            "parent_checkpoint": parent, "dataset": "dolly",
            "target_tokens": tokens, "max_seconds": seconds,
            "eval_plan": "assistant probes (instructions primary) + hygiene + capability regression check",
            "expected_risks": "possible conversation dip or overfitting to Dolly phrasing",
            "stop_conditions": "hygiene failure or capability regression beyond tolerance",
        }
    reuse_reason = _proposal_reuse_reason(candidate, summary.get("research_outcomes") or [])
    if reuse_reason:
        return {"stop": True, "reason": reuse_reason}
    return candidate


AGENT_SYSTEM = (
    "You are a careful ML experiment planner for a small local language model. "
    "Propose exactly ONE focused experiment as a single JSON object with keys: "
    "hypothesis, parent_checkpoint (must be one of the listed known checkpoints), "
    "dataset (one of tinystories, wikitext103, fineweb_edu, mixture, dolly, dailydialog), "
    "mixture (required only for mixture), target_tokens (integer within remaining budget), "
    "max_seconds (within remaining time), eval_plan, expected_risks, stop_conditions. "
    "Or return {\"stop\": true, \"reason\": \"...\"} when no worthwhile experiment exists. "
    "Never propose deleting checkpoints, editing code, shell commands, fetching new data, "
    "or repeating tiny data silently. Output JSON only."
)


def ai_propose(summary: dict, *, limits: dict, root: Path) -> dict:
    """Ask the approved free-model list for one proposal; validate before use.

    Walks primary -> fallbacks on rate limits / outages (free models only,
    never paid). Returns an envelope:
      {"outcome": <proposal dict or {"stop": True, ...}>,
       "supervisor_model": <id that produced it>,
       "attempts": [...per-model audit trail...],
       "api_requests": <HTTP attempts consumed>}
    Raises ProviderExhausted (with .attempts) when no model could answer —
    the supervisor then pauses instead of failing the session.
    """
    config = read_provider_config(root)
    if not config.get("enabled"):
        raise ValueError("AI planner is disabled; enable it or use local-only mode.")
    if not config.get("allow_external_eval"):
        can_send = [c for c in (config.get("send_categories") or []) if c != "example-replies"]
    else:
        can_send = config.get("send_categories") or []
    allowed = limits.get("allowed_datasets")
    allowed = list(allowed) if allowed is not None else list(SUPPORTED_DATASETS)
    stage_cap = limits.get("stage_token_cap")
    rules = ("One focused experiment. Supported datasets only. Token/time caps are hard. "
             "Respond with JSON only."
             f" Session datasets: {', '.join(allowed)} — propose only one of these."
             + (f" Per-stage cap: {int(stage_cap)} tokens." if stage_cap is not None else "")
             + (" Data repetition is NOT confirmed: stay within one pass of the diet."
                if not limits.get("allow_repetition", False) else "")
             + (" Do not repeat a failed or rejected configuration from the same parent; "
                "change the parent, dataset, mixture, token target, or hypothesis materially."
                if summary.get("research_outcomes") else ""))
    compact = {
        "approved": summary.get("approved"),
        "assistant_head": summary.get("assistant_head"),
        "last_promotion_eval": summary.get("last_promotion_eval"),
        "research_outcomes": summary.get("research_outcomes", []),
        "datasets": summary.get("datasets"),
        "disk_free_gib": summary.get("disk_free_gib"),
        "remaining_tokens": int(limits.get("max_tokens", 0)) - int(limits.get("_tokens_used", 0)),
        "remaining_seconds": float(limits.get("max_seconds", 0)),
        "send_categories": can_send,
        "known_checkpoints": [approved_ckpt(summary)],
        "allowed_datasets": allowed,
        "rules": rules,
    }
    call = _chat_completion_with_fallback(
        root,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Project state:\n" + json.dumps(compact)[:6000]},
        ],
        requests_used=int(limits.get("_api_used", 0)),
        requests_cap=int(limits.get("_api_cap", 0)),
    )
    text = call["text"]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ProviderError(
            f"Supervisor model '{call['model']}' returned no JSON proposal.",
            kind="bad_response", model=call["model"], retryable=True)
    try:
        proposal = json.loads(match.group(0))
    except json.JSONDecodeError:
        raise ProviderError(
            f"Supervisor model '{call['model']}' returned invalid JSON.",
            kind="bad_response", model=call["model"], retryable=True) from None
    if proposal.get("stop"):
        outcome = {"stop": True,
                   "reason": str(proposal.get("reason") or "provider declined to propose")[:1000]}
    else:
        outcome = validate_proposal(proposal, summary=summary, limits=limits, root=root)
    return {"outcome": outcome, "supervisor_model": call["model"],
            "attempts": call["attempts"], "api_requests": call["api_requests"]}


def approved_ckpt(summary: dict) -> str:
    return str(((summary.get("approved") or {}).get("checkpoint")) or "")


def _single_pass_tokens(dataset: str, root: Path = ROOT) -> int | None:
    """Measured single-pass size for finite SFT diets, else None (streaming).

    Sizes come from the curated mix manifest (dolly 3.03M / dailydialog
    5.26M training tokens). Streaming pretraining diets have no fixed pass,
    so the repetition rule cannot apply to them and they return None.
    """
    try:
        from . import auto_train

        stats = auto_train.read_mix_stats(root)
    except Exception:
        return None
    if dataset == "dolly":
        return int(stats["dolly_tokens"])
    if dataset == "dailydialog":
        return int(stats["dd_tokens"])
    return None


def _normalize_allowed_datasets(value) -> list[str]:
    if value is None:
        return list(SUPPORTED_DATASETS)
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(",", " ").split()]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    else:
        raise ValueError("Dataset selection must be a list of dataset names.")
    cleaned = [p for p in parts if p]
    if not cleaned:
        raise ValueError("Select at least one dataset for the session.")
    unknown = [p for p in cleaned if p not in SUPPORTED_DATASETS]
    if unknown:
        raise ValueError(f"Unsupported dataset(s) {unknown}. Allowed: {', '.join(SUPPORTED_DATASETS)}.")
    seen = []
    for name in cleaned:
        if name not in seen:
            seen.append(name)
    return seen


# --------------------------------------------------------------------------
# Sessions: persistence + bounded supervisor loop.
# --------------------------------------------------------------------------

def _session_path(session_id: str, root: Path) -> Path:
    return Path(root) / SESSIONS_DIR / f"{session_id}.json"


def new_session(root: Path = ROOT, *, max_tokens: int, max_seconds: float,
                max_experiments: int = 3, max_api_requests: int = 25,
                max_api_cost_usd: float = 0.0, max_repetition: int = 3,
                allow_external_eval: bool = False,
                require_manual_promotion: bool = True,
                allowed_ops: list | None = None,
                use_ai: bool = False,
                goal: str = "",
                allowed_datasets: list | None = None,
                stage_token_cap: int = 1_000_000,
                allow_repetition: bool = False,
                mixture: str = "") -> dict:
    if max_tokens <= 0 or max_seconds <= 0:
        raise ValueError("Session needs a positive token budget and time limit.")
    if float(max_api_cost_usd or 0.0) != 0.0:
        raise ValueError("Maximum API cost must stay 0 (free-models-only operation).")
    datasets = _normalize_allowed_datasets(allowed_datasets)
    try:
        stage_cap = int(stage_token_cap)
    except (TypeError, ValueError):
        raise ValueError("Per-stage token cap must be a whole number.") from None
    if stage_cap <= 0:
        raise ValueError("Per-stage token cap must be positive.")
    mix_text = str(mixture or "").strip()
    if "mixture" in datasets and mix_text:
        from .gui_core import validate_mixture

        validate_mixture(mix_text)
    session_id = f"research-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    session = {
        "session_id": session_id,
        "created_at": time.time(),
        "started_at": None,
        "updated_at": time.time(),
        "status": "created",
        "goal": str(goal or "")[:1000],
        "use_ai": bool(use_ai),
        "limits": {
            "max_tokens": int(max_tokens),
            "max_seconds": float(max_seconds),
            "max_experiments": int(max_experiments),
            "max_api_requests": int(max_api_requests),
            "max_api_cost_usd": float(max_api_cost_usd),
            "max_repetition": int(max_repetition),
            "allow_external_eval": bool(allow_external_eval),
            "require_manual_promotion": bool(require_manual_promotion),
            "allowed_ops": allowed_ops or ["train", "evaluate", "compare"],
            "allowed_datasets": datasets,
            "stage_token_cap": stage_cap,
            "allow_repetition": bool(allow_repetition),
            "mixture": mix_text,
        },
        "budget_used": {"tokens": 0, "seconds": 0.0, "api_requests": 0, "failures": 0},
        "experiments": [],
        "stop_reason": None,
        "report": None,
        # Provider-fallback state: per-model outage notes and pause record so
        # a later resume knows exactly what was tried without repeating it.
        "provider_state": {"paused": False, "reason": None, "paused_at": None,
                           "attempts": [], "last_model": None},
    }
    atomic_json(_session_path(session_id, root), session)
    return session


def read_session(session_id: str, root: Path = ROOT) -> dict:
    data = read_json(_session_path(session_id, root), {})
    if not data or data.get("session_id") != session_id:
        raise ValueError(f"Research session '{session_id}' was not found.")
    return data


def latest_session(root: Path = ROOT) -> dict | None:
    directory = Path(root) / SESSIONS_DIR
    if not directory.exists():
        return None
    candidates = [read_json(p, {}) for p in directory.glob("research-*.json")]
    candidates = [c for c in candidates if c.get("session_id")]
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.get("updated_at", 0), c.get("created_at", 0)))


def _save(session: dict, root: Path) -> dict:
    session["updated_at"] = time.time()
    atomic_json(_session_path(session["session_id"], root), session)
    return session


def _deadline(session: dict) -> float:
    return (session.get("started_at") or 0) + float(session["limits"]["max_seconds"])


def _remaining_seconds(session: dict) -> float:
    if not session.get("started_at"):
        return float(session["limits"]["max_seconds"])
    return max(0.0, _deadline(session) - time.time())


class Ctx:
    """Host wiring (real backend or tests). Mirrors auto_train's Supervisor ctx."""

    def launch(self, command, cwd, log_path):
        log = open(log_path, "a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [str(c) for c in command], cwd=str(cwd),
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            log.close()
            raise
        return process

    def read_stage_status(self, run_dir):
        return read_json(Path(run_dir) / "status.json", {})

    def latest_checkpoint(self, run_dir):
        from .gui_core import latest_checkpoint

        return latest_checkpoint(Path(run_dir))

    def request_stop(self, run_dir):
        from .gui_core import request_stop

        request_stop(Path(run_dir))

    def sleep(self, seconds):
        time.sleep(seconds)

    def now(self):
        return time.time()


def _experiment_run_dir(session_id: str, index: int, root: Path) -> Path:
    return Path(root) / "results" / f"{session_id}-e{index}"


def _stage_command(root: Path, *, proposal: dict, run_dir: Path, remaining: float) -> list[str]:
    dataset = proposal["dataset"]
    if dataset in ("dolly", "dailydialog"):
        config_path = root / "config" / ("sft-dolly.json" if dataset == "dolly" else "sft-dailydialog.json")
    elif dataset in ("tinystories", "wikitext103", "fineweb_edu", "mixture"):
        config_path = root / "config" / "pretrain-general.json"
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")
    if proposal.get("config"):
        candidate = root / proposal["config"]
        if candidate.is_file():
            config_path = candidate
    plan = build_plan(mode="continuation", dataset=dataset,
                      additional_tokens=int(proposal["target_tokens"]),
                      duration_seconds=min(float(proposal["max_seconds"]), remaining),
                      parent_checkpoint=Path(proposal["parent_checkpoint"]),
                      run_dir=run_dir, config_path=config_path, root=root)
    return [str(c) for c in plan.command]


class Supervisor:
    def __init__(self, session_id, root=ROOT, ctx=None):
        self.session_id = session_id
        self.root = Path(root)
        self.ctx = ctx or Ctx()

    def run(self):
        try:
            self._run_inner()
        except Exception as exc:
            try:
                session = read_session(self.session_id, self.root)
                if session.get("status") == "running":
                    session["status"] = "failed"
                    session["stop_reason"] = f"supervisor error: {exc}"
                    _save(session, self.root)
            except Exception:
                pass
        finally:
            gpu_session.release(self.root)

    def _run_inner(self):
        session = read_session(self.session_id, self.root)
        if session.get("status") not in ("running",):
            return
        if session.get("started_at") is None:
            session["started_at"] = self.ctx.now()
            _save(session, self.root)
        while True:
            session = read_session(self.session_id, self.root)
            if session.get("status") != "running":
                return
            stop = self._check_budgets(session)
            if stop:
                session["status"] = "stopped"
                session["stop_reason"] = stop
                self._write_report(_save(session, self.root))
                return
            if len(session["experiments"]) >= int(session["limits"]["max_experiments"]):
                session["status"] = "stopped"
                session["stop_reason"] = "experiment budget reached"
                self._write_report(_save(session, self.root))
                return
            self._run_experiment(session)

    def _check_budgets(self, session: dict) -> str | None:
        used = session.get("budget_used", {})
        if int(used.get("tokens", 0)) >= int(session["limits"]["max_tokens"]):
            return "Reached token budget"
        if self.ctx.now() >= _deadline(session):
            return "Reached time budget"
        # API request cap is enforced during the propose phase by the
        # fallback walker, which records per-model attempts and pauses
        # with full audit trail. We do NOT stop here so the supervisor
        # can produce a proper "paused-provider" record.
        if int(used.get("failures", 0)) >= 3:
            return "Repeated failures — stopping for review"
        return None

    def _pause_for_provider(self, session: dict, index: int, exc: ProviderError) -> None:
        """Pause at an AI-dependent decision without losing anything.

        Called only from the propose phase, where no training stage can be
        running (the loop is strictly sequential). Completed stages,
        checkpoints, budgets, and the approved model are all preserved;
        resuming continues with the next experiment index — nothing re-runs.
        """
        attempts = list(getattr(exc, "attempts", []) or [])
        reason = str(exc)[:1000]
        session["provider_state"] = {
            "paused": True,
            "reason": reason,
            "kind": getattr(exc, "kind", "exhausted"),
            "paused_at": self.ctx.now(),
            "attempts": attempts,
            "last_model": session.get("provider_state", {}).get("last_model"),
        }
        session["status"] = "paused-provider"
        session["stop_reason"] = (
            f"Provider pause at experiment {index} decision: {reason} "
            "Resume any time — completed stages are kept and never re-run.")
        session["experiments"].append({
            "index": index,
            "status": "paused-provider",
            "phase": "propose",
            "reason": session["stop_reason"],
            "supervisor_model": session["provider_state"]["last_model"],
            "supervisor_attempts": attempts,
        })
        self._write_report(_save(session, self.root))

    def _run_experiment(self, session: dict):
        from . import eval_full, versions

        index = len(session["experiments"]) + 1
        summary = inspect_project(self.root)
        limits = dict(session["limits"])
        limits["_tokens_used"] = int(session.get("budget_used", {}).get("tokens", 0))
        # PHASE A+B — inspect & propose (persisted before execution).
        proposal = None
        supervisor_model = None
        supervisor_attempts: list = []
        if session.get("use_ai"):
            limits["_api_used"] = int(session.get("budget_used", {}).get("api_requests", 0))
            limits["_api_cap"] = int(session["limits"].get("max_api_requests", 0))
            try:
                result = ai_propose(summary, limits=limits, root=self.root)
            except ProviderError as exc:
                # Provider-side outage (rate limits on every approved model,
                # account-wide quota, bad key, request cap): pause safely at
                # this AI decision. No training is running here, nothing is
                # marked failed, and the approved model is untouched.
                self._pause_for_provider(session, index, exc)
                return
            except ValueError as exc:
                # ai_propose validates provider output too.  A duplicate
                # outcome is a deliberate terminal decision, not an AI or
                # trainer failure that should consume another experiment slot.
                if str(exc).startswith("This repeats failed/rejected experiment"):
                    session["status"] = "stopped"
                    session["stop_reason"] = str(exc)
                    session["experiments"].append({"index": index, "status": "declined",
                                                   "phase": "propose", "reason": str(exc)})
                    self._write_report(_save(session, self.root))
                    return
                session["budget_used"]["failures"] = int(session["budget_used"].get("failures", 0)) + 1
                session["experiments"].append({"index": index, "status": "failed",
                                               "phase": "propose",
                                               "failure": f"AI proposal rejected ({exc}); nothing launched."})
                _save(session, self.root)
                return
            except Exception as exc:
                session["budget_used"]["failures"] = int(session["budget_used"].get("failures", 0)) + 1
                experiment = {"index": index, "status": "failed",
                              "phase": "propose",
                              "failure": f"AI proposal unavailable ({exc}); state preserved, no training started."}
                session["experiments"].append(experiment)
                _save(session, self.root)
                return
            session["budget_used"]["api_requests"] = int(
                session["budget_used"].get("api_requests", 0)) + int(result["api_requests"])
            proposal = result["outcome"]
            supervisor_model = result["supervisor_model"]
            supervisor_attempts = result["attempts"]
            state = session.setdefault("provider_state", {})
            state.update({"paused": False, "last_model": supervisor_model,
                          "attempts": supervisor_attempts})
            _save(session, self.root)
        else:
            proposal = local_heuristic_proposal(summary, limits=limits)
        if proposal.get("stop"):
            session["status"] = "stopped"
            session["stop_reason"] = f"No useful next step: {proposal.get('reason')}"
            session["experiments"].append({"index": index, "status": "declined",
                                           "reason": proposal.get("reason")})
            self._write_report(_save(session, self.root))
            return
        validated = None
        try:
            proposal_for_check = dict(proposal)
            if (proposal_for_check.get("dataset") == "mixture"
                    and not str(proposal_for_check.get("mixture") or "").strip()
                    and str(session["limits"].get("mixture") or "").strip()):
                proposal_for_check["mixture"] = str(session["limits"]["mixture"]).strip()
            validated = validate_proposal(proposal_for_check, summary=summary, limits=limits,
                                          root=self.root)
        except ValueError as exc:
            if str(exc).startswith("This repeats failed/rejected experiment"):
                session["status"] = "stopped"
                session["stop_reason"] = str(exc)
                session["experiments"].append({"index": index, "status": "declined",
                                               "phase": "validate", "reason": str(exc),
                                               "plan": proposal})
                self._write_report(_save(session, self.root))
                return
            session["budget_used"]["failures"] = int(session["budget_used"].get("failures", 0)) + 1
            experiment = {"index": index, "status": "failed",
                          "phase": "validate",
                          "failure": f"Proposal rejected ({exc}); nothing launched, budgets intact."}
            session["experiments"].append(experiment)
            _save(session, self.root)
            return
        run_dir = _experiment_run_dir(session["session_id"], index, self.root)
        experiment = {"index": index, "status": "planned", "plan": validated,
                      "run_dir": str(run_dir), "started_at": self.ctx.now(),
                      "supervisor_model": supervisor_model,
                      "supervisor_attempts": supervisor_attempts}
        session["experiments"].append(experiment)
        _save(session, self.root)
        # PHASE C — train through the existing local CLI (bounded).
        remaining = _deadline(session) - self.ctx.now()
        if remaining <= 0:
            experiment["status"] = "failed"
            experiment["failure"] = "Overall session deadline exhausted before launch."
            session["status"] = "stopped"
            session["stop_reason"] = "Reached time budget"
            self._write_report(_save(session, self.root))
            return
        if gpu_session.is_busy(self.root):
            experiment["status"] = "failed"
            experiment["failure"] = "GPU is busy (training/inference holds it); experiment not started."
            session["budget_used"]["failures"] = int(session["budget_used"].get("failures", 0)) + 1
            _save(session, self.root)
            return
        try:
            command = _stage_command(self.root, proposal=validated, run_dir=run_dir,
                                     remaining=remaining)
        except Exception as exc:
            experiment["status"] = "failed"
            experiment["failure"] = f"Unsupported training configuration: {exc}"
            session["budget_used"]["failures"] = int(session["budget_used"].get("failures", 0)) + 1
            _save(session, self.root)
            return
        claim = gpu_session.try_acquire(self.root, kind="training",
                                        label=f"research:{session['session_id']}-e{index}",
                                        run=run_dir.name)
        if not claim:
            experiment["status"] = "failed"
            experiment["failure"] = "GPU became busy just before launch; experiment not started."
            session["budget_used"]["failures"] = int(session["budget_used"].get("failures", 0)) + 1
            _save(session, self.root)
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        from .gui_core import write_session

        write_session(run_dir, {"run_name": run_dir.name, "mode": "continuation",
                                "dataset": validated["dataset"],
                                "started_at": self.ctx.now(),
                                "max_seconds": min(float(validated["max_seconds"]), remaining),
                                "target_tokens": int(validated["target_tokens"]),
                                "research_session": session["session_id"]})
        handle = self.ctx.launch(command, self.root, run_dir / "web-training.log")
        experiment["status"] = "training"
        _save(session, self.root)
        stop_sent = False
        while True:
            self.ctx.sleep(5)
            session = read_session(self.session_id, self.root)
            experiment = session["experiments"][index - 1]
            if session.get("status") != "running":
                if not stop_sent:
                    try:
                        self.ctx.request_stop(run_dir)
                    except OSError:
                        pass
                    stop_sent = True
            if self.ctx.now() >= _deadline(session) and not stop_sent:
                try:
                    self.ctx.request_stop(run_dir)
                except OSError:
                    pass
                stop_sent = True
            if handle.poll() is not None:
                break
        gpu_session.release(self.root)
        status = self.ctx.read_stage_status(run_dir)
        state = str(status.get("state") or "unknown")
        checkpoint = self.ctx.latest_checkpoint(run_dir)
        try:
            actual_tokens = int(status.get("tokens") or 0)
        except (TypeError, ValueError):
            actual_tokens = 0
        session["budget_used"]["tokens"] = int(session["budget_used"].get("tokens", 0)) + actual_tokens
        experiment["checkpoint"] = str(checkpoint) if checkpoint else None
        experiment["tokens_trained"] = actual_tokens
        experiment["trainer_state"] = state
        if state not in ("target_reached",) and not state.startswith("data_stop"):
            experiment["status"] = "failed"
            experiment["failure"] = f"trainer ended as '{state}'"
            if session.get("status") == "running":
                session["budget_used"]["failures"] = int(session["budget_used"].get("failures", 0)) + 1
            _save(session, self.root)
            return
        # PHASE D+E — test with the real inference system, fixed seeds.
        experiment["status"] = "evaluating"
        _save(session, self.root)
        try:
            approved_ckpt = versions.resolve_approved_checkpoint(self.root)
        except ValueError as exc:
            experiment["status"] = "failed"
            experiment["failure"] = str(exc)
            _save(session, self.root)
            return
        try:
            report = eval_full.compare_checkpoints(str(approved_ckpt), str(checkpoint),
                                                   root=self.root, device="cpu",
                                                   old_label="approved", new_label=f"candidate-e{index}")
            eval_path = eval_full.save_report(
                report, root=self.root, name=f"{session['session_id']}-e{index}.json")
        except Exception as exc:
            experiment["status"] = "failed"
            experiment["failure"] = f"evaluation failed: {exc}"
            session["budget_used"]["failures"] = int(session["budget_used"].get("failures", 0)) + 1
            _save(session, self.root)
            return
        # PHASE F — decide (conservative; promotion needs manual approval).
        card = report["scorecard"]
        experiment["eval_report"] = str(eval_path)
        experiment["eval"] = {k: card.get(k) for k in
                              ("instr_f1_old", "instr_f1_new", "conv_f1_old", "conv_f1_new",
                               "suite_accuracy_old", "suite_accuracy_new", "promote", "reason")}
        experiment["examples"] = report.get("examples", [])[:4]
        if card.get("promote"):
            if session["limits"].get("require_manual_promotion", True):
                experiment["status"] = "awaiting-approval"
                experiment["decision"] = ("Candidate shows convincing improvement; "
                                          "promotion awaits manual approval. Approved assistant unchanged.")
            else:
                experiment["status"] = "promoted"
                experiment["decision"] = card.get("reason")
        else:
            experiment["status"] = "rejected"
            experiment["decision"] = card.get("reason")
            session["budget_used"]["failures"] = int(session["budget_used"].get("failures", 0)) + 1
        _save(session, self.root)
        self._write_report(session)

    def _write_report(self, session: dict):
        lines = [
            f"# Research session {session['session_id']}",
            "",
            f"Status: {session.get('status')}" + (f" — {session.get('stop_reason')}" if session.get("stop_reason") else ""),
            f"Budget used: {format_number(session.get('budget_used', {}).get('tokens', 0))} tokens, "
            f"{format_duration(time.time() - (session.get('started_at') or time.time()))} elapsed.",
            "",
        ]
        for exp in session.get("experiments", []):
            lines.append(f"## Experiment {exp.get('index')} — {exp.get('status')}")
            if exp.get("supervisor_model"):
                lines.append(f"Supervisor model: {exp.get('supervisor_model')}")
            if exp.get("supervisor_attempts"):
                tried = ", ".join(
                    f"{a.get('model')} ({'ok' if a.get('ok') else a.get('kind')})"
                    for a in exp["supervisor_attempts"])
                lines.append(f"Supervisor attempts: {tried}")
            plan = exp.get("plan") or {}
            if plan:
                lines.append(f"Hypothesis: {plan.get('hypothesis')}")
                lines.append(f"Parent: {plan.get('parent_checkpoint')}")
                lines.append(f"Dataset: {plan.get('dataset')} · {format_number(plan.get('target_tokens'))} tokens")
            if exp.get("decision"):
                lines.append(f"Decision: {exp.get('decision')}")
            if exp.get("eval"):
                lines.append(f"Eval: {json.dumps(exp.get('eval'))}")
            if exp.get("failure"):
                lines.append(f"Failure: {exp.get('failure')}")
            if exp.get("reason") and exp.get("status") in ("paused-provider", "declined"):
                lines.append(f"Note: {exp.get('reason')}")
            if exp.get("examples"):
                for ex in exp["examples"][:3]:
                    lines.append(f"Prompt: {ex.get('prompt')}")
                    lines.append(f"  Approved: {str(ex.get('approved_reply'))[:220]!r}")
                    lines.append(f"  Candidate: {str(ex.get('candidate_reply'))[:220]!r}")
            lines.append("")
        report_md = "\n".join(lines)
        base = Path(self.root) / SESSIONS_DIR / session["session_id"]
        try:
            (base.with_suffix(".report.md")).write_text(report_md, encoding="utf-8")
            atomic_json(base.with_suffix(".report.json"),
                        {"session_id": session["session_id"], "status": session.get("status"),
                         "stop_reason": session.get("stop_reason"),
                         "experiments": session.get("experiments", [])})
            session["report"] = str(base.with_suffix(".report.md"))
        except OSError:
            pass


def preview_research(root: Path = ROOT, *, max_tokens: int, max_seconds: float,
                     max_experiments: int = 3,
                     allowed_datasets: list | None = None,
                     stage_token_cap: int = 1_000_000,
                     allow_repetition: bool = False,
                     mixture: str = "", use_ai: bool = False,
                     goal: str = "") -> dict:
    """Pure preflight preview for a research session. Creates nothing,
    contacts no provider, starts no training.

    Runs the same inspect → heuristic-propose → validate chain the supervisor
    will use for its first local decision, so the preview is accurate for
    local mode. For AI mode the supervisor's live proposal happens at start,
    but under these same validated limits (preview says so explicitly).
    """
    root = Path(root)
    datasets = _normalize_allowed_datasets(allowed_datasets)
    if int(max_tokens) <= 0 or float(max_seconds) <= 0:
        raise ValueError("Session needs a positive token budget and time limit.")
    try:
        stage_cap = int(stage_token_cap)
    except (TypeError, ValueError):
        raise ValueError("Per-stage token cap must be a whole number.") from None
    if stage_cap <= 0:
        raise ValueError("Per-stage token cap must be positive.")
    mix_text = str(mixture or "").strip()
    if "mixture" in datasets and mix_text:
        from .gui_core import validate_mixture

        validate_mixture(mix_text)
    summary = inspect_project(root)
    limits = {"max_tokens": int(max_tokens), "max_seconds": float(max_seconds),
              "max_experiments": int(max_experiments),
              "max_api_requests": 25, "max_api_cost_usd": 0.0,
              "allowed_ops": ["train", "evaluate", "compare"],
              "allowed_datasets": datasets, "stage_token_cap": stage_cap,
              "allow_repetition": bool(allow_repetition),
              "mixture": mix_text, "_tokens_used": 0}
    proposal = local_heuristic_proposal(summary, limits=limits)
    validated = None
    validation_error = None
    if not proposal.get("stop"):
        if mix_text and proposal.get("dataset") == "mixture" and not proposal.get("mixture"):
            proposal = dict(proposal, mixture=mix_text)
        try:
            validated = validate_proposal(proposal, summary=summary, limits=limits, root=root)
        except ValueError as exc:
            validation_error = str(exc)
    passes = {}
    for name in datasets:
        size = _single_pass_tokens(name, root)
        if size is not None:
            passes[name] = size
    return {
        "goal": str(goal or "")[:1000],
        "use_ai": bool(use_ai),
        "approved": summary.get("approved"),
        "assistant_head": summary.get("assistant_head"),
        "last_promotion_eval": summary.get("last_promotion_eval"),
        "limits": {k: limits[k] for k in ("max_tokens", "max_seconds", "max_experiments",
                                          "allowed_datasets", "stage_token_cap",
                                          "allow_repetition", "mixture")},
        "single_pass_tokens": passes,
        "first_decision": proposal,
        "first_decision_valid": validated,
        "first_decision_error": validation_error,
        "ai_note": ("AI supervisor proposes live at start under these same limits; "
                    "this preview shows the local-heuristic equivalent only."
                    if use_ai else None),
        "starts_nothing": True,
    }


def start_session(session_id: str, root: Path = ROOT, ctx=None) -> dict:
    session = read_session(session_id, root)
    if session.get("status") == "running":
        raise ValueError(f"Research session '{session_id}' is already running.")
    if session.get("status") == "done":
        raise ValueError(f"Research session '{session_id}' is done and cannot resume.")
    if gpu_session.is_busy(root):
        owner = gpu_session.describe(root)
        raise ValueError(f"GPU is busy ({owner.get('kind')}: {owner.get('label')}). Stop it first.")
    _reconcile_interrupted_experiments(session, root)
    session = read_session(session_id, root)
    session["status"] = "running"
    if not session.get("started_at"):
        session["started_at"] = time.time()
    # Resuming never replans: the original deadline, token budget, completed
    # experiments, and provider history all carry over untouched.
    session["stop_reason"] = None
    provider_state = session.setdefault("provider_state",
                                        {"paused": False, "reason": None,
                                         "paused_at": None, "attempts": [],
                                         "last_model": None})
    if provider_state.get("paused"):
        provider_state.update({"paused": False, "resumed_at": time.time()})
    atomic_json(_session_path(session_id, root),
                {**session, "updated_at": time.time()})
    thread = threading.Thread(target=Supervisor(session_id, root, ctx or Ctx()).run,
                              daemon=True, name=f"research-{session_id}")
    _threads[session_id] = thread
    thread.start()
    return {"session_id": session_id, "status": "running"}


def _reconcile_interrupted_experiments(session: dict, root: Path) -> None:
    """Mark experiments left mid-flight by a crash/restart — never re-run them.

    For each experiment stuck in planned/training/evaluating: if its trainer
    lock is still held, refuse the resume (that stage is genuinely still
    running elsewhere); otherwise record it as interrupted with its saved
    checkpoint/tokens intact. The loop then continues at the next index, and
    a future proposal may name the interrupted checkpoint as a parent — so
    no checkpoint progress is lost and no stage is duplicated.
    """
    from .gui_core import trainer_active

    root = Path(root)
    changed = False
    for exp in session.get("experiments", []):
        if exp.get("status") not in ("planned", "training", "evaluating"):
            continue
        run_dir = exp.get("run_dir")
        run_path = Path(run_dir) if run_dir else None
        if run_path is not None and run_path.exists():
            try:
                if trainer_active(run_path):
                    raise ValueError(
                        f"Experiment {exp.get('index')} stage '{run_path.name}' is still "
                        "running. Stop it (or wait for it) before resuming this session — "
                        "resuming now would duplicate the stage.")
            except ValueError:
                raise
            except OSError:
                pass
        exp["status"] = "interrupted"
        exp["failure"] = (
            "Interrupted by backend restart/stop before a terminal state; "
            "saved checkpoints (if any) remain and may parent a later stage. "
            "Not re-run on resume.")
        changed = True
    if changed:
        _save(session, root)


def stop_session(session_id: str, root: Path = ROOT) -> dict:
    session = read_session(session_id, root)
    if session.get("status") not in ("running",):
        return {"session_id": session_id, "already_stopped": True,
                "status": session.get("status")}
    session["status"] = "stopped"
    session["stop_reason"] = "Stopped by operator; completed work preserved, stages never duplicated on resume."
    _save(session, root)
    for exp in session.get("experiments", []):
        if exp.get("status") == "training" and exp.get("run_dir"):
            try:
                from .gui_core import request_stop

                request_stop(Path(exp["run_dir"]))
            except OSError:
                pass
    return {"session_id": session_id, "already_stopped": False, "status": "stopped"}


def session_overview(session: dict | None) -> dict:
    if not session:
        return {"status": "none"}
    return {
        "session_id": session.get("session_id"),
        "status": session.get("status"),
        "goal": session.get("goal"),
        "use_ai": session.get("use_ai"),
        "limits": session.get("limits"),
        "budget_used": session.get("budget_used"),
        "provider_state": session.get("provider_state", {"paused": False}),
        "experiments": [
            {**{k: e.get(k) for k in ("index", "status", "run_dir", "checkpoint",
                                     "tokens_trained", "decision", "failure",
                                     "reason", "eval", "eval_report",
                                     "supervisor_model", "supervisor_attempts")},
             "plan": ({k: (e.get("plan") or {}).get(k) for k in
                       ("hypothesis", "dataset", "mixture", "target_tokens",
                        "max_seconds", "eval_plan", "expected_risks",
                        "stop_conditions", "parent_checkpoint")}
                      if e.get("plan") else None)}
            for e in session.get("experiments", [])
        ],
        "stop_reason": session.get("stop_reason"),
        "report": session.get("report"),
    }
