"""Extended local evaluation: candidate vs approved scorecards.

Combines three honest signals (all local, all fixed-seed, all identical
settings for both checkpoints):
1. Assistant skill probes — ``scripts/eval_assistant.py`` held-out
   greetings / instructions / follow-ups with hygiene checks (CPU).
2. Capability suite — ``config/eval-suite.json`` via ``evaluation.py``
   (grammar, coherence, knowledge, reasoning, unsupported/abstention).
3. Retention signal — capability accuracy on general-language categories
   plus each checkpoint's recorded validation loss (reported, never used
   as a promotion criterion alone).

Unigram F1 stays supplementary: a reply that copies reference words while
being factually wrong is NOT automatically counted successful — the
scorecard always shows raw replies beside the numbers.

Promotion rule (conservative, matches auto_train):
promote only on hygiene pass + win-or-tie on both skills + no capability
regression beyond a small tolerance. Anything else keeps the default.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

from .gui_core import ROOT, atomic_json, read_json

EVALS_DIR = "results/evals"
SUITE_PATH = "config/eval-suite.json"
CAPABILITY_TOLERANCE = 0.02  # allowed accuracy dip before it counts as regression


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def unigram_f1(reply: str, reference: str) -> float:
    hyp, ref = Counter(_words(reply)), Counter(_words(reference))
    if not hyp or not ref:
        return 0.0
    overlap = sum(min(hyp[w], ref[w]) for w in hyp.keys() & ref.keys())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(hyp.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def _skill_split(rows: list[dict]) -> dict:
    buckets = {"conv": [], "instr": []}
    for row in rows or []:
        if "turns" in row:
            for turn in row["turns"]:
                buckets["conv"].append({"reply": turn.get("reply", ""), "reference": None,
                                       "checks": ((row.get("checks") or [{}])[0]
                                                  if isinstance(row.get("checks"), list) else {})})
            continue
        bucket = "instr" if row.get("kind") == "instruction" else "conv"
        checks = row.get("checks") or {}
        buckets[bucket].append({"reply": row.get("reply", ""),
                                "reference": row.get("reference"),
                                "checks": checks if isinstance(checks, dict) else {}})
    return buckets


def _hygiene(entries: list[dict]) -> list[str]:
    problems = []
    for i, entry in enumerate(entries):
        reply = entry.get("reply", "")
        checks = entry.get("checks") or {}
        if not reply or not reply.strip():
            problems.append(f"reply {i}: empty")
        if "\ufffd" in reply:
            problems.append(f"reply {i}: replacement character")
        if "User:" in reply:
            problems.append(f"reply {i}: prompt echo")
        for key in ("non_empty", "no_replacement_char", "no_prompt_echo", "no_runaway_repeat"):
            if key in checks and not checks[key]:
                problems.append(f"reply {i}: failed {key}")
        ratio = checks.get("printable_ratio")
        if isinstance(ratio, (int, float)) and ratio < 0.95:
            problems.append(f"reply {i}: printable ratio {ratio}")
    return problems


def _mean_f1(entries: list[dict]) -> tuple[float, int]:
    scores = [unigram_f1(e["reply"], e["reference"]) for e in entries if e.get("reference")]
    if not scores:
        return 0.0, 0
    return sum(scores) / len(scores), len(scores)


def assistant_scores(rows: list[dict]) -> dict:
    buckets = _skill_split(rows)
    entries = buckets["conv"] + buckets["instr"]
    instr, instr_n = _mean_f1(buckets["instr"])
    conv, conv_n = _mean_f1(buckets["conv"])
    return {
        "instr_f1": round(instr, 4), "instr_n": instr_n,
        "conv_f1": round(conv, 4), "conv_n": conv_n,
        "hygiene_problems": _hygiene(entries),
        "rows": len(rows),
    }


def run_assistant_probes(checkpoint: str, *, label: str, timeout: int = 1800,
                         root: Path = ROOT) -> list[dict]:
    """Run the held-out assistant probes in-process on CPU (no GPU claim)."""
    import sys

    scripts = Path(root) / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(Path(root) / "src"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("adam_eval_assistant", scripts / "eval_assistant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_checkpoint(label, checkpoint)
    return result["rows"]


def run_capability_suite(checkpoint: str, *, root: Path = ROOT,
                         device: str = "cpu") -> dict:
    """Run config/eval-suite.json greedily (deterministic, identical for A/B)."""
    from . import evaluation as ev

    suite_path = str(Path(root / SUITE_PATH))
    report = ev.evaluate_checkpoints([checkpoint], suite_path, output_path=None, device=device)
    entry = (report.get("checkpoints") or [{}])[0]
    summary = entry.get("summary", {}) or {}
    return {
        "accuracy": summary.get("accuracy"),
        "correct": summary.get("correct"),
        "scored": summary.get("scored"),
        "by_category": summary.get("by_category", {}),
        "labels": summary.get("labels", {}),
        "results": entry.get("results", []),
    }


def read_validation_loss(checkpoint: str | Path, *, root: Path = ROOT) -> float | None:
    """Recorded validation loss for a checkpoint's run dir, if any (read-only).

    Reported for context only — a lower loss alone never justifies promotion.
    Returns None when the run dir or value is missing/unusable.
    """
    try:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = Path(root) / path
        run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
        value = float(read_json(run_dir / "status.json", {}).get("validation_loss"))
    except (OSError, TypeError, ValueError):
        return None
    return value if value == value else None


def scorecard(old_rows: list[dict], new_rows: list[dict], *,
              old_suite: dict | None = None, new_suite: dict | None = None,
              old_label: str = "approved", new_label: str = "candidate",
              old_validation_loss=None, new_validation_loss=None) -> dict:
    """Build the promotion scorecard from two consistent eval runs."""
    old_s, new_s = assistant_scores(old_rows), assistant_scores(new_rows)
    problems = new_s["hygiene_problems"]
    reasons: list[str] = []
    promote = True
    if problems:
        promote = False
        reasons.append(f"held: {len(problems)} hygiene problem(s) in new replies ({problems[0]})")
    if new_s["instr_f1"] < old_s["instr_f1"]:
        promote = False
        reasons.append(f"held: instructions regressed (new F1 {new_s['instr_f1']:.3f} < old F1 {old_s['instr_f1']:.3f})")
    if new_s["conv_f1"] < old_s["conv_f1"]:
        promote = False
        reasons.append(f"held: conversation regressed (new F1 {new_s['conv_f1']:.3f} < old F1 {old_s['conv_f1']:.3f})")
    suite_delta = None
    if old_suite and new_suite:
        try:
            old_acc = float(old_suite.get("accuracy") or 0)
            new_acc = float(new_suite.get("accuracy") or 0)
            suite_delta = round(new_acc - old_acc, 4)
            if new_acc + CAPABILITY_TOLERANCE < old_acc:
                promote = False
                reasons.append(f"held: general capability regressed ({old_acc:.3f} → {new_acc:.3f})")
        except (TypeError, ValueError):
            suite_delta = None
    if promote:
        reasons.append(f"promotable: instructions F1 {old_s['instr_f1']:.3f}→{new_s['instr_f1']:.3f}, "
                       f"conversation F1 {old_s['conv_f1']:.3f}→{new_s['conv_f1']:.3f}, all hygiene checks passing")
    return {
        "old_label": old_label,
        "new_label": new_label,
        "instr_f1_old": old_s["instr_f1"], "instr_f1_new": new_s["instr_f1"], "instr_n": new_s["instr_n"],
        "conv_f1_old": old_s["conv_f1"], "conv_f1_new": new_s["conv_f1"], "conv_n": new_s["conv_n"],
        "hygiene_problems": problems,
        "suite_accuracy_old": (old_suite or {}).get("accuracy"),
        "suite_accuracy_new": (new_suite or {}).get("accuracy"),
        "suite_delta": suite_delta,
        "suite_by_category_old": (old_suite or {}).get("by_category"),
        "suite_by_category_new": (new_suite or {}).get("by_category"),
        "old_validation_loss": old_validation_loss,
        "new_validation_loss": new_validation_loss,
        "promote": promote,
        "reasons": reasons,
        "reason": "; ".join(reasons),
        "note": ("Unigram F1 is supplementary; inspect the saved example replies. "
                 "Lower validation loss alone never promotes."),
    }


def _example_pairs(old_rows: list[dict], new_rows: list[dict], *, limit: int = 6) -> list[dict]:
    pairs = []
    for old, new in zip(old_rows or [], new_rows or []):
        if len(pairs) >= limit:
            break
        if "turns" in old or "turns" in new:
            continue
        pairs.append({
            "id": old.get("id") or new.get("id"),
            "kind": old.get("kind") or new.get("kind"),
            "prompt": old.get("instruction") or new.get("instruction"),
            "reference": (old.get("reference") or "")[:300],
            "approved_reply": (old.get("reply") or "")[:600],
            "candidate_reply": (new.get("reply") or "")[:600],
        })
    return pairs


def compare_checkpoints(old_checkpoint: str, new_checkpoint: str, *,
                        root: Path = ROOT, device: str = "cpu",
                        old_label: str = "approved", new_label: str = "candidate",
                        include_suite: bool = True,
                        old_validation_loss=None, new_validation_loss=None) -> dict:
    """Full local A/B: identical prompts, seeds, and settings on both sides.

    Validation losses are filled from each checkpoint's recorded status.json
    when not given explicitly; they are reported, never promotion criteria.
    """
    old_rows = run_assistant_probes(old_checkpoint, label=old_label, root=root)
    new_rows = run_assistant_probes(new_checkpoint, label=new_label, root=root)
    old_suite = run_capability_suite(old_checkpoint, root=root, device=device) if include_suite else None
    new_suite = run_capability_suite(new_checkpoint, root=root, device=device) if include_suite else None
    if old_validation_loss is None:
        old_validation_loss = read_validation_loss(old_checkpoint, root=root)
    if new_validation_loss is None:
        new_validation_loss = read_validation_loss(new_checkpoint, root=root)
    card = scorecard(old_rows, new_rows, old_suite=old_suite, new_suite=new_suite,
                     old_label=old_label, new_label=new_label,
                     old_validation_loss=old_validation_loss,
                     new_validation_loss=new_validation_loss)
    return {
        "format_version": 1,
        "compared_at": time.time(),
        "old_checkpoint": str(old_checkpoint),
        "new_checkpoint": str(new_checkpoint),
        "settings": "assistant probes: tokens=60 temp=0.8 top_k=40 seed=42 CPU; capability suite: greedy_argmax",
        "scorecard": card,
        "examples": _example_pairs(old_rows, new_rows),
        "old_rows": old_rows,
        "new_rows": new_rows,
    }


def save_report(report: dict, *, root: Path = ROOT, name: str | None = None) -> Path:
    directory = Path(root) / EVALS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = directory / (name or f"compare-{stamp}.json")
    atomic_json(path, report)
    return path


def load_report(name: str, *, root: Path = ROOT) -> dict:
    data = read_json(Path(root) / EVALS_DIR / name, {})
    if not data:
        raise ValueError(f"Evaluation report '{name}' was not found.")
    return data


def list_reports(*, root: Path = ROOT, limit: int = 20) -> list[dict]:
    directory = Path(root) / EVALS_DIR
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        data = read_json(path, {})
        card = (data or {}).get("scorecard", {}) or {}
        items.append({
            "report": path.name,
            "compared_at": data.get("compared_at"),
            "promote": card.get("promote"),
            "reason": card.get("reason"),
        })
    return items
