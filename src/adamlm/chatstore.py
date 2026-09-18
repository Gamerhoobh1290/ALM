"""Saved conversations and approved user-feedback examples.

Three strictly separate concepts:
1. Conversation context — previous turns included in the inference prompt
   (in-memory, per conversation, trimmed to the 512-token budget).
2. Saved conversation history — messages stored locally for the user's
   convenience. Explicitly opt-in per conversation; never training data.
3. Learning from conversations — user-corrected examples that become
   eligible for a future training experiment ONLY after explicit approval,
   formatted with the compatible SFT template and response-only masking.

Deleting stored conversations removes the stored copy; it never claims to
remove information already learned into model weights.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from .gui_core import ROOT, atomic_json, read_json

CONVERSATIONS_DIR = "results/conversations"
FEEDBACK_DIR = "results/feedback"


def _conversations_dir(root: Path) -> Path:
    path = Path(root) / CONVERSATIONS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _feedback_dir(root: Path) -> Path:
    path = Path(root) / FEEDBACK_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clean_messages(messages) -> list[dict]:
    cleaned = []
    for message in messages or []:
        role = str((message or {}).get("role") or "").strip().lower()
        text = str((message or {}).get("text") or "")
        if role not in ("user", "assistant") or not text.strip():
            continue
        cleaned.append({"role": role, "text": text[:8000]})
    return cleaned[-60:]


def save_conversation(root: Path = ROOT, *, messages, title: str = "",
                      approved_version: str | None = None,
                      conversation_id: str | None = None) -> dict:
    """Persist one conversation snapshot. Never marks it as training data."""
    root = Path(root)
    cleaned = _clean_messages(messages)
    if not cleaned:
        raise ValueError("Nothing to save: the conversation is empty.")
    cid = conversation_id or f"chat-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    record = {
        "conversation_id": cid,
        "title": str(title or "")[:120],
        "saved_at": time.time(),
        "approved_version": approved_version,
        "message_count": len(cleaned),
        "messages": cleaned,
        "training_use": "never — saved history only; promotion to a training example requires explicit feedback approval",
    }
    atomic_json(_conversations_dir(root) / f"{cid}.json", record)
    return {"conversation_id": cid, "message_count": len(cleaned), "saved_at": record["saved_at"]}


def list_conversations(root: Path = ROOT, *, limit: int = 20) -> list[dict]:
    directory = Path(root) / CONVERSATIONS_DIR
    if not directory.exists():
        return []
    records = []
    for path in directory.glob("*.json"):
        data = read_json(path, {})
        if data:
            records.append({k: data.get(k) for k in
                            ("conversation_id", "title", "saved_at", "approved_version", "message_count")})
    records.sort(key=lambda r: r.get("saved_at") or 0, reverse=True)
    return records[: max(1, min(int(limit or 20), 100))]


def load_conversation(root: Path = ROOT, conversation_id: str = "") -> dict:
    path = Path(root) / CONVERSATIONS_DIR / f"{conversation_id}.json"
    data = read_json(path, {})
    if not data:
        raise ValueError(f"Saved conversation '{conversation_id}' was not found.")
    return data


def delete_conversation(root: Path = ROOT, conversation_id: str = "") -> dict:
    path = Path(root) / CONVERSATIONS_DIR / f"{conversation_id}.json"
    try:
        path.unlink()
    except OSError:
        raise ValueError(f"Saved conversation '{conversation_id}' was not found.") from None
    return {"deleted": conversation_id,
            "note": "Stored copy removed. Model weights are unchanged by this action."}


def submit_feedback(root: Path = ROOT, *, conversation_id: str = "",
                    assistant_text: str = "", corrected_text: str = "",
                    note: str = "") -> dict:
    """Record a user correction as a PENDING example (not yet training data)."""
    if not corrected_text.strip():
        raise ValueError("Provide the corrected answer before submitting feedback.")
    if corrected_text.strip() == assistant_text.strip():
        raise ValueError("The correction is identical to the model reply; nothing to learn.")
    root = Path(root)
    fid = f"feedback-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    record = {
        "feedback_id": fid,
        "conversation_id": conversation_id,
        "created_at": time.time(),
        "status": "pending",
        "assistant_text": assistant_text[:4000],
        "corrected_text": corrected_text[:4000],
        "note": str(note or "")[:500],
        "training_use": "pending — becomes eligible for a future SFT stage only after explicit approval",
    }
    atomic_json(_feedback_dir(root) / f"{fid}.json", record)
    return {"feedback_id": fid, "status": "pending"}


def review_feedback(root: Path = ROOT, *, feedback_id: str = "", approve: bool) -> dict:
    """Approve (eligible for future training) or reject a feedback example."""
    path = Path(root) / FEEDBACK_DIR / f"{feedback_id}.json"
    data = read_json(path, {})
    if not data:
        raise ValueError(f"Feedback '{feedback_id}' was not found.")
    data["status"] = "approved" if approve else "rejected"
    data["reviewed_at"] = time.time()
    if approve:
        data["training_use"] = ("approved — eligible for a future SFT stage with response-only "
                                "loss masking; held-out feedback stays held out")
    else:
        data["training_use"] = "rejected — never used for training"
    atomic_json(path, data)
    return {"feedback_id": feedback_id, "status": data["status"]}


def list_feedback(root: Path = ROOT, *, status: str | None = None) -> list[dict]:
    directory = Path(root) / FEEDBACK_DIR
    if not directory.exists():
        return []
    records = []
    for path in directory.glob("*.json"):
        data = read_json(path, {})
        if not data:
            continue
        if status and data.get("status") != status:
            continue
        records.append({k: data.get(k) for k in
                        ("feedback_id", "conversation_id", "created_at", "reviewed_at",
                         "status", "note")})
    records.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return records[:100]


def approved_examples(root: Path = ROOT) -> list[dict]:
    """Approved corrections formatted for a future SFT stage (read-only view)."""
    directory = Path(root) / FEEDBACK_DIR
    if not directory.exists():
        return []
    examples = []
    for path in directory.glob("*.json"):
        data = read_json(path, {})
        if data and data.get("status") == "approved" and data.get("corrected_text", "").strip():
            examples.append({
                "feedback_id": data.get("feedback_id"),
                "instruction": "",
                "response": data.get("corrected_text", "")[:4000],
                "source": f"user-feedback:{data.get('feedback_id')}",
            })
    return examples
