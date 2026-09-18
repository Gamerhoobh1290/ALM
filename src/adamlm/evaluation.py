"""Reproducible, read-only checkpoint evaluation for AdamLM."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

import torch

from .bpe import BPETokenizer
from .model import DecoderTransformer, ModelConfig
from .tokenizer import ByteTokenizer


BYTE_TOKENIZER_ID = "utf8-byte-v1"
VALID_TASKS = {"generation", "paired_choice", "sample"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {normalize_text(text)} "
    normalized_phrase = normalize_text(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def _matches_expected(item: dict, completion: str) -> bool:
    expected = item.get("expected", [])
    match = item.get("match", "contains")
    if match == "exact":
        value = normalize_text(completion)
        return any(value == normalize_text(candidate) for candidate in expected)
    if match == "contains":
        return any(_contains_phrase(completion, candidate) for candidate in expected)
    if match == "regex":
        return any(re.search(pattern, completion, re.IGNORECASE) is not None for pattern in expected)
    raise ValueError(f"Unknown match method: {match}")


def classify_completion(
    item: dict,
    completion: str,
    abstention_markers: list[str],
    continuation_patterns: list[str],
) -> tuple[str, bool | None, str]:
    """Assign one transparent behavior label using suite-owned rules."""
    if item["task"] == "sample":
        return "prompt_continuation", None, "qualitative sample; inspect the saved raw output"

    abstained = any(_contains_phrase(completion, marker) for marker in abstention_markers)
    continued = not completion.strip() or any(
        re.search(pattern, completion, re.IGNORECASE) is not None
        for pattern in continuation_patterns
    )
    if item.get("expect_abstention"):
        if abstained:
            return "abstention", True, "an explicit uncertainty marker was present"
        if continued:
            return "prompt_continuation", False, "no answer or a narrative continuation was produced"
        return "hallucination", False, "the model asserted an answer to an unsupported question"
    if _matches_expected(item, completion):
        return "answer", True, f"matched the suite's {item.get('match', 'contains')} answer rule"
    if abstained:
        return "abstention", False, "the question was supported but the model abstained"
    if continued:
        return "prompt_continuation", False, "no expected answer was given; output continued as narrative"
    return "hallucination", False, "the completion did not match an accepted answer"


def load_suite(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    raw = path.read_bytes()
    suite = json.loads(raw)
    if suite.get("version") != 1 or not isinstance(suite.get("items"), list):
        raise ValueError("Evaluation suite must have version 1 and an items list")
    seen: set[str] = set()
    for item in suite["items"]:
        item_id = item.get("id")
        task = item.get("task")
        if not item_id or item_id in seen:
            raise ValueError("Evaluation item IDs must be nonempty and unique")
        seen.add(item_id)
        if task not in VALID_TASKS or not item.get("category") or not item.get("prompt"):
            raise ValueError(f"Invalid evaluation item: {item_id}")
        if task == "paired_choice":
            choices = item.get("choices", [])
            correct = item.get("correct_choice")
            if len(choices) < 2 or not isinstance(correct, int) or not 0 <= correct < len(choices):
                raise ValueError(f"Invalid paired choices: {item_id}")
        elif task == "generation" and not (item.get("expected") or item.get("expect_abstention")):
            raise ValueError(f"Generation item needs expected answers or abstention: {item_id}")
        if item.get("match", "contains") not in {"exact", "contains", "regex"}:
            raise ValueError(f"Invalid match method: {item_id}")
    return suite, hashlib.sha256(raw).hexdigest()


@torch.no_grad()
def greedy_completion(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, str]:
    """Return completion and full decoded output using deterministic argmax decoding."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Evaluation prompts must encode to at least one token")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(ids[:, -model.config.block_size :])
            token = int(logits[:, -1].float().argmax(dim=-1).item())
            if token == getattr(tokenizer, "boundary_id", None):
                break
            generated.append(token)
            ids = torch.cat((ids, torch.tensor([[token]], device=device)), dim=1)
    finally:
        model.train(was_training)
    raw_output = tokenizer.decode(prompt_ids + generated)
    completion = raw_output[len(prompt) :] if raw_output.startswith(prompt) else tokenizer.decode(generated)
    return completion, raw_output


@torch.no_grad()
def score_choices(model, tokenizer, prompt: str, choices: list[str]) -> list[dict]:
    """Score choices by mean conditional log probability per choice token."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Choice prompts must encode to at least one token")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    scored = []
    try:
        for choice in choices:
            choice_ids = tokenizer.encode(choice)
            if not choice_ids:
                raise ValueError("Choices must encode to at least one token")
            full = prompt_ids + choice_ids
            if len(full) - 1 > model.config.block_size:
                raise ValueError("Prompt and choice exceed the model context")
            inputs = torch.tensor([full[:-1]], dtype=torch.long, device=device)
            logits, _ = model(inputs)
            start = len(prompt_ids) - 1
            candidate_logits = logits[0, start : start + len(choice_ids)].float()
            targets = torch.tensor(choice_ids, dtype=torch.long, device=device)
            token_logprobs = candidate_logits.log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
            scored.append(
                {
                    "choice": choice,
                    "token_count": len(choice_ids),
                    "mean_log_probability": token_logprobs.mean().item(),
                    "sum_log_probability": token_logprobs.sum().item(),
                }
            )
    finally:
        model.train(was_training)
    return scored


def _resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("step_*.pt"))
        if not candidates:
            raise ValueError(f"Checkpoint directory is empty: {path}")
        path = candidates[-1]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_checkpoint(path: str | Path, device: str):
    """Load a trusted local AdamLM checkpoint and verify its tokenizer identity."""
    path = _resolve_checkpoint(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state.get("model_config")
    if not config:
        raise ValueError(f"Checkpoint is missing model_config: {path}")
    model = DecoderTransformer(ModelConfig(**config)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    extra = state.get("extra", {})
    expected_tokenizer_hash = extra.get("tokenizer_sha256")
    if expected_tokenizer_hash:
        tokenizer_path = Path(extra.get("tokenizer_path", ""))
        if not tokenizer_path.is_file():
            project_path = Path(__file__).resolve().parents[2] / tokenizer_path
            tokenizer_path = project_path
        tokenizer = BPETokenizer(tokenizer_path)
        if tokenizer.sha256 != expected_tokenizer_hash:
            raise ValueError("Tokenizer hash does not match checkpoint")
        tokenizer_info = {
            "kind": "bpe",
            "path": str(tokenizer_path.resolve()),
            "sha256": tokenizer.sha256,
        }
    elif model.config.vocab_size == 256:
        tokenizer = ByteTokenizer()
        tokenizer_info = {
            "kind": "byte",
            "identity": BYTE_TOKENIZER_ID,
            "sha256": hashlib.sha256(BYTE_TOKENIZER_ID.encode()).hexdigest(),
        }
    else:
        raise ValueError("Non-byte checkpoint is missing its tokenizer identity")
    metadata = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "step": int(state.get("step", 0)),
        "model_config": config,
        "tokenizer": tokenizer_info,
    }
    return model, tokenizer, metadata


def _summarize(results: list[dict]) -> dict:
    labels = Counter(result["label"] for result in results)
    scored = [result for result in results if result["correct"] is not None]
    categories: dict[str, dict] = {}
    for category in sorted({result["category"] for result in results}):
        rows = [result for result in results if result["category"] == category]
        category_scored = [row for row in rows if row["correct"] is not None]
        correct = sum(row["correct"] is True for row in category_scored)
        categories[category] = {
            "items": len(rows),
            "scored": len(category_scored),
            "correct": correct,
            "accuracy": correct / len(category_scored) if category_scored else None,
        }
    correct = sum(result["correct"] is True for result in scored)
    return {
        "items": len(results),
        "scored": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "labels": dict(sorted(labels.items())),
        "by_category": categories,
    }


def evaluate_checkpoint(model, tokenizer, suite: dict, max_new_tokens: int | None = None) -> tuple[list[dict], dict]:
    results = []
    abstention_markers = suite.get("abstention_markers", [])
    continuation_patterns = suite.get("continuation_patterns", [])
    default_tokens = int(suite.get("generation", {}).get("max_new_tokens", 32))
    for item in suite["items"]:
        base = {key: item[key] for key in ("id", "category", "task", "prompt")}
        if item["task"] == "paired_choice":
            scores = score_choices(model, tokenizer, item["prompt"], item["choices"])
            selected = max(range(len(scores)), key=lambda index: scores[index]["mean_log_probability"])
            correct = selected == item["correct_choice"]
            results.append(
                {
                    **base,
                    "choices": scores,
                    "correct_choice": item["correct_choice"],
                    "selected_choice": selected,
                    "completion": item["choices"][selected],
                    "raw_output": item["prompt"] + item["choices"][selected],
                    "label": "answer" if correct else "hallucination",
                    "correct": correct,
                    "reason": "selected the highest mean conditional token log probability",
                }
            )
            continue
        tokens = max_new_tokens or int(item.get("max_new_tokens", default_tokens))
        completion, raw_output = greedy_completion(model, tokenizer, item["prompt"], tokens)
        label, correct, reason = classify_completion(
            item, completion, abstention_markers, continuation_patterns
        )
        results.append(
            {
                **base,
                "max_new_tokens": tokens,
                "evaluation_rule": (
                    {"manual_review": True}
                    if item["task"] == "sample"
                    else {
                        "expected": item.get("expected", []),
                        "match": item.get("match", "contains"),
                        "expect_abstention": bool(item.get("expect_abstention")),
                    }
                ),
                "completion": completion,
                "raw_output": raw_output,
                "label": label,
                "correct": correct,
                "reason": reason,
            }
        )
    return results, _summarize(results)


def evaluate_checkpoints(
    checkpoint_paths: list[str | Path],
    suite_path: str | Path,
    output_path: str | Path | None,
    device: str = "auto",
    max_new_tokens: int | None = None,
) -> dict:
    suite, suite_hash = load_suite(suite_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    report = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": {
            "path": str(Path(suite_path).resolve()),
            "sha256": suite_hash,
            "name": suite.get("name"),
            "version": suite["version"],
        },
        "settings": {
            "device": device,
            "decoding": "greedy_argmax",
            "choice_scoring": "mean conditional log probability per separately encoded choice token",
            "max_new_tokens_override": max_new_tokens,
        },
        "checkpoints": [],
    }
    seen = set()
    for requested in checkpoint_paths:
        resolved = _resolve_checkpoint(requested).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        model, tokenizer, metadata = load_checkpoint(resolved, device)
        results, summary = evaluate_checkpoint(model, tokenizer, suite, max_new_tokens)
        report["checkpoints"].append({"checkpoint": metadata, "summary": summary, "results": results})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, output_path)

    return report
