from pathlib import Path
from types import SimpleNamespace

import torch

from adamlm import eval_full
from adamlm.bpe import BPETokenizer
from adamlm.inference import build_chat_prompt, extract_response
from adamlm.evaluation import (
    classify_completion,
    greedy_completion,
    load_suite,
    score_choices,
)
from adamlm.tokenizer import ByteTokenizer


class AlwaysAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(block_size=64)

    def forward(self, ids):
        logits = torch.full((*ids.shape, 256), -10.0, device=ids.device)
        logits[:, :, ord("A")] = 10.0
        return logits, None


def test_frozen_suite_is_valid_and_unique():
    suite, digest = load_suite("config/eval-suite.json")
    assert len(suite["items"]) >= 25
    assert len(digest) == 64
    assert {item["category"] for item in suite["items"]} >= {
        "english_grammar",
        "coherent_generation",
        "general_knowledge",
        "computer_windows",
        "reasoning_instruction",
        "unsupported_questions",
    }


def test_greedy_generation_and_choice_scoring_are_deterministic():
    model = AlwaysAModel()
    tokenizer = ByteTokenizer()
    first = greedy_completion(model, tokenizer, "Q:", 3)
    second = greedy_completion(model, tokenizer, "Q:", 3)
    assert first == second == ("AAA", "Q:AAA")
    scores = score_choices(model, tokenizer, "Q:", ["A", "B"])
    assert scores[0]["mean_log_probability"] > scores[1]["mean_log_probability"]
    assert model.training


def test_behavior_labels_and_reasons_are_transparent():
    supported = {"task": "generation", "expected": ["Paris"], "match": "contains"}
    unsupported = {"task": "generation", "expect_abstention": True}
    abstentions = ["I don't know", "not enough information"]
    continuations = [r"^\s*[\"“]", r"\bonce upon a time\b"]

    assert classify_completion(supported, " Paris.", abstentions, continuations)[:2] == ("answer", True)
    assert classify_completion(supported, "I don't know.", abstentions, continuations)[:2] == ("abstention", False)
    assert classify_completion(unsupported, "There is not enough information.", abstentions, continuations)[:2] == ("abstention", True)
    assert classify_completion(unsupported, ' "Once upon a time,"', abstentions, continuations)[:2] == ("prompt_continuation", False)
    assert classify_completion(unsupported, "It is Tuesday.", abstentions, continuations)[:2] == ("hallucination", False)


def _probe_row(reply, ref="red apple green"):
    return {"id": "a", "kind": "instruction", "reply": reply, "reference": ref, "checks": {}}


def test_scorecard_reports_validation_loss_without_promoting_on_it():
    tied_old = [_probe_row("red apple green")]
    tied_new = [_probe_row("red apple green")]
    card = eval_full.scorecard(tied_old, tied_new, old_validation_loss=2.9,
                               new_validation_loss=1.1)
    assert card["promote"] is True  # probes tie; the lower loss is only reported
    assert card["old_validation_loss"] == 2.9
    assert card["new_validation_loss"] == 1.1
    assert "never promotes" in card["note"]
    regressed = eval_full.scorecard(
        tied_old, [_probe_row("zzz qqq")],
        old_suite={"accuracy": 0.5}, new_suite={"accuracy": 0.5},
        old_validation_loss=2.9, new_validation_loss=1.1)
    assert regressed["promote"] is False  # better loss cannot rescue a skill regression


def test_read_validation_loss_prefers_recorded_status(tmp_path):
    import json
    run_dir = tmp_path / "results" / "r1"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "step_00000100.pt"
    ckpt.write_bytes(b"x")
    (run_dir / "status.json").write_text(json.dumps({"validation_loss": 2.5259}),
                                         encoding="utf-8")
    assert eval_full.read_validation_loss(ckpt, root=tmp_path) == 2.5259
    assert eval_full.read_validation_loss(str(ckpt), root=tmp_path) == 2.5259
    (run_dir / "status.json").write_text(json.dumps({}), encoding="utf-8")
    assert eval_full.read_validation_loss(ckpt, root=tmp_path) is None
    assert eval_full.read_validation_loss(tmp_path / "missing.pt", root=tmp_path) is None


def test_chat_prompt_uses_exact_token_budget_and_preserves_roles():
    tokenizer = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    messages = [
        {"role": "assistant", "text": "Earlier assistant text"},
        {"role": "user", "text": "x" * 2000},
        {"role": "user", "text": "What were we talking about?"},
    ]
    built = build_chat_prompt(messages, tokenizer, block_size=128, max_new_tokens=32)
    assert built["prompt_tokens"] <= 96
    assert built["prompt_tokens"] + built["max_new_tokens"] <= built["context_limit"]
    assert built["trimmed"]
    # Roles come from stored messages rather than alternating by array index.
    short = build_chat_prompt(messages[:1] + messages[-1:], tokenizer, 512, 32)
    assert "Context:\nAssistant:\nEarlier assistant text" in short["prompt"]


def test_response_extraction_stops_before_leaked_user_turn():
    prompt = "User:\nhi\n\nAssistant:\n"
    raw = prompt + "Hello!\n\nUser:\nsecond turn"
    assert extract_response(raw, prompt) == "Hello!"
