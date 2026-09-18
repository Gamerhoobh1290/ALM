"""ONE AdamLM: GPU session, version registry, chat store, research validation.

Hermetic unit tests only: temporary directories, no training, no GPU, no
network, no production results/ writes.
"""
import json
import os
import time
from pathlib import Path

import pytest

from adamlm import chatstore, gpu_session, research, versions, web
from adamlm.gui_core import RunInfo


def _make_run(root, name, *, tokens=1000, step=100, stage="sft", dataset="dailydialog",
              parent=None):
    run_dir = Path(root) / "results" / name
    (run_dir / "checkpoints").mkdir(parents=True)
    checkpoint = run_dir / "checkpoints" / f"step_{step:08d}.pt"
    checkpoint.write_bytes(b"fake-weights")
    launcher = {"target_tokens": tokens + 100, "stage": stage, "dataset": dataset}
    if parent:
        launcher["parent_checkpoint"] = parent
    (run_dir / "launcher-config.json").write_text(json.dumps(launcher), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps(
        {"state": "target_reached", "tokens": tokens, "target_tokens": tokens + 100,
         "stage": stage, "dataset": dataset, "step": step,
         "checkpoint_step": step}), encoding="utf-8")
    return run_dir, checkpoint


# ---- GPU session -----------------------------------------------------------

def _write_claim(root, *, kind="training", label="training:dead", run=None, age_seconds=0.0):
    import time as _time

    path = Path(root) / "results" / ".gpu-session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": kind, "label": label, "run": run,
                                "started_at": _time.time() - age_seconds}),
                    encoding="utf-8")


def test_gpu_acquire_release_and_refuse_double_claim(tmp_path):
    first = gpu_session.try_acquire(tmp_path, kind="training", label="training:a", run="a")
    assert first and first["kind"] == "training"
    assert gpu_session.try_acquire(tmp_path, kind="inference", label="inference:x") is None
    assert gpu_session.is_busy(tmp_path) is True
    assert gpu_session.release(tmp_path, kind="training", label="training:a") is True
    assert gpu_session.is_busy(tmp_path) is False


def test_gpu_stale_claim_reclaimed_after_owner_exit(tmp_path):
    # A training claim naming a run with no live trainer lock, past the
    # startup grace window, is stale and reclaimable — no PID checks needed.
    _write_claim(tmp_path, run="dead-run", age_seconds=3600.0)
    assert gpu_session.is_busy(tmp_path) is False
    assert gpu_session.force_release_stale(tmp_path) is True
    claim = gpu_session.try_acquire(tmp_path, kind="inference", label="inference:new")
    assert claim is not None
    gpu_session.release(tmp_path)


def test_gpu_adopt_hands_backend_claim_to_trainer(tmp_path):
    assert gpu_session.try_acquire(tmp_path, kind="training",
                                   label="training:myrun", run="myrun") is not None
    assert gpu_session.is_busy(tmp_path) is True
    adopted = gpu_session.adopt(tmp_path, kind="training",
                                label="training:myrun", run="myrun")
    assert adopted is not None  # same run: backend -> trainer handoff allowed
    other = gpu_session.try_acquire(tmp_path, kind="training",
                                    label="training:otherrun", run="otherrun")
    assert other is None  # a different live run must never steal the claim


def test_gpu_repeated_cycles_never_wedge(tmp_path):
    """Regression: acquire/refuse/release/stale sequences must terminate.

    A prior implementation probed dead PIDs with os.kill, whose repeated
    calls wedge this machine's process-handle path and hang the caller
    inside the C call. The current design uses only lock probes and age
    caps, so hammer it to prove no poisoning accumulates.
    """
    import time as _time

    for i in range(25):
        assert gpu_session.try_acquire(tmp_path, kind="training",
                                       label=f"training:a{i}", run="a") is not None
        assert gpu_session.try_acquire(tmp_path, kind="inference",
                                       label=f"inference:{i}") is None
        assert gpu_session.is_busy(tmp_path) is True
        assert gpu_session.release(tmp_path, kind="training") is True
        assert gpu_session.is_busy(tmp_path) is False
        _write_claim(tmp_path, run="dead-run", age_seconds=3600.0)
        assert gpu_session.is_busy(tmp_path) is False
        assert gpu_session.force_release_stale(tmp_path) is True
    assert gpu_session.is_busy(tmp_path) is False
    assert _time.time() > 0  # wall clock available; test reached the end


# ---- Version registry ------------------------------------------------------

def test_registry_labels_approved_candidate_and_unknown_ancestry(tmp_path, monkeypatch):
    import adamlm.versions as vmod

    _, approved_ckpt = _make_run(tmp_path, "auto-assistant-20260917-231947-s3",
                                 tokens=8000, step=2024)
    _make_run(tmp_path, "auto-assistant-20260917-233243-s1", tokens=9000, step=100)
    _make_run(tmp_path, "general-training-500m", tokens=500000,
              step=15259, stage="pretrain", dataset="mixture")
    pointer = {"run": "auto-assistant-20260917-231947-s3",
               "checkpoint": str(approved_ckpt),
               "previous": {"run": "old", "checkpoint": "C:/missing/step.pt"}}
    (tmp_path / "results" / "assistant_default.json").write_text(
        json.dumps(pointer), encoding="utf-8")
    import adamlm.auto_train as at

    monkeypatch.setattr(at, "ROOT", tmp_path)
    monkeypatch.setattr(vmod.gui_core, "ROOT", tmp_path)
    registry = versions.build_registry(tmp_path)
    by_approval = {}
    for entry in registry["versions"]:
        by_approval.setdefault(entry["approval"], []).append(entry["run"])
    assert "auto-assistant-20260917-231947-s3" in by_approval.get("approved", [])
    assert "general-training-500m" in by_approval.get("foundation", [])
    experimental = [e for e in registry["versions"]
                    if e["approval"] in ("experimental", "candidate")]
    assert experimental and all(e["ancestry"] == ["unknown"] or e["ancestry"]
                                for e in experimental)


def test_promote_refuses_regression_and_missing_checkpoint(tmp_path):
    _, ckpt = _make_run(tmp_path, "auto-assistant-x-s1", tokens=100, step=10)
    good = {"instr_f1_old": 0.3, "instr_f1_new": 0.35,
            "conv_f1_old": 0.1, "conv_f1_new": 0.1, "hygiene_problems": []}
    pointer = versions.promote_candidate(tmp_path, checkpoint=str(ckpt),
                                         eval_info=good, reason="test win")
    assert pointer["previous"] is None  # first approval has no previous
    _, ckpt2 = _make_run(tmp_path, "auto-assistant-x-s2", tokens=200, step=20)
    bad = dict(good, instr_f1_new=0.1)
    with pytest.raises(ValueError, match="regressed"):
        versions.promote_candidate(tmp_path, checkpoint=str(ckpt2),
                                   eval_info=bad, reason="should fail")
    with pytest.raises(ValueError, match="missing"):
        versions.promote_candidate(tmp_path, checkpoint=str(tmp_path / "nope.pt"),
                                   eval_info=good, reason="should fail")
    with pytest.raises(ValueError, match="missing evaluation"):
        versions.promote_candidate(tmp_path, checkpoint=str(ckpt2),
                                   eval_info=None, reason="incomplete")
    # Successful second promotion retains the first as rollback target.
    pointer2 = versions.promote_candidate(tmp_path, checkpoint=str(ckpt2),
                                          eval_info=good, reason="test win 2")
    assert pointer2["previous"]["checkpoint"] == pointer["checkpoint"]
    rolled = versions.rollback_to_previous(tmp_path, confirm=True)
    assert rolled["checkpoint"] == pointer["checkpoint"]
    with pytest.raises(ValueError, match="confirmation"):
        versions.rollback_to_previous(tmp_path, confirm=False)


def test_resolve_approved_requires_existing_file(tmp_path):
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "results" / "assistant_default.json").write_text(
        json.dumps({"run": "x", "checkpoint": str(tmp_path / "gone.pt")}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        versions.resolve_approved_checkpoint(tmp_path)


# ---- Chat store ------------------------------------------------------------

def test_conversations_and_feedback_are_separate_from_training(tmp_path):
    messages = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}]
    saved = chatstore.save_conversation(tmp_path, messages=messages, title="t")
    assert chatstore.list_conversations(tmp_path)[0]["conversation_id"] == saved["conversation_id"]
    loaded = chatstore.load_conversation(tmp_path, saved["conversation_id"])
    assert loaded["training_use"].startswith("never")
    feedback = chatstore.submit_feedback(
        tmp_path, conversation_id=saved["conversation_id"],
        assistant_text="hello", corrected_text="Hello! How can I help?")
    assert feedback["status"] == "pending"
    assert chatstore.approved_examples(tmp_path) == []
    reviewed = chatstore.review_feedback(tmp_path, feedback_id=feedback["feedback_id"], approve=True)
    assert reviewed["status"] == "approved"
    assert len(chatstore.approved_examples(tmp_path)) == 1
    with pytest.raises(ValueError):
        chatstore.submit_feedback(tmp_path, assistant_text="x", corrected_text="x")
    deleted = chatstore.delete_conversation(tmp_path, saved["conversation_id"])
    assert "weights are unchanged" in deleted["note"]


# ---- Research validation ---------------------------------------------------

def test_research_proposal_validation_rejects_overreach(tmp_path):
    _, ckpt = _make_run(tmp_path, "auto-assistant-20260917-231947-s3",
                         tokens=8000, step=2024)
    summary = {"approved": {"checkpoint": str(ckpt)}}
    limits = {"max_tokens": 1000000, "_tokens_used": 0, "max_seconds": 3600,
              "allowed_ops": ["train", "evaluate", "compare"]}
    base = {"hypothesis": "h", "parent_checkpoint": str(ckpt), "dataset": "dolly",
            "target_tokens": 500000, "max_seconds": 600, "eval_plan": "assistant probes"}
    assert research.validate_proposal(dict(base), summary=summary, limits=limits,
                                          root=tmp_path)["dataset"] == "dolly"
    with pytest.raises(ValueError, match="budget"):
        research.validate_proposal({**base, "target_tokens": 5000000},
                                   summary=summary, limits=limits, root=tmp_path)
    with pytest.raises(ValueError, match="Unsupported dataset"):
        research.validate_proposal({**base, "dataset": "shady"}, summary=summary,
                                   limits=limits, root=tmp_path)
    evil = tmp_path / "evil.pt"
    evil.write_bytes(b"not-a-checkpoint")
    with pytest.raises(ValueError, match="not a known saved checkpoint"):
        research.validate_proposal({**base, "parent_checkpoint": str(evil)},
                                   summary=summary, limits=limits, root=tmp_path)


def test_local_heuristic_declines_without_hypothesis():
    stopped = research.local_heuristic_proposal(
        {"approved": {}, "assistant_head": {}, "last_promotion_eval": {}},
        limits={"max_tokens": 100, "_tokens_used": 0, "max_seconds": 60})
    assert stopped.get("stop") is True


# ---- Server-side chat composition ------------------------------------------
def test_chat_prompt_trims_honestly_and_requires_user_last():
    long_turn = "word " * 2000
    messages = [{"role": "user", "text": long_turn}] + [
        {"role": role, "text": f"turn {i} " + "x " * 500}
        for i, role in enumerate(["assistant", "user"] * 6)
    ] + [{"role": "user", "text": "final question"}]
    composed = web._compose_chat_prompt(messages, 60)
    assert composed["total_turns"] == len(messages)
    assert composed["kept_turns"] <= composed["total_turns"]
    assert "Assistant:" in composed["prompt"]
    with pytest.raises(ValueError):
        web._compose_chat_prompt([{"role": "user", "text": "hi"},
                                  {"role": "assistant", "text": "yo"}], 60)


# ---- Extended evaluation scorecard -----------------------------------------

def _probe_rows(instr_reply, conv_reply="hello there friend"):
    good_checks = {"non_empty": True, "no_replacement_char": True,
                   "no_prompt_echo": True, "no_runaway_repeat": True,
                   "printable_ratio": 1.0}
    return [
        {"id": "instr-1", "kind": "instruction", "reference": "red apple cherry",
         "reply": instr_reply, "checks": dict(good_checks)},
        {"id": "greet-1", "kind": "greeting", "reference": "hello there friend",
         "reply": conv_reply, "checks": dict(good_checks)},
    ]


def test_scorecard_promotes_on_wins_holds_on_regression():
    from adamlm import eval_full

    old = _probe_rows("red apple banana")
    new = _probe_rows("red apple cherry")
    card = eval_full.scorecard(old, new)
    assert card["promote"] is True
    assert "promotable" in card["reason"]
    regressed = _probe_rows("completely unrelated zebra")
    card2 = eval_full.scorecard(old, regressed)
    assert card2["promote"] is False
    assert "regressed" in card2["reason"]
    echo = _probe_rows("red apple cherry")
    echo[0]["reply"] = "User: hi echo"
    echo[0]["checks"]["no_prompt_echo"] = False
    card3 = eval_full.scorecard(old, echo)
    assert card3["promote"] is False
    assert "hygiene" in card3["reason"]


# ---- Research session decline path (no training launched) -------------------

def test_research_session_declines_without_hypothesis_and_reports(tmp_path):
    import time as _time

    session = research.new_session(tmp_path, max_tokens=10, max_seconds=60,
                                   max_experiments=1, use_ai=False)
    assert session["status"] == "created"

    class DryCtx(research.Ctx):
        def launch(self, command, cwd, log_path):
            raise AssertionError("must not launch training in this test")

        def sleep(self, seconds):
            pass

    research.start_session(session["session_id"], tmp_path, ctx=DryCtx())
    current = research.read_session(session["session_id"], tmp_path)
    deadline = _time.time() + 30
    while current.get("status") == "running" and _time.time() < deadline:
        _time.sleep(0.1)
        current = research.read_session(session["session_id"], tmp_path)
    assert current["status"] == "stopped"
    assert current["stop_reason"] is not None
    report = tmp_path / "results" / "research" / (session["session_id"] + ".report.md")
    assert report.is_file()
    overview = research.session_overview(current)
    assert overview["session_id"] == session["session_id"]


# ---- Provider fallback, free-only gate, pause-on-exhaustion ----

def test_free_model_gate_rejects_paid_ids(tmp_path):
    from adamlm import research
    with pytest.raises(ValueError, match="free-models-only"):
        research.write_provider_config(tmp_path, {
            "model": "vendor/paid-model",
            "fallback_models": ["vendor/other-paid"],
        })
    with pytest.raises(ValueError, match="free-models-only"):
        research.write_provider_config(tmp_path, {
            "model": "vendor/free-model:free",
            "fallback_models": ["vendor/paid-model"],
        })
    with pytest.raises(ValueError, match="Maximum API cost must stay 0"):
        research.write_provider_config(tmp_path, {
            "model": "vendor/free-model:free",
            "max_api_cost_usd": 1.0,
        })
    # Valid free-only config accepted
    cfg = research.write_provider_config(tmp_path, {
        "model": "vendor/primary:free",
        "fallback_models": ["vendor/backup:free", "vendor/alt:free"],
        "max_api_cost_usd": 0.0,
    })
    assert cfg["model"] == "vendor/primary:free"
    assert cfg["fallback_models"] == ["vendor/backup:free", "vendor/alt:free"]


def test_supervisor_models_order_and_dedup():
    from adamlm import research
    cfg = {"model": "a:free", "fallback_models": ["b:free", "a:free", "c:free"]}
    assert research.supervisor_models(cfg) == ["a:free", "b:free", "c:free"]


def test_classify_http_error_and_quota():
    from adamlm import research
    assert research._classify_http_error(429, "rate limit exceeded") == "rate_limited"
    assert research._classify_http_error(429, "daily free models quota exhausted") == "quota_exhausted"
    assert research._classify_http_error(401, "unauthorized") == "auth"
    assert research._classify_http_error(500, "internal") == "unavailable"


def test_is_free_model_id():
    from adamlm import research
    assert research.is_free_model_id("vendor/model:free")
    assert research.is_free_model_id("vendor/model : free".replace(" ", ""))
    assert not research.is_free_model_id("vendor/model")
    assert not research.is_free_model_id("")


class _FakeResponse:
    """Minimal stand-in for a urlopen context manager."""

    def __init__(self, content):
        self._content = content

    def read(self):
        return json.dumps({"choices": [{"message": {"content": self._content}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, body):
    import io
    import urllib.error
    return urllib.error.HTTPError("https://provider.test/chat/completions", code,
                                  "error", {}, io.BytesIO(body.encode()))


def _free_config(tmp_path):
    from adamlm import research
    research.write_provider_config(tmp_path, {
        "enabled": True,
        "model": "vendor/primary:free",
        "fallback_models": ["vendor/backup:free", "vendor/alt:free"],
    })
    research.store_api_key(tmp_path, "test-key")


def test_fallback_walks_to_next_free_model_on_rate_limit(tmp_path):
    from unittest.mock import patch

    from adamlm import research
    _free_config(tmp_path)
    called = []

    def fake_urlopen(request, timeout=None):
        model = json.loads(request.data.decode())["model"]
        called.append(model)
        if model == "vendor/primary:free":
            raise _http_error(429, "rate limit exceeded")
        return _FakeResponse('{"hypothesis": "h"}')

    with patch("urllib.request.urlopen", fake_urlopen):
        result = research._chat_completion_with_fallback(
            tmp_path, messages=[{"role": "user", "content": "x"}])

    # Order is honoured and the walk stops at the first model that answers.
    assert called == ["vendor/primary:free", "vendor/backup:free"]
    assert result["model"] == "vendor/backup:free"
    assert result["api_requests"] == 2
    assert [a["kind"] for a in result["attempts"]] == ["rate_limited", "ok"]


def test_account_quota_exhaustion_stops_walk_immediately(tmp_path):
    from unittest.mock import patch

    from adamlm import research
    _free_config(tmp_path)
    called = []

    def fake_urlopen(request, timeout=None):
        called.append(json.loads(request.data.decode())["model"])
        raise _http_error(429, "daily free models quota exhausted for this account")

    with patch("urllib.request.urlopen", fake_urlopen):
        with pytest.raises(research.ProviderExhausted) as excinfo:
            research._chat_completion_with_fallback(
                tmp_path, messages=[{"role": "user", "content": "x"}])

    # Account-wide quota is not per-model: burning the rest of the list is pointless.
    assert called == ["vendor/primary:free"]
    assert excinfo.value.kind == "quota_exhausted"


def test_all_free_models_failing_tries_each_then_exhausts(tmp_path):
    from unittest.mock import patch

    from adamlm import research
    _free_config(tmp_path)
    called = []

    def fake_urlopen(request, timeout=None):
        called.append(json.loads(request.data.decode())["model"])
        raise _http_error(503, "service unavailable")

    with patch("urllib.request.urlopen", fake_urlopen):
        with pytest.raises(research.ProviderExhausted) as excinfo:
            research._chat_completion_with_fallback(
                tmp_path, messages=[{"role": "user", "content": "x"}])

    assert called == ["vendor/primary:free", "vendor/backup:free", "vendor/alt:free"]
    assert excinfo.value.kind == "exhausted"
    assert len(excinfo.value.attempts) == 3


def test_hand_edited_paid_model_is_refused_before_any_request(tmp_path):
    """The per-call free gate holds even if the config file is edited directly."""
    from unittest.mock import patch

    from adamlm import research
    research.atomic_json(tmp_path / "config" / "research-provider.json", {
        "enabled": True,
        "base_url": "https://openrouter.ai/api/v1",
        "model": "vendor/sneaky-paid",
        "fallback_models": ["vendor/also-paid"],
    })
    research.store_api_key(tmp_path, "test-key")

    def no_network(*a, **k):
        raise AssertionError("a paid model must never reach the network")

    with patch("urllib.request.urlopen", no_network):
        with pytest.raises(research.ProviderExhausted) as excinfo:
            research._chat_completion_with_fallback(
                tmp_path, messages=[{"role": "user", "content": "x"}])
    assert excinfo.value.kind == "not_configured"


def test_research_pauses_on_exhaustion_without_failing(tmp_path):
    import time as _time
    from adamlm import research
    from unittest.mock import patch

    tmp = tmp_path
    research.write_provider_config(tmp, {
        'enabled': True,
        'model': 'vendor/primary:free',
        'fallback_models': ['vendor/fallback:free'],
    })
    research.store_api_key(tmp, 'test-key')

    session = research.new_session(tmp, max_tokens=100, max_seconds=60,
                                   max_api_requests=1, use_ai=True)
    sid = session["session_id"]
    # An earlier decision already consumed the single allowed request, so the
    # next AI-dependent decision must pause before sending anything.
    session["budget_used"]["api_requests"] = 1
    research.atomic_json(tmp / "results" / "research" / f"{sid}.json", session)

    class DryCtx(research.Ctx):
        def launch(self, *a, **k):
            raise AssertionError("must not reach training in this test")
        def sleep(self, s): pass

    def no_network(*a, **k):
        raise AssertionError("cap reached: no provider request may be sent")

    with patch('urllib.request.urlopen', no_network):
        research.start_session(sid, tmp, ctx=DryCtx())
        current = research.read_session(sid, tmp)
        deadline = _time.time() + 30
        while current.get("status") == "running" and _time.time() < deadline:
            _time.sleep(0.1)
            current = research.read_session(sid, tmp)

    assert current["status"] == "paused-provider"
    assert "API request cap reached" in current.get("stop_reason", "")
    assert current["provider_state"]["paused"] is True
    assert current["provider_state"]["kind"] == "api_cap"
    assert len(current["experiments"]) == 1
    assert current["experiments"][0]["status"] == "paused-provider"
    assert "supervisor_attempts" in current["experiments"][0]


def test_research_resume_after_provider_pause_reconciles(tmp_path):
    import time as _time
    from adamlm import research
    from unittest.mock import patch

    tmp = tmp_path
    research.write_provider_config(tmp, {
        'enabled': True,
        'model': 'vendor/primary:free',
        'fallback_models': ['vendor/fallback:free'],
    })
    research.store_api_key(tmp, 'test-key')

    session = research.new_session(tmp, max_tokens=100, max_seconds=60,
                                   max_api_requests=1, use_ai=True)
    sid = session["session_id"]

    class DryCtx(research.Ctx):
        def launch(self, *a, **k):
            raise AssertionError("must not reach training in this test")
        def sleep(self, s): pass

    class MockResponse:
        def __init__(self, model):
            self.model = model
        def read(self):
            return b'{"choices": [{"message": {"content": "{\\"stop\\": true, \\"reason\\": \\"done\\"}"}}]}'
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def mock_urlopen(request, timeout=None):
        return MockResponse(None)

    with patch('urllib.request.urlopen', mock_urlopen):
        research.start_session(sid, tmp, ctx=DryCtx())
        current = research.read_session(sid, tmp)
        deadline = _time.time() + 30
        while current.get("status") == "running" and _time.time() < deadline:
            _time.sleep(0.1)
            current = research.read_session(sid, tmp)

    # Mark as provider-paused for resume test
    current["status"] = "paused-provider"
    current["provider_state"] = {"paused": True, "reason": "test", "paused_at": _time.time(),
                                 "attempts": [], "last_model": "test:free"}
    research.atomic_json(tmp / "results" / "research" / f"{sid}.json", current)

    # Resume should clear paused and continue
    class MockResponse2:
        def __init__(self, model):
            self.model = model
        def read(self):
            return b'{"choices": [{"message": {"content": "{\\"stop\\": true, \\"reason\\": \\"done\\"}"}}]}'
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def mock_urlopen2(request, timeout=None):
        return MockResponse2(None)

    with patch('urllib.request.urlopen', mock_urlopen2):
        resumed = research.start_session(sid, tmp, ctx=DryCtx())
        current = research.read_session(sid, tmp)
        deadline = _time.time() + 30
        while current.get("status") == "running" and _time.time() < deadline:
            _time.sleep(0.1)
            current = research.read_session(sid, tmp)

    # Should stop with api_cap again but NOT duplicate the first experiment
    assert current["status"] in ("stopped", "paused-provider")
    # Original experiment count preserved
    exp_count = len([e for e in current["experiments"] if e.get("index") == 1])
    assert exp_count <= 1  # Never re-run index 1


# ---- Research session controls (2.0): datasets, stage cap, repetition -----

def _write_mix_manifest(root, dd_tokens=5258299, dolly_tokens=3033000):
    base = Path(root) / "data" / "sft_assistant"
    base.mkdir(parents=True, exist_ok=True)
    (base / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (base / "processed-manifest.json").write_text(json.dumps({
        "format": "adamlm-response-sft-v1", "revision": "test",
        "sources": {"dailydialog": {"rows": 10, "tokens": dd_tokens},
                    "dolly": {"rows": 10, "tokens": dolly_tokens}},
        "files": {"train": {"path": "data/sft_assistant/train.jsonl", "examples": 20}}}),
        encoding="utf-8")


def _session_limits(**over):
    limits = {"max_tokens": 10_000_000, "_tokens_used": 0, "max_seconds": 3600,
              "allowed_ops": ["train", "evaluate", "compare"]}
    limits.update(over)
    return limits


def test_new_session_dataset_controls_and_validation(tmp_path):
    s = research.new_session(tmp_path, max_tokens=1000000, max_seconds=600,
                             allowed_datasets=["dolly", "dailydialog"],
                             stage_token_cap=500000)
    assert s["limits"]["allowed_datasets"] == ["dolly", "dailydialog"]
    assert s["limits"]["stage_token_cap"] == 500000
    assert s["limits"]["allow_repetition"] is False
    assert s["status"] == "created"
    with pytest.raises(ValueError, match="Unsupported dataset"):
        research.new_session(tmp_path, max_tokens=1000, max_seconds=60,
                             allowed_datasets=["shady"])
    with pytest.raises(ValueError, match="at least one dataset"):
        research.new_session(tmp_path, max_tokens=1000, max_seconds=60,
                             allowed_datasets=[])
    with pytest.raises(ValueError, match="Per-stage token cap must be positive"):
        research.new_session(tmp_path, max_tokens=1000, max_seconds=60,
                             stage_token_cap=0)


def test_validate_proposal_enforces_session_controls(tmp_path):
    _, ckpt = _make_run(tmp_path, "auto-assistant-20260917-231947-s3",
                        tokens=8000, step=2024)
    _write_mix_manifest(tmp_path)
    summary = {"approved": {"checkpoint": str(ckpt)}}
    base = {"hypothesis": "h", "parent_checkpoint": str(ckpt), "dataset": "dolly",
            "target_tokens": 500000, "max_seconds": 600, "eval_plan": "probes"}
    ok_limits = _session_limits(allowed_datasets=["dolly", "dailydialog"],
                                stage_token_cap=1000000, allow_repetition=True)
    assert research.validate_proposal(dict(base), summary=summary,
                                      limits=ok_limits, root=tmp_path)["dataset"] == "dolly"
    with pytest.raises(ValueError, match="not selected"):
        research.validate_proposal(dict(base), summary=summary,
                                   limits=_session_limits(allowed_datasets=["dailydialog"]),
                                   root=tmp_path)
    with pytest.raises(ValueError, match="per-stage"):
        research.validate_proposal(dict(base), summary=summary,
                                   limits=_session_limits(stage_token_cap=100000),
                                   root=tmp_path)
    big = dict(base, target_tokens=3033001)  # one row past the dolly pass
    with pytest.raises(ValueError, match="[Rr]epeats"):
        research.validate_proposal(big, summary=summary,
                                   limits=_session_limits(allow_repetition=False),
                                   root=tmp_path)
    assert research.validate_proposal(big, summary=summary,
                                      limits=_session_limits(allow_repetition=True),
                                      root=tmp_path)["target_tokens"] == 3033001


def test_heuristic_honors_allow_list_and_stage_cap():
    summary = {"approved": {"checkpoint": "ckpt"},
               "assistant_head": {"checkpoint": "ckpt"},
               "last_promotion_eval": {"instr_f1_new": 0.9, "conv_f1_new": 0.1}}
    p = research.local_heuristic_proposal(summary, limits=_session_limits(
        allowed_datasets=["dolly", "dailydialog"], stage_token_cap=250000))
    assert p["dataset"] == "dailydialog" and p["target_tokens"] == 250000
    only_dolly = research.local_heuristic_proposal(summary, limits=_session_limits(
        allowed_datasets=["dolly"]))
    assert only_dolly["dataset"] == "dolly" and "not selected" in only_dolly["hypothesis"]
    declined = research.local_heuristic_proposal(summary, limits=_session_limits(
        allowed_datasets=["tinystories"]))
    assert declined.get("stop") is True and "not selected" in declined["reason"]


def test_research_reuses_saved_outcomes_and_blocks_identical_failed_plan(tmp_path):
    _, ckpt = _make_run(tmp_path, "approved-assistant", tokens=8000, step=2024)
    _write_mix_manifest(tmp_path)
    summary = {"approved": {"checkpoint": str(ckpt)},
               "assistant_head": {"checkpoint": str(ckpt)},
               "last_promotion_eval": {"instr_f1_new": 0.9, "conv_f1_new": 0.1}}
    limits = _session_limits(allowed_datasets=["dailydialog"], stage_token_cap=2_000_000)
    proposal = research.local_heuristic_proposal(summary, limits=limits)
    assert proposal["dataset"] == "dailydialog"

    previous = research.new_session(tmp_path, max_tokens=2_000_000, max_seconds=600,
                                    allowed_datasets=["dailydialog"], stage_token_cap=2_000_000)
    previous["experiments"].append({
        "index": 1, "status": "rejected", "plan": proposal,
        "decision": "conversation score regressed",
    })
    research.atomic_json(tmp_path / "results" / "research" / f"{previous['session_id']}.json", previous)

    outcomes = research.research_outcomes(tmp_path)
    assert outcomes[0]["status"] == "rejected"
    assert outcomes[0]["reason"] == "conversation score regressed"
    history_summary = {**summary, "research_outcomes": outcomes}
    stopped = research.local_heuristic_proposal(history_summary, limits=limits)
    assert stopped["stop"] is True and "repeats failed/rejected" in stopped["reason"]
    with pytest.raises(ValueError, match="repeats failed/rejected"):
        research.validate_proposal(proposal, summary=history_summary, limits=limits, root=tmp_path)

    changed = dict(proposal, target_tokens=proposal["target_tokens"] + 100_000)
    assert research.validate_proposal(changed, summary=history_summary,
                                      limits=limits, root=tmp_path)["target_tokens"] == 1_100_000


def test_preview_creates_nothing_and_validates(tmp_path):
    _, ckpt = _make_run(tmp_path, "auto-assistant-20260917-231947-s3",
                        tokens=8000, step=2024)
    _write_mix_manifest(tmp_path)
    (Path(tmp_path) / "results" / "assistant_default.json").write_text(json.dumps(
        {"run": "auto-assistant-20260917-231947-s3", "checkpoint": str(ckpt),
         "eval": {"instr_f1_new": 0.9, "conv_f1_new": 0.1}}), encoding="utf-8")
    before = {p for p in Path(tmp_path).rglob("*")}
    preview = research.preview_research(tmp_path, max_tokens=1000000, max_seconds=600,
                                        allowed_datasets=["dolly", "dailydialog"])
    assert {p for p in Path(tmp_path).rglob("*")} == before  # pure: no writes
    assert preview["starts_nothing"] is True
    assert preview["first_decision"]["dataset"] == "dailydialog"
    assert preview["first_decision_valid"]["dataset"] == "dailydialog"
    assert preview["first_decision_error"] is None
    assert preview["single_pass_tokens"]["dolly"] == 3033000
    # The heuristic stays diagnostic (<=1M) so a huge session cap alone
    # never previews repeated data.
    over = research.preview_research(tmp_path, max_tokens=10_000_000, max_seconds=600,
                                     allowed_datasets=["dolly"],
                                     stage_token_cap=10_000_000)
    assert over["first_decision"]["target_tokens"] <= 1_000_000
    assert over["first_decision_error"] is None


# ---- Version archive (2.0): hide, never delete ------------------------------

def test_archive_hides_without_deleting(tmp_path):
    _, ckpt = _make_run(tmp_path, "sft-smoke-x", tokens=100, step=10)
    reg = versions.write_registry(tmp_path)
    vid = next(v["version_id"] for v in reg["versions"] if v["run"] == "sft-smoke-x")
    out = versions.set_archived(tmp_path, version_id=vid, archived=True)
    assert out == {"version_id": vid, "archived": True}
    assert ckpt.is_file()  # nothing deleted
    reread = versions.read_registry(tmp_path)
    flagged = next(v for v in reread["versions"] if v["version_id"] == vid)
    assert flagged["archived"] is True and vid in reread["archived"]
    back = versions.set_archived(tmp_path, version_id=vid, archived=False)
    assert back["archived"] is False
    assert versions.read_registry(tmp_path)["archived"] == []
    with pytest.raises(ValueError, match="Unknown version"):
        versions.set_archived(tmp_path, version_id="nope@step-00000000")


def test_archive_refuses_protected_versions(tmp_path):
    run_dir, ckpt = _make_run(tmp_path, "auto-assistant-x1", tokens=8000, step=2024)
    (Path(tmp_path) / "results" / "assistant_default.json").write_text(json.dumps(
        {"run": "auto-assistant-x1", "checkpoint": str(ckpt),
         "previous": {"run": "auto-assistant-x0", "checkpoint": str(ckpt)}}),
        encoding="utf-8")
    reg = versions.write_registry(tmp_path)
    approved_vid = next(v["version_id"] for v in reg["versions"]
                        if v["approval"] == "approved")
    with pytest.raises(ValueError, match="[Rr]efusing to archive"):
        versions.set_archived(tmp_path, version_id=approved_vid, archived=True)


# ---- Provider interface (2.0): seam, guards, bounded default -----------------

def test_provider_registry_rejects_unknown_and_reserved():
    from adamlm import providers

    assert providers.get_provider("openrouter").display_name == "OpenRouter"
    assert providers.get_provider(None).provider_id == "openrouter"
    with pytest.raises(ValueError, match="reserved"):
        providers.get_provider("ollama")
    with pytest.raises(ValueError, match="Unknown supervisor provider"):
        providers.get_provider("shady")


def test_provider_free_gate_and_bounded_cap(tmp_path):
    from adamlm import providers

    assert providers.default_provider_config()["max_api_requests"] == 25
    assert providers.is_free_model_id("vendor/model:free")
    assert not providers.is_free_model_id("vendor/model")
    good = providers.write_provider_config(tmp_path, {
        "provider": "openrouter", "model": "vendor/main:free",
        "fallback_models": ["vendor/backup:free"], "max_api_requests": 25,
        "max_api_cost_usd": 0.0})
    assert good["provider"] == "openrouter"
    with pytest.raises(ValueError, match="reserved|Unknown"):
        providers.write_provider_config(tmp_path, {"provider": "ollama"})
    with pytest.raises(ValueError, match="non-free"):
        providers.write_provider_config(tmp_path, {"model": "vendor/paid"})
    with pytest.raises(ValueError, match="must stay 0"):
        providers.write_provider_config(tmp_path, {"max_api_cost_usd": 5})
    with pytest.raises(ValueError, match="[Bb]ounded|positive"):
        providers.write_provider_config(tmp_path, {"max_api_requests": 10 ** 12})
    # Research module keeps working aliases over the same implementation.
    assert research.is_free_model_id("vendor/model:free")
    assert research.supervisor_models({"model": "a:free", "fallback_models": []}) == ["a:free"]
