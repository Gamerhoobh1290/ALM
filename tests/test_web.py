"""Focused dashboard tests: active-run tracking, stop targeting, session metadata.

Uses only temporary directories and held file locks to simulate trainers.
Never touches real results/, checkpoints, or the GPU.
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace

from filelock import FileLock

from adamlm import gui_core, web
from adamlm.gui_core import RunInfo


class _FakeGpu:
    """In-memory GPU-session stand-in: unit tests must never touch results/."""

    def __init__(self):
        self.owner = None

    def current(self, root):
        return self.owner

    def try_acquire(self, root, *, kind, label, run=None):
        if self.owner is not None:
            return None
        self.owner = {"kind": kind, "label": label, "run": run,
                      "started_at": 0.0}
        return self.owner

    def release(self, root, *, kind=None, label=None):
        self.owner = None
        return True

    def describe(self, root):
        if not self.owner:
            return {"busy": False, "kind": None, "label": None, "started_at": None}
        return {"busy": True, "kind": self.owner["kind"],
                "label": self.owner["label"], "started_at": self.owner["started_at"]}

    def read_session(self, root):
        return None

    def force_release_stale(self, root):
        self.owner = None
        return True


import pytest as _pytest


@_pytest.fixture(autouse=True)
def _isolated_gpu_session(monkeypatch):
    fake = _FakeGpu()
    monkeypatch.setattr(web.gpu_session, "current", fake.current)
    monkeypatch.setattr(web.gpu_session, "try_acquire", fake.try_acquire)
    monkeypatch.setattr(web.gpu_session, "release", fake.release)
    monkeypatch.setattr(web.gpu_session, "describe", fake.describe)
    monkeypatch.setattr(web.gpu_session, "read_session", fake.read_session)
    monkeypatch.setattr(web.gpu_session, "force_release_stale", fake.force_release_stale)
    yield fake


def _make_run(tmp_path, name, *, state="running", tokens=1000, target=5000,
              stage="pretrain", dataset="mixture", hold_lock=False, with_checkpoint=True):
    run_dir = tmp_path / name
    (run_dir / "checkpoints").mkdir(parents=True)
    checkpoint = None
    if with_checkpoint:
        checkpoint = run_dir / "checkpoints" / "step_00001000.pt"
        checkpoint.touch()
    (run_dir / "launcher-config.json").write_text(
        json.dumps({"target_tokens": target, "stage": stage, "dataset": dataset}), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"state": state, "tokens": tokens, "target_tokens": target,
                    "stage": stage, "dataset": dataset}), encoding="utf-8")
    lock = FileLock(str(run_dir / ".training.lock")) if hold_lock else None
    if lock is not None:
        lock.acquire()
    active = gui_core.trainer_active(run_dir)
    run = RunInfo(run_dir, {"target_tokens": target, "stage": stage, "dataset": dataset},
                  {"state": state, "tokens": tokens, "target_tokens": target,
                   "stage": stage, "dataset": dataset}, checkpoint, active)
    return run, lock


def test_display_state_distinguishes_running_stopping_and_stale(tmp_path):
    running, lock = _make_run(tmp_path, "live", state="running", hold_lock=True)
    try:
        assert gui_core.display_state(running) == "running"
        (tmp_path / "live" / "STOP_REQUESTED").touch()
        assert gui_core.display_state(running) == "stopping"
    finally:
        lock.release()
    stale, _ = _make_run(tmp_path, "stale", state="running")
    assert gui_core.display_state(stale) == "interrupted"
    stopped, _ = _make_run(tmp_path, "done", state="stopped")
    assert gui_core.display_state(stopped) == "stopped"
    reached, _ = _make_run(tmp_path, "full", state="target_reached")
    assert gui_core.display_state(reached) == "target_reached"


def test_checkpoint_category_labels_smoke_first(tmp_path):
    prod, _ = _make_run(tmp_path, "bpe512-local-run", stage="pretrain", dataset="tinystories")
    assert gui_core.checkpoint_category(prod) == "production"
    general, _ = _make_run(tmp_path, "general-pretrain-run")
    assert gui_core.checkpoint_category(general) == "general-pretraining"
    sft, _ = _make_run(tmp_path, "sft-dolly-run", stage="sft", dataset="dolly")
    assert gui_core.checkpoint_category(sft) == "sft"
    smoke, _ = _make_run(tmp_path, "sft-smoke", stage="sft", dataset="dolly")
    assert gui_core.checkpoint_category(smoke) == "smoke-test"
    bench, _ = _make_run(tmp_path, "web-ui-cycle-20260917-1342")
    assert gui_core.checkpoint_category(bench) == "smoke-test"


def test_session_timing_prefers_session_file_then_metrics(tmp_path):
    run_dir = tmp_path / "sess"
    run_dir.mkdir()
    assert gui_core.session_max_seconds(run_dir) is None
    assert gui_core.session_started_at(run_dir) is None
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"event": "start", "step": 0, "max_session_seconds": 3600.0}) + "\n", encoding="utf-8")
    assert gui_core.session_max_seconds(run_dir) == 3600.0
    assert gui_core.session_started_at(run_dir) is None  # never invented
    gui_core.write_session(run_dir, {"run_name": "sess", "started_at": 1700000000.0, "max_seconds": 60.0})
    assert gui_core.session_max_seconds(run_dir) == 60.0
    assert gui_core.session_started_at(run_dir) == 1700000000.0


def test_choose_stop_target_prefers_active_over_selection(tmp_path):
    parent, _ = _make_run(tmp_path, "parent-run", state="target_reached")
    child, lock = _make_run(tmp_path, "child-stage", state="running", hold_lock=True)
    try:
        runs = [parent, child]
        # The reported bug: selection points at the parent while the child trains.
        target, active, stopped = web._choose_stop_target("parent-run", runs)
        assert not stopped and target.name == "child-stage" and active.name == "child-stage"
        # No selection at all still finds the trainer (backend-restart case).
        target, _, _ = web._choose_stop_target(None, runs)
        assert target.name == "child-stage"
        # Selecting the active run targets it directly.
        target, _, _ = web._choose_stop_target("child-stage", runs)
        assert target.name == "child-stage"
    finally:
        lock.release()
    target, active, stopped = web._choose_stop_target("parent-run", [parent])
    assert stopped and target is None and active is None


def test_stop_training_requests_active_and_reports_idle(tmp_path, monkeypatch):
    parent, _ = _make_run(tmp_path, "parent-run", state="target_reached")
    child, lock = _make_run(tmp_path, "child-stage", state="running", hold_lock=True)
    monkeypatch.setattr(web, "_known_runs", lambda: [parent, child])
    try:
        result = web._stop_training({"run_name": "parent-run"})
        assert result["already_stopped"] is False
        assert result["run_name"] == "child-stage"
        assert result["display_state"] == "stopping"
        assert (tmp_path / "child-stage" / "STOP_REQUESTED").is_file()
        assert not (tmp_path / "parent-run" / "STOP_REQUESTED").exists()
    finally:
        lock.release()
    monkeypatch.setattr(web, "_known_runs", lambda: [parent])
    result = web._stop_training({"run_name": "parent-run"})
    assert result["already_stopped"] is True
    assert "No active trainer" in result["message"]


def test_active_run_rediscovered_from_lock_without_jobs(tmp_path, monkeypatch):
    child, lock = _make_run(tmp_path, "child-stage", state="running", hold_lock=True)
    try:
        with web._jobs_lock:
            web._jobs.clear()
        # Backend restart clears handles, but the lock scan still finds the trainer.
        assert web._active_run([child]).name == "child-stage"
        assert gui_core.active_runs([child])[0].name == "child-stage"
    finally:
        lock.release()


def test_start_training_returns_identity_writes_session_and_blocks_duplicates(tmp_path, monkeypatch):
    plan = SimpleNamespace(mode="continuation", run_dir=tmp_path / "next-stage",
                           dataset="mixture", command=[web.sys.executable, "-x"],
                           checkpoint=None, additional_tokens=1000, cumulative_tokens=0,
                           remaining_tokens=1000, effective_target=1000, updates=1,
                           total_updates=1, learning_rate=0.0003, minimum_learning_rate=0.00003,
                           warmup_steps=50, throughput=None, estimated_seconds=None, params=100)
    monkeypatch.setattr(web, "_plan_from_payload", lambda payload: plan)
    monkeypatch.setattr(web, "_active_run", lambda runs=None: None)
    created = {}

    class FakeProcess:
        pass

    def fake_popen(*args, **kwargs):
        created["command"] = args[0]
        return FakeProcess()

    monkeypatch.setattr(web.subprocess, "Popen", fake_popen)
    before = time.time()
    result = web._start_training({"stage_name": "next-stage"})
    assert result["run_name"] == "next-stage"
    assert result["run_dir"] == str(tmp_path / "next-stage")
    session = json.loads((tmp_path / "next-stage" / "web-session.json").read_text(encoding="utf-8"))
    assert session["run_name"] == "next-stage" and session["started_at"] >= before
    with web._jobs_lock:
        web._jobs.pop("trainer:next-stage", None)

    live = SimpleNamespace(name="other-stage")
    monkeypatch.setattr(web, "_active_run", lambda runs=None: live)
    try:
        web._start_training({"stage_name": "next-stage"})
    except RuntimeError as exc:
        assert "other-stage" in str(exc)
    else:
        raise AssertionError("duplicate training session was not blocked")


def test_resume_rejects_completed_target_without_changing_schedule(tmp_path):
    run_dir = tmp_path / "full"
    checkpoint = run_dir / "checkpoints" / "step_00001000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    run = RunInfo(run_dir, {"target_tokens": 5000, "dataset": "mixture"},
                  {"tokens": 5000, "target_tokens": 5000}, checkpoint, False)
    try:
        gui_core.build_plan(mode="resume", dataset="mixture", additional_tokens=1000,
                            duration_seconds=60, selected_run=run, root=tmp_path)
    except ValueError as exc:
        assert "continuation" in str(exc).lower()
    else:
        raise AssertionError("completed run allowed a resume that would rewrite its schedule")


def test_session_limit_from_command():
    assert web._session_limit_from_command([web.sys.executable, "--run-until-stopped"]) is None
    assert web._session_limit_from_command([web.sys.executable, "--max-seconds", "3600.0"]) == 3600.0
    assert web._session_limit_from_command([web.sys.executable]) is None


def test_playground_seed_random_by_default_locked_when_fixed():
    s1 = web._parse_playground_seed(None)
    s2 = web._parse_playground_seed("random")
    assert 0 <= s1 <= 4294967295 and 0 <= s2 <= 4294967295
    assert web._parse_playground_seed("42") == 42
    assert web._parse_playground_seed(123) == 123
    for bad in ("abc", "-1", "4294967296", "3.5"):
        try:
            web._parse_playground_seed(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad seed {bad!r} was accepted")


def test_playground_generation_passes_seed_without_touching_train_eval_seeds(tmp_path, monkeypatch):
    # Training/eval reproducibility stays fixed: CLI default 42, eval SEED 42.
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    gen_src = (root / "scripts" / "generate.py").read_text(encoding="utf-8")
    assert "--seed" in gen_src and "default=42" in gen_src
    eval_src = (root / "scripts" / "eval_assistant.py").read_text(encoding="utf-8")
    assert "TOKENS, TEMP, TOPK, SEED = 60, 0.8, 40, 42" in eval_src

    run_dir = tmp_path / "sft-assistant"
    ckpt = run_dir / "checkpoints" / "step_00000100.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    (run_dir / "launcher-config.json").write_text(json.dumps({"stage": "sft"}), encoding="utf-8")
    monkeypatch.setattr(web, "_resolve_checkpoint", lambda value: ckpt)
    monkeypatch.setattr(web, "_active_run", lambda runs=None: None)
    seen = {}

    class FakeProc:
        pass

    def fake_popen(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(web.subprocess, "Popen", fake_popen)
    res = web._start_generation({"checkpoint": str(ckpt), "prompt": "hi",
                                 "temperature": 0.8, "top_k": 40, "tokens": 16, "seed": "7"})
    assert res["seed"] == 7
    assert "--seed" in seen["cmd"] and "7" in seen["cmd"]
    res2 = web._start_generation({"checkpoint": str(ckpt), "prompt": "hi",
                                  "temperature": 0.8, "top_k": 40, "tokens": 16})
    assert 0 <= res2["seed"] <= 4294967295
    assert "--seed" in seen["cmd"]
    with web._jobs_lock:
        web._jobs.pop(res["job_id"], None)
        web._jobs.pop(res2["job_id"], None)


def test_start_generation_refuses_while_trainer_active(tmp_path, monkeypatch):
    import pytest as _pytest
    from types import SimpleNamespace
    run_dir = tmp_path / "sft-assistant"
    ckpt = run_dir / "checkpoints" / "step_00000100.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    (run_dir / "launcher-config.json").write_text(json.dumps({"stage": "sft"}), encoding="utf-8")
    monkeypatch.setattr(web, "_resolve_checkpoint", lambda value: ckpt)
    monkeypatch.setattr(web, "_active_run", lambda runs=None: SimpleNamespace(name="busy-stage"))
    with _pytest.raises(RuntimeError, match="busy-stage"):
        web._start_generation({"checkpoint": str(ckpt), "prompt": "hi"})


def test_cancel_generation_marks_cancelled(tmp_path):
    import threading
    import time as _time

    class FakeProc:
        def __init__(self):
            self.killed = False

        def kill(self):
            self.killed = True

    proc = FakeProc()
    with web._jobs_lock:
        web._jobs["job-test-cancel"] = {"state": "running", "output": "", "process": proc}
    try:
        result = web._cancel_generation({"job_id": "job-test-cancel"})
        assert result["already_done"] is False
        assert proc.killed is True
        with web._jobs_lock:
            assert web._jobs["job-test-cancel"]["state"] == "cancelled"
        again = web._cancel_generation({"job_id": "job-test-cancel"})
        assert again["already_done"] is True
        unknown = web._cancel_generation({"job_id": "job-test-missing"})
        assert unknown["already_done"] is True
    finally:
        with web._jobs_lock:
            web._jobs.pop("job-test-cancel", None)


def test_streaming_worker_assembles_incremental_output(tmp_path, monkeypatch):
    import time as _time
    run_dir = tmp_path / "sft-assistant"
    ckpt = run_dir / "checkpoints" / "step_00000100.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    (run_dir / "launcher-config.json").write_text(json.dumps({"stage": "sft"}), encoding="utf-8")
    monkeypatch.setattr(web, "_resolve_checkpoint", lambda value: ckpt)
    monkeypatch.setattr(web, "_active_run", lambda runs=None: None)

    class FakeStdout(list):
        def close(self):
            pass

    class FakeStderr:
        def read(self):
            return ""

    class FakeProc:
        returncode = 0

        def __init__(self, lines):
            self.stdout = FakeStdout(lines)
            self.stderr = FakeStderr()

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    lines = ['{"text": "hel"}\n', 'not-json\n', '{"text": "hello there"}\n']
    monkeypatch.setattr(web.subprocess, "Popen", lambda *a, **k: FakeProc(lines))
    res = web._start_generation({"checkpoint": str(ckpt), "prompt": "hi", "tokens": 16, "seed": "3"})
    deadline = _time.time() + 10
    while _time.time() < deadline:
        with web._jobs_lock:
            state = web._jobs.get(res["job_id"], {}).get("state")
            output = web._jobs.get(res["job_id"], {}).get("output")
        if state == "done":
            break
        _time.sleep(0.05)
    try:
        assert state == "done"
        assert output == "hello there"
    finally:
        with web._jobs_lock:
            web._jobs.pop(res["job_id"], None)


def test_plan_overview_carries_eval_scorecard(tmp_path):
    from adamlm import auto_train
    plan = {"plan_id": "auto-assistant-20990101-000000", "status": "done",
            "created_at": 1.0, "started_at": 1.0, "updated_at": 2.0,
            "requested_tokens": 2000, "effective_tokens": 2000,
            "head": {"run": "sft-assistant", "checkpoint": "c.pt", "step": 1, "tokens": 1},
            "time": {"max_seconds": 60}, "stages": [],
            "warnings": [], "stop_reason": None, "failure": None,
            "produced_checkpoint": "n.pt",
            "promotion": {"promoted": True, "reason": "promoted: instructions F1 0.100->0.200"},
            "eval": {"instr_f1_old": 0.1, "instr_f1_new": 0.2, "conv_f1_old": 0.3,
                     "conv_f1_new": 0.35, "hygiene_problems": ["reply 0: empty"] * 10}}
    auto_train.write_plan(plan, tmp_path)
    overview = auto_train.plan_overview(tmp_path)
    assert overview["eval"]["instr_f1_new"] == 0.2
    assert overview["eval"]["hygiene_problem_count"] == 10
    assert len(overview["eval"]["hygiene_problems"]) <= 8


def test_assistant_overview_uses_lineage_not_step_alone(tmp_path, monkeypatch):
    from adamlm import auto_train
    # Build three runs: head has most tokens but restarted step number;
    # unrelated experiment has the highest step yet must be excluded.
    def _mk(name, *, tokens, step, stage="sft"):
        d = tmp_path / name
        (d / "checkpoints").mkdir(parents=True)
        cp = d / "checkpoints" / f"step_{step:08d}.pt"
        cp.touch()
        launcher = {"stage": stage, "dataset": "dailydialog"}
        status = {"state": "target_reached", "tokens": tokens, "target_tokens": tokens + 100,
                  "stage": stage, "dataset": "dailydialog", "step": step, "checkpoint_step": step}
        return RunInfo(d, launcher, status, cp, False)
    runs = [
        _mk("sft-assistant", tokens=1003520, step=245),
        _mk("auto-assistant-20260917-215605-s1", tokens=8290304, step=2024),
        _mk("sft-dailydialog-run", tokens=99999999, step=99999),  # unrelated prefix: excluded
        _mk("sft-smoke", tokens=99999999, step=99999),  # smoke: excluded
    ]
    monkeypatch.setattr(auto_train, "ROOT", tmp_path)
    monkeypatch.setattr(web, "ROOT", tmp_path)
    overview = web._assistant_overview(runs)
    assert overview["head"]["run"] == "auto-assistant-20260917-215605-s1"
    assert overview["head"]["tokens"] == 8290304
    assert {e["run"] for e in overview["lineage"]} == {"sft-assistant", "auto-assistant-20260917-215605-s1"}
    # Promotion status preserved: no default file in tmp => default None (legacy fallback).
    assert overview["default"] is None
