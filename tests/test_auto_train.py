"""Focused Auto Train tests: lineage, allocation, preflight, supervise, promote.

Uses only temporary directories and fake executors. Never touches real
results/, launches no trainers, uses no GPU.
"""
import json
from pathlib import Path

import pytest
import torch

from adamlm import auto_train, sft_data
from adamlm.auto_train import Supervisor
from adamlm.gui_core import RunInfo


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _write_config(root, budget_gb=25):
    config = {
        "seed": 1, "stage": "sft", "dataset": "dailydialog",
        "model": {"vocab_size": 64, "block_size": 8, "n_layer": 1, "n_head": 2, "n_embd": 8},
        "tokenizer": "tokenizers/bpe4096/tokenizer.json",
        "validation": "data/sft_assistant/processed-manifest.json",
        "run_dir": "results/x", "precision": "bf16",
        "batch_size": 2, "accumulation": 1, "target_tokens": 1000,
        "learning_rate": 0.0001, "minimum_learning_rate": 0.00001, "warmup_steps": 2,
        "checkpoint_every": 10, "validation_every": 10, "sample_every": 10,
        "max_session_seconds": 3600,
        "data": {"manifest": "data/sft_assistant/processed-manifest.json", "page_size": 4},
        "storage": {"project_budget_gb": budget_gb, "cache_limit_gb": 2,
                    "minimum_free_disk_gb": 5, "checkpoint_keep": 3, "disk_poll_seconds": 30},
    }
    path = Path(root) / "config" / "sft-assistant.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _write_manifest(root, dd_tokens=8000, dolly_tokens=2000):
    manifest = {
        "format": "adamlm-response-sft-v1", "dataset": "dailydialog+dolly-assistant-v1",
        "revision": "test", "sources": {
            "dailydialog": {"manifest": "data/dailydialog/processed-manifest.json",
                            "rows": 80, "tokens": dd_tokens, "dropped_truncated": 0},
            "dolly": {"manifest": "data/dolly/processed-manifest.json",
                      "rows": 20, "tokens": dolly_tokens, "dropped_truncated": 0},
        },
        "files": {"train": {"path": "data/sft_assistant/train.jsonl", "examples": 100}},
    }
    base = Path(root) / "data" / "sft_assistant"
    base.mkdir(parents=True, exist_ok=True)
    (base / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (base / "processed-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_run(root, name, *, tokens=1000, stage="sft", state="target_reached",
              checkpoint=True, category_stage=None, cursor=None):
    run_dir = Path(root) / "results" / name
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = None
    if checkpoint:
        ckpt = run_dir / "checkpoints" / "step_00000100.pt"
        # A real (tiny) checkpoint: the planner reads the corpus position
        # from this file, so a placeholder would not exercise that path.
        torch.save({"step": 100, "dataset": cursor if cursor is not None else {}}, ckpt)
    (run_dir / "launcher-config.json").write_text(json.dumps(
        {"target_tokens": tokens + 500, "stage": stage, "dataset": "dailydialog"}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps(
        {"state": state, "tokens": tokens, "target_tokens": tokens + 500,
         "stage": category_stage or stage, "dataset": "dailydialog"}), encoding="utf-8")
    launcher = {"target_tokens": tokens + 500, "stage": stage, "dataset": "dailydialog"}
    status = {"state": state, "tokens": tokens, "target_tokens": tokens + 500,
              "stage": category_stage or stage, "dataset": "dailydialog"}
    return RunInfo(run_dir, launcher, status, ckpt, False)


def _standard_runs(root):
    _write_config(root)
    _write_manifest(root)
    foundation = _make_run(root, "general-training-500m", tokens=500_000_000,
                           stage="pretrain", state="target_reached")
    foundation.launcher["stage"] = "pretrain"
    foundation.status["stage"] = "pretrain"
    head = _make_run(root, "sft-assistant", tokens=1_000_000)
    return [foundation, head,
            _make_run(root, "sft-dailydialog-run", tokens=2_000_000),  # not assistant lineage
            _make_run(root, "sft-smoke", tokens=9_999_999),  # newest but smoke
            _make_run(root, "bpe512-local-run", tokens=1, stage="pretrain",
                      state="target_reached")]


class FakeHandle:
    def __init__(self, ctx):
        self.ctx = ctx
        self.exited = False
        self.returncode = None
        self._post_stop_polls = 0

    def poll(self):
        if self.exited:
            return self.returncode
        # A graceful stop takes effect shortly after being requested.
        if self.ctx.stop_exits and self.ctx.stops:
            self._post_stop_polls += 1
            if self._post_stop_polls >= self.ctx.polls_after_stop:
                self.exited = True
                self.returncode = 0
                return 0
        return None


class FakeCtx:
    """Deterministic stand-in for the backend wiring (no processes)."""

    def __init__(self, *, live=None, exit_state="target_reached", checkpoint_name="step_00000100.pt",
                 stop_exits=True, eval_rows=None):
        self.live = live
        self.exit_state = exit_state
        self.checkpoint_name = checkpoint_name
        self.stop_exits = stop_exits
        self.eval_rows = eval_rows
        self.launched = []
        self.stops = []
        self.polls_after_stop = 1
        self.time_jump = 5.0
        self._now = 1000.0
        self.handles = []

    def now(self):
        return self._now

    def sleep(self, seconds):
        self._now += self.time_jump

    def active_trainer(self):
        return self.live

    def launch(self, command, cwd, log_path):
        handle = FakeHandle(self)
        self.handles.append(handle)
        self.launched.append([str(c) for c in command])
        if self.exit_state == "instant":
            handle.exited = True
            handle.returncode = 0
        return handle

    def read_stage_status(self, run_dir):
        # exit_state doubles as the reported end state (handles model liveness
        # only); "instant" means an immediate clean target_reached.
        state = "target_reached" if self.exit_state == "instant" else self.exit_state
        return {"state": state, "tokens": 123,
                "detail": None if state != "failed-x" else "boom"}

    def latest_checkpoint(self, run_dir):
        path = Path(run_dir) / "checkpoints" / self.checkpoint_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ckpt")
        return path

    def request_stop(self, run_dir):
        self.stops.append(str(run_dir))

    def run_eval(self, old_checkpoint, new_checkpoint, out_path, timeout=1800):
        Path(out_path).write_text(json.dumps(self.eval_rows), encoding="utf-8")
        return self.eval_rows["old"], self.eval_rows["new"]


def _eval_rows(reply_new, reply_old="I know .", ref="red apple green"):
    def row(reply):
        return {"id": "instr-heldout-x", "kind": "instruction", "source": "held-out",
                "instruction": "Name something", "context": "", "reference": ref,
                "reply": reply,
                "checks": {"non_empty": True, "no_replacement_char": True,
                           "printable_ratio": 1.0, "no_prompt_echo": True,
                           "no_runaway_repeat": True}}
    return {"old": [row(reply_old)], "new": [row(reply_new)]}


def _plan_dict(root, stages=1, status="running", max_seconds=600):
    run_dirs = []
    stages_list = []
    for i in range(1, stages + 1):
        run_dir = f"results/auto-assistant-20990101-000000-s{i}"
        run_dirs.append(run_dir)
        (Path(root) / run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        stages_list.append({"index": i, "run_dir": run_dir, "target_tokens": 1000,
                            "epoch_label": "pass 1", "status": "pending",
                            "parent_checkpoint": str(Path(root) / "results" / "sft-assistant"
                                                     / "checkpoints" / "step_00000100.pt")})
    return {"plan_id": "auto-assistant-20990101-000000", "status": status,
            "created_at": 1.0, "started_at": None, "updated_at": 1.0,
            "requested_tokens": 1000 * stages, "effective_tokens": 1000 * stages,
            "head": {"run": "sft-assistant",
                     "checkpoint": str(Path(root) / "results" / "sft-assistant"
                                       / "checkpoints" / "step_00000100.pt"),
                     "step": 100, "tokens": 1000},
            "time": {"max_seconds": max_seconds}, "stages": stages_list,
            "warnings": [], "stop_reason": None, "failure": None,
            "produced_checkpoint": None, "promotion": None, "eval": None}


# --------------------------------------------------------------------------
# lineage + foundation
# --------------------------------------------------------------------------

def test_head_is_most_trained_lineage_not_newest_or_smoke(tmp_path):
    runs = _standard_runs(tmp_path)
    head = auto_train.find_assistant_head(runs, tmp_path)
    assert head.name == "sft-assistant"  # not sft-smoke (more tokens) nor production
    parents = [r.name for r in auto_train.assistant_runs(runs, tmp_path)]
    assert parents == ["sft-assistant"]


def test_head_requires_checkpoint_file(tmp_path):
    runs = _standard_runs(tmp_path)
    for run in runs:
        if run.name == "sft-assistant" and run.checkpoint:
            run.checkpoint.unlink()
    with pytest.raises(ValueError, match="[Ll]ineage"):
        auto_train.find_assistant_head(runs, tmp_path)


def test_foundation_must_be_completed(tmp_path):
    runs = _standard_runs(tmp_path)
    assert auto_train.check_foundation(runs, tmp_path).name == "general-training-500m"
    for run in runs:
        if run.name == "general-training-500m":
            run.status["state"] = "stopped"
    with pytest.raises(ValueError, match="[Ff]oundation"):
        auto_train.check_foundation(runs, tmp_path)


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------

def test_allocation_single_stage_seventy_thirty():
    alloc = auto_train.allocate(1_000_000, 8_000_000)
    assert alloc["effective_tokens"] == 1_000_000 and not alloc["capped"]
    assert alloc["dd_share"] == pytest.approx(0.70)
    assert len(alloc["stage_chunks"]) == 1
    assert "70%" in alloc["basis"] and "response" in alloc["basis"]


def test_allocation_multi_stage_labels_repetition_explicitly():
    alloc = auto_train.allocate(20_000_000, 8_000_000, pass_policy="repeat")
    assert alloc["effective_tokens"] == 20_000_000
    assert len(alloc["stage_chunks"]) == 3
    assert any("repeat" in chunk["epoch_label"] for chunk in alloc["stage_chunks"])
    assert sum(c["tokens"] for c in alloc["stage_chunks"]) == 20_000_000


def test_allocation_has_no_arbitrary_pass_ceiling_and_rejects_implicit_repeat():
    with pytest.raises(ValueError, match="repeat policy"):
        auto_train.allocate(99_000_000, 8_000_000)
    alloc = auto_train.allocate(99_000_000, 8_000_000, pass_policy="repeat")
    assert alloc["effective_tokens"] == 99_000_000
    assert len(alloc["stage_chunks"]) == 13
    assert alloc["repeated_data"] and alloc["unique_tokens_upper_bound"] <= 8_000_000


def test_allocation_rounds_full_pass_to_batchable_capacity_without_tail_debt():
    alloc = auto_train.allocate(8_291_299 * 5, 8_291_299, tokens_per_step=4096,
                                pass_policy="repeat")
    assert alloc["batchable_epoch_tokens"] == 8_290_304
    assert alloc["discarded_tail_tokens"] == 995
    assert [c["tokens"] for c in alloc["stage_chunks"]] == [8_290_304] * 5
    assert alloc["effective_tokens"] == 8_290_304 * 5


def test_allocation_tilts_to_measured_weak_skill_with_clamps():
    weak_instr = auto_train.allocate(1_000_000, 8_000_000, {"instr_f1": 0.1, "conv_f1": 0.5})
    assert weak_instr["dd_share"] == pytest.approx(0.60) and "tilted" in weak_instr["basis"]
    weak_conv = auto_train.allocate(1_000_000, 8_000_000, {"instr_f1": 0.5, "conv_f1": 0.1})
    assert weak_conv["dd_share"] == pytest.approx(0.75)
    tie = auto_train.allocate(1_000_000, 8_000_000, {"instr_f1": 0.3, "conv_f1": 0.31})
    assert tie["dd_share"] == pytest.approx(0.70) and "no tilt" in tie["basis"]
    with pytest.raises(ValueError):
        auto_train.allocate(0, 8_000_000)


def test_allocation_schedules_only_the_data_the_cursor_has_not_used():
    """The planner starts where the lineage stands, not at the top of the file."""
    fresh = auto_train.allocate(4_000_000, 8_291_299, tokens_per_step=4096)
    used = auto_train.allocate(4_000_000, 8_291_299, tokens_per_step=4096,
                               consumed_tokens=4_096_000, epoch_index=0)
    assert fresh["remaining_epoch_tokens"] == 8_290_304
    assert used["remaining_epoch_tokens"] == 8_290_304 - 4_096_000
    # Same request, but only the unused remainder is offered as fresh data.
    assert used["stage_chunks"][0]["repeated_data"] is False
    assert sum(c["tokens"] for c in used["stage_chunks"] if not c["repeated_data"]) \
        <= used["remaining_epoch_tokens"]


def test_allocation_refuses_to_pass_the_end_of_the_data_without_a_repeat():
    with pytest.raises(ValueError, match="unused tokens left in pass"):
        auto_train.allocate(5_000_000, 8_291_299, tokens_per_step=4096,
                            consumed_tokens=8_000_000, pass_policy="unique_once")
    # The same request is allowed once repeats are chosen explicitly, and the
    # tokens that come from a repeat are labeled and excluded from unique.
    alloc = auto_train.allocate(5_000_000, 8_291_299, tokens_per_step=4096,
                                consumed_tokens=8_000_000, pass_policy="repeat")
    assert alloc["repeated_data"]
    assert alloc["unique_tokens_upper_bound"] == alloc["remaining_epoch_tokens"]
    assert any("repeated data" in c["epoch_label"] for c in alloc["stage_chunks"])


def test_allocation_continues_lineage_pass_numbering():
    """Pass numbers continue the lineage; they never restart at 1 each plan."""
    alloc = auto_train.allocate(12_000_000, 8_291_299, tokens_per_step=4096,
                                consumed_tokens=4_000_000, epoch_index=3,
                                pass_policy="repeat")
    assert [c["pass_number"] for c in alloc["stage_chunks"]] == [4, 5]
    assert alloc["stage_chunks"][0]["repeated_data"] is False
    assert alloc["stage_chunks"][1]["repeated_data"] is True


def test_allocation_treats_an_exhausted_pass_as_a_repeat_from_the_first_stage():
    alloc = auto_train.allocate(1_000_000, 8_291_299, tokens_per_step=4096,
                                consumed_tokens=8_290_304, epoch_index=2,
                                pass_policy="repeat")
    assert alloc["first_stage_repeats"] and alloc["remaining_epoch_tokens"] == 0
    assert alloc["stage_chunks"][0]["pass_number"] == 4
    assert alloc["stage_chunks"][0]["repeated_data"] is True
    assert alloc["unique_tokens_upper_bound"] == 0


def test_planner_and_trainer_size_a_pass_with_the_same_arithmetic():
    """Preflight's capacity figures are the trainer's, not a parallel estimate."""
    shared = sft_data.epoch_capacity(8_291_299, 4096, consumed=1_234_567 // 4096 * 4096)
    alloc = auto_train.allocate(1_000_000, 8_291_299, tokens_per_step=4096,
                                consumed_tokens=shared["consumed_tokens"])
    assert alloc["batchable_epoch_tokens"] == shared["batchable_tokens"]
    assert alloc["discarded_tail_tokens"] == shared["discarded_tail_tokens"]
    assert alloc["remaining_epoch_tokens"] == shared["remaining_tokens"]


def test_plan_ids_are_unique_and_readable(tmp_path):
    (Path(tmp_path) / "results").mkdir()
    first = auto_train.plan_id_for(tmp_path, now=0)
    assert first.startswith("auto-assistant-")
    (Path(tmp_path) / "results" / first).mkdir()
    (Path(tmp_path) / "results" / f"{first}.auto.json").write_text("{}", encoding="utf-8")
    second = auto_train.plan_id_for(tmp_path, now=0)
    assert second != first and second.startswith(first)


# --------------------------------------------------------------------------
# preflight (integration over tmp fixtures, incl. storage gate)
# --------------------------------------------------------------------------

def test_preflight_happy_path_uses_real_planner(tmp_path):
    runs = _standard_runs(tmp_path)
    pf = auto_train.preflight(5_000, 600, tmp_path, runs)  # fixture epoch is 10k tokens
    assert pf["head"]["run"] == "sft-assistant"
    assert pf["foundation"]["run"] == "general-training-500m"
    assert pf["effective_tokens"] == 5_008 and len(pf["stages"]) == 1
    assert pf["stages"][0]["run_dir"].startswith("results/auto-assistant-")
    assert pf["diet"]["dailydialog"]["tokens"] == 8000
    assert pf["storage"]["fits_budget"] and pf["storage"]["floor_ok"]
    assert pf["storage"]["config"] == auto_train.MIX_CONFIG


def test_preflight_reads_the_head_cursor_and_plans_from_there(tmp_path):
    """End to end: a head that already used most of its pass gets the rest."""
    _write_config(tmp_path)
    _write_manifest(tmp_path)  # 10_000 source tokens, 8-token batches
    foundation = _make_run(tmp_path, "general-training-500m", tokens=500_000_000,
                           stage="pretrain", state="target_reached")
    head = _make_run(tmp_path, "sft-assistant", tokens=1_000_000,
                     cursor={"sft_source": {"dataset": "mix"}, "byte_offset": 900,
                             "epoch": 2, "epoch_tokens": 8_192})
    runs = [foundation, head]
    position = auto_train.head_cursor(head, tmp_path)
    assert position == {"epoch": 2, "epoch_tokens": 8_192, "byte_offset": 900, "tracked": True}

    pf = auto_train.preflight(1_000, 600, tmp_path, runs)
    assert pf["allocation"]["epoch_index"] == 2
    assert pf["allocation"]["consumed_tokens"] == 8_192
    assert pf["allocation"]["remaining_epoch_tokens"] == 10_000 - 8_192
    assert pf["stages"][0]["pass_number"] == 3
    assert pf["stages"][0]["repeat_epoch"] is False
    # Asking for more than the pass has left is refused, not silently replayed.
    with pytest.raises(ValueError, match="unused tokens left in pass"):
        auto_train.preflight(5_000, 600, tmp_path, runs)


def test_preflight_refuses_a_head_whose_cursor_cannot_be_read(tmp_path):
    runs = _standard_runs(tmp_path)
    head = next(run for run in runs if run.name == "sft-assistant")
    head.checkpoint.write_bytes(b"not-a-checkpoint")
    with pytest.raises(ValueError, match="Cannot read the dataset cursor"):
        auto_train.preflight(1_000, 600, tmp_path, runs)


def test_repeat_stages_carry_the_approval_into_the_launch_command(tmp_path):
    """The trainer never rewinds on its own, so the plan must tell it to."""
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path)
    plan["stages"][0]["repeat_epoch"] = True
    ctx = FakeCtx()
    plan["started_at"] = ctx.now()
    auto_train.write_plan(plan, tmp_path)
    supervisor = Supervisor(plan["plan_id"], tmp_path, ctx)
    command = supervisor.stage_command(plan, plan["stages"][0])
    assert "--repeat-epoch" in command

    plan["stages"][0]["repeat_epoch"] = False
    assert "--repeat-epoch" not in supervisor.stage_command(plan, plan["stages"][0])


def test_preflight_refuses_when_storage_would_pause(tmp_path):
    runs = _standard_runs(tmp_path)
    _write_config(tmp_path, budget_gb=0.000001)
    with pytest.raises(ValueError, match="[Bb]udget"):
        auto_train.preflight(500_000, 600, tmp_path, runs, pass_policy="repeat")


def test_estimate_storage_flags_floor(tmp_path):
    _write_config(tmp_path)
    est = auto_train.estimate_storage(tmp_path, stages=1, checkpoint_bytes=10**12)
    assert not est["fits_budget"]


def test_storage_bump_config_is_resume_compatible():
    from adamlm.bpe_train import protocol_config
    base = {"run_dir": "x", "storage": {"project_budget_gb": 15}, "seed": 1}
    bumped = {"run_dir": "x", "storage": {"project_budget_gb": 25}, "seed": 1}
    assert protocol_config(base) == protocol_config(bumped)


# --------------------------------------------------------------------------
# supervisor: start / stop / resume / failure / duplicate
# --------------------------------------------------------------------------

def test_supervisor_completes_stage_and_holds_eval_win_for_manual_promotion(tmp_path):
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path)
    auto_train.write_plan(plan, tmp_path)
    rows = _eval_rows("red apple green banana")
    ctx = FakeCtx(exit_state="instant", eval_rows=rows)
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    done = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert done["status"] == "done"
    assert done["stages"][0]["status"] == "done"
    assert done["promotion"]["promoted"] is False
    assert done["promotion"]["status"] == "hold"
    assert not (Path(tmp_path) / "results" / "assistant_default.json").exists()
    assert ctx.launched and "--max-seconds" in ctx.launched[0]


def test_supervisor_holds_default_on_eval_regression(tmp_path):
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path)
    auto_train.write_plan(plan, tmp_path)
    rows = _eval_rows("I know .")  # worse than old's "I know ." tie? old tie -> promote; make old better
    rows["old"] = [{"id": "instr-heldout-x", "kind": "instruction", "source": "held-out",
                    "instruction": "Name something", "context": "",
                    "reference": "red apple green", "reply": "red apple green",
                    "checks": {"non_empty": True, "no_replacement_char": True,
                               "printable_ratio": 1.0, "no_prompt_echo": True,
                               "no_runaway_repeat": True}}]
    ctx = FakeCtx(exit_state="instant", eval_rows=rows)
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    done = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert done["status"] == "done" and done["promotion"]["promoted"] is False
    assert not (Path(tmp_path) / "results" / "assistant_default.json").exists()


def test_supervisor_stop_then_resume_continues_without_replanning(tmp_path):
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path, stages=2)
    auto_train.write_plan(plan, tmp_path)
    ctx = FakeCtx(exit_state="stopped")  # handle lives until stop requested
    launch_seen = []

    original_launch = ctx.launch

    def launch_and_stop(command, cwd, log_path):
        handle = original_launch(command, cwd, log_path)
        launch_seen.append(True)
        if len(launch_seen) == 1:
            auto_train.request_plan_stop(plan["plan_id"], tmp_path, ctx)
        return handle

    ctx.launch = launch_and_stop
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    stopped = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert stopped["status"] == "stopped"
    assert stopped["stages"][0]["status"] == "stopped"
    assert stopped["stages"][1]["status"] == "pending"
    assert ctx.stops  # graceful stop was requested on the stage dir

    # Resume picks up the same plan: stage 1 reruns to done, stage 2 runs.
    ctx2 = FakeCtx(exit_state="instant",
                   eval_rows=_eval_rows("red apple green banana"))
    resumed = auto_train.read_plan(plan["plan_id"], tmp_path)
    resumed["status"] = "running"
    resumed["started_at"] = ctx2.now()
    auto_train.touch(resumed, tmp_path)
    Supervisor(plan["plan_id"], tmp_path, ctx2).run()
    done = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert done["status"] == "done"
    assert [s["status"] for s in done["stages"]] == ["done", "done"]
    assert done["stages"][0]["run_dir"] == stopped["stages"][0]["run_dir"]  # same dirs, no replan


def test_supervisor_refuses_duplicate_trainer(tmp_path):
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path)
    auto_train.write_plan(plan, tmp_path)
    live = object()
    Supervisor(plan["plan_id"], tmp_path, FakeCtx(live=live)).run()
    failed = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert failed["status"] == "failed" and "already active" in failed["failure"]


def test_supervisor_marks_crash_failed_and_time_limit(tmp_path):
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path)
    auto_train.write_plan(plan, tmp_path)
    ctx = FakeCtx(exit_state="instant")
    ctx.read_stage_status = lambda run_dir: {"state": "mystery", "detail": "gone"}
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    failed = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert failed["status"] == "failed" and "mystery" in failed["failure"]

    plan2 = _plan_dict(tmp_path, max_seconds=600)
    plan2["plan_id"] = "auto-assistant-20990102-000000"
    auto_train.write_plan(plan2, tmp_path)
    ctx2 = FakeCtx(exit_state="session_limit")  # trainer's own timer fires too
    ctx2.time_jump = 10_000.0  # deadline passes mid-stage
    Supervisor(plan2["plan_id"], tmp_path, ctx2).run()
    timed = auto_train.read_plan(plan2["plan_id"], tmp_path)
    assert timed["status"] == "time-exceeded"
    assert timed["stages"][0]["status"] == "time-exceeded"
    assert ctx2.stops  # asked gracefully, never killed


# --------------------------------------------------------------------------
# promotion gate unit tests
# --------------------------------------------------------------------------

def _row(kind, reply, ref="red apple green", source="held-out"):
    return {"id": "x", "kind": kind, "source": source, "instruction": "q",
            "context": "", "reference": ref, "reply": reply,
            "checks": {"non_empty": bool(reply.strip()), "no_replacement_char": True,
                       "printable_ratio": 1.0, "no_prompt_echo": "User:" not in reply,
                       "no_runaway_repeat": True}}


def test_compare_promotes_on_ties_and_wins():
    old = [_row("instruction", "red apple"), _row("greeting", "hello there red")]
    new = [_row("instruction", "red apple"), _row("greeting", "hello there red")]
    verdict = auto_train.compare_evals(old, new)
    assert verdict["promote"] is True and "promoted" in verdict["reason"]
    better = [_row("instruction", "red apple green"), _row("greeting", "hello there red")]
    verdict = auto_train.compare_evals(old, better)
    assert verdict["promote"] is True


def test_compare_holds_on_regression_or_hygiene():
    old = [_row("instruction", "red apple green"), _row("greeting", "hello red apple")]
    worse_instr = [_row("instruction", "I know ."), _row("greeting", "hello red apple")]
    assert auto_train.compare_evals(old, worse_instr)["promote"] is False
    worse_conv = [_row("instruction", "red apple green"), _row("greeting", "zzz qqq")]
    assert auto_train.compare_evals(old, worse_conv)["promote"] is False
    echo = [_row("instruction", "red apple green"), _row("greeting", "User: hello")]
    verdict = auto_train.compare_evals(old, echo)
    assert verdict["promote"] is False and "hygiene" in verdict["reason"]


def test_compare_uses_manual_suite_criteria_and_reports_loss():
    from adamlm import eval_full
    old = [_row("instruction", "red apple green"), _row("greeting", "hello red apple")]
    tied = [_row("instruction", "red apple green"), _row("greeting", "hello red apple")]
    regressed = {"accuracy": 0.2222}
    worse = {"accuracy": 0.1852}  # -0.037, beyond the 0.02 tolerance
    verdict = auto_train.compare_evals(old, tied, old_suite=regressed, new_suite=worse,
                                       old_validation_loss=2.5259, new_validation_loss=2.6062)
    assert verdict["promote"] is False and "capability" in verdict["reason"]
    assert verdict["scores"]["suite_delta"] == pytest.approx(-0.037, abs=1e-4)
    assert verdict["scores"]["old_validation_loss"] == 2.5259
    assert verdict["scores"]["new_validation_loss"] == 2.6062
    # Same gate as the manual scorecard on identical inputs.
    card = eval_full.scorecard(
        [{"id": "a", "kind": "instruction", "reply": "red apple green",
          "reference": "red apple green", "checks": {}}],
        [{"id": "a", "kind": "instruction", "reply": "red apple green",
          "reference": "red apple green", "checks": {}}],
        old_suite=regressed, new_suite=worse)
    assert card["promote"] is False and "capability" in card["reason"]


def test_compare_loss_alone_never_promotes():
    old = [_row("instruction", "red apple green"), _row("greeting", "hello red apple")]
    worse_instr = [_row("instruction", "I know ."), _row("greeting", "hello red apple")]
    verdict = auto_train.compare_evals(old, worse_instr,
                                       old_validation_loss=2.9, new_validation_loss=1.1)
    assert verdict["promote"] is False and "instructions regressed" in verdict["reason"]
    assert verdict["scores"]["new_validation_loss"] == 1.1  # reported, not decisive


def test_compare_suite_tie_with_losses_still_promotes():
    old = [_row("instruction", "red apple"), _row("greeting", "hello there red")]
    new = [_row("instruction", "red apple"), _row("greeting", "hello there red")]
    suite = {"accuracy": 0.5}
    verdict = auto_train.compare_evals(old, new, old_suite=suite, new_suite=dict(suite),
                                       old_validation_loss=2.5, new_validation_loss=2.4)
    assert verdict["promote"] is True and "promoted" in verdict["reason"]
    assert verdict["scores"]["suite_delta"] == 0
    # Probes-only calls keep the legacy score keys working.
    legacy = auto_train.compare_evals(old, new)
    assert legacy["promote"] is True
    assert legacy["scores"]["suite_delta"] is None
    assert legacy["scores"]["old_validation_loss"] is None


def _write_pointer(root, evaluation):
    path = Path(root) / "results" / "assistant_default.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run": "sft-assistant", "checkpoint": "x",
                                "eval": evaluation}), encoding="utf-8")


def test_read_skill_profile_understands_saved_eval_keys(tmp_path):
    assert auto_train.read_skill_profile(tmp_path) is None  # no pointer yet
    # Real approved-pointer shape: per-side keys, current head (_new) wins.
    _write_pointer(tmp_path, {"instr_f1_old": 0.2716, "instr_f1_new": 0.3129, "instr_n": 4,
                              "conv_f1_old": 0.0, "conv_f1_new": 0.0541, "conv_n": 2,
                              "hygiene_problems": []})
    assert auto_train.read_skill_profile(tmp_path) == {"instr_f1": 0.3129, "conv_f1": 0.0541}
    # Legacy flat keys still work.
    _write_pointer(tmp_path, {"instr_f1": 0.2, "conv_f1": 0.1})
    assert auto_train.read_skill_profile(tmp_path) == {"instr_f1": 0.2, "conv_f1": 0.1}
    # Rollback / unevaluated pointers mean "no profile", never a crash.
    _write_pointer(tmp_path, {"rollback": True})
    assert auto_train.read_skill_profile(tmp_path) is None
    _write_pointer(tmp_path, {"instr_f1_new": 0.3})  # half a pair is unusable
    assert auto_train.read_skill_profile(tmp_path) is None


def test_preflight_uses_fixed_skill_profile_for_tilt(tmp_path):
    runs = _standard_runs(tmp_path)
    # Approved shape: conv 0.0541 << instr 0.3129 -> +5pp toward conversation.
    _write_pointer(tmp_path, {"instr_f1_old": 0.2716, "instr_f1_new": 0.3129,
                              "conv_f1_old": 0.0, "conv_f1_new": 0.0541})
    pf = auto_train.preflight(5_000, 600, tmp_path, runs)
    assert pf["allocation"]["dd_share"] == pytest.approx(0.75)
    assert "tilted +5pp" in pf["allocation"]["basis"]


def test_preflight_budget_states_repeat_policy_and_creates_nothing(tmp_path):
    runs = _standard_runs(tmp_path)  # fixture epoch is 10k tokens
    before = {p for p in Path(tmp_path).rglob("*")}
    pf = auto_train.preflight(100_000, 600, tmp_path, runs, pass_policy="repeat")
    assert {p for p in Path(tmp_path).rglob("*")} == before  # preview is read-only
    budget = pf["budget"]
    assert budget["requested_tokens"] == 100_000
    assert budget["effective_tokens"] == 100_000
    assert budget["batchable_epoch_tokens"] == 10_000
    assert not budget["capped"] and budget["repeated_data"]
    assert budget["pass_policy"] == "repeat"
    assert "not unique" in budget["note"]
    assert len(pf["stages"]) == 10


def test_unigram_f1_edge_cases():
    assert auto_train.unigram_f1("", "ref") == 0.0
    # hyp {a:2,b:1}, ref {a:1,b:1,c:1}: overlap 2, P=R=2/3 -> F1 2/3.
    assert auto_train.unigram_f1("a a b", "a b c") == pytest.approx(2 / 3)


def test_plan_overview_without_plans_and_pointer_validation(tmp_path):
    assert auto_train.plan_overview(tmp_path) == {"status": "none"}
    assert auto_train.read_assistant_default(tmp_path) is None
    bad = Path(tmp_path) / "results" / "assistant_default.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(json.dumps({"run": "x", "checkpoint": "nope.pt"}), encoding="utf-8")
    assert auto_train.read_assistant_default(tmp_path) is None


# --------------------------------------------------------------------------
# web endpoint wiring (fake planner/executor, no processes)
# --------------------------------------------------------------------------

def test_auto_start_refuses_duplicate_trainer(tmp_path, monkeypatch):
    from adamlm import web
    import types
    monkeypatch.setattr(web, "_active_run", lambda runs=None: types.SimpleNamespace(name="other"))
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="other"):
        web._auto_start({"additional_tokens": "1000", "max_seconds": 60})


def test_auto_start_rejects_bad_inputs():
    from adamlm import web
    import pytest as _pytest
    with _pytest.raises(ValueError, match="[Pp]ositive"):
        web._auto_inputs({"additional_tokens": "0", "max_seconds": 60})
    with _pytest.raises(ValueError, match="[Pp]ositive"):
        web._auto_inputs({"additional_tokens": "1000", "max_seconds": -5})


def test_auto_stop_reports_idle_and_stop_marks_plan(tmp_path, monkeypatch):
    from adamlm import web
    monkeypatch.setattr(auto_train, "latest_plan", lambda root: None)
    assert web._auto_stop()["already_stopped"] is True
    plan = _plan_dict(tmp_path)
    auto_train.write_plan(plan, tmp_path)
    monkeypatch.setattr(auto_train, "latest_plan",
                        lambda root: auto_train.read_plan(plan["plan_id"], tmp_path))
    real_stop = auto_train.request_plan_stop
    monkeypatch.setattr(auto_train, "request_plan_stop",
                        lambda plan_id, root, ctx=None: real_stop(plan_id, tmp_path, ctx))
    result = web._auto_stop()
    assert result["already_stopped"] is False
    assert auto_train.read_plan(plan["plan_id"], tmp_path)["status"] == "stopped"


def test_auto_start_writes_plan_and_spawns_supervisor(tmp_path, monkeypatch):
    from adamlm import web
    runs = _standard_runs(tmp_path)
    monkeypatch.setattr(web, "_active_run", lambda runs=None: None)
    monkeypatch.setattr(auto_train, "latest_plan", lambda root: None)
    real_preflight = auto_train.preflight
    monkeypatch.setattr(auto_train, "preflight",
                        lambda tokens, seconds, root, **kwargs: real_preflight(
                            tokens, seconds, tmp_path, runs, **kwargs))
    real_write = auto_train.write_plan
    monkeypatch.setattr(auto_train, "write_plan",
                        lambda plan, root: real_write(plan, tmp_path))
    spawned = []
    monkeypatch.setattr(web, "_spawn_supervisor", lambda plan_id: spawned.append(plan_id))
    result = web._auto_start({"additional_tokens": "5000", "max_seconds": 600})
    assert spawned == [result["plan_id"]]
    saved = auto_train.read_plan(result["plan_id"], tmp_path)
    assert saved["status"] == "running"
    assert saved["stages"][0]["parent_checkpoint"].endswith("step_00000100.pt")
    assert "sft-assistant" in saved["head"]["run"]


def test_auto_resume_rejects_done_and_adopts_stale(tmp_path, monkeypatch):
    from adamlm import web
    import pytest as _pytest
    monkeypatch.setattr(auto_train, "latest_plan", lambda root: None)
    with _pytest.raises(ValueError, match="[Nn]o resumable"):
        web._auto_resume({})
    plan = _plan_dict(tmp_path)
    plan["status"] = "done"
    auto_train.write_plan(plan, tmp_path)
    monkeypatch.setattr(auto_train, "latest_plan",
                        lambda root: auto_train.read_plan(plan["plan_id"], tmp_path))
    with _pytest.raises(ValueError, match="cannot resume"):
        web._auto_resume({})
    stopped = auto_train.read_plan(plan["plan_id"], tmp_path)
    stopped["status"] = "stopped"
    auto_train.touch(stopped, tmp_path)
    real_touch = auto_train.touch
    monkeypatch.setattr(auto_train, "touch", lambda plan, root: real_touch(plan, tmp_path))
    monkeypatch.setattr(web, "_active_run", lambda runs=None: None)
    spawned = []
    monkeypatch.setattr(web, "_spawn_supervisor", lambda plan_id: spawned.append(plan_id))
    result = web._auto_resume({"max_seconds": 300})
    assert spawned == [plan["plan_id"]]
    resumed = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert resumed["status"] == "running" and resumed["time"]["max_seconds"] == 300
    assert result["plan_id"] == plan["plan_id"]


def test_auto_resume_rejects_failed_plan_and_reports_last_checkpoint(tmp_path, monkeypatch):
    from adamlm import web
    plan = _plan_dict(tmp_path)
    checkpoint = Path(tmp_path) / plan["stages"][0]["run_dir"] / "checkpoints" / "step_00000100.pt"
    checkpoint.write_bytes(b"intact")
    plan["stages"][0].update(status="failed", checkpoint=str(checkpoint))
    plan["status"] = "failed"
    auto_train.write_plan(plan, tmp_path)
    monkeypatch.setattr(auto_train, "latest_plan",
                        lambda root: auto_train.read_plan(plan["plan_id"], tmp_path))
    with pytest.raises(ValueError, match="cannot resume in place.*step_00000100.pt"):
        web._auto_resume({})


# --------------------------------------------------------------------------
# full-pass exhaustion vs early exhaustion + checkpoint handoff
# --------------------------------------------------------------------------

def _data_stop_ctx(tmp_path, *, tokens_list, checkpoint_ok=True, eval_rows=None):
    """FakeCtx that returns data_stop/target_reached in sequence for focused tests."""
    base = FakeCtx(exit_state="instant",
                   eval_rows=eval_rows or _eval_rows("red apple green banana"))
    calls = {"n": 0}
    # Capture the bound method before overwriting to avoid recursion.
    orig_latest_fn = base.latest_checkpoint

    def read_status(run_dir):
        idx = min(calls["n"], len(tokens_list) - 1)
        calls["n"] += 1
        entry = tokens_list[idx]
        return {"state": entry["state"], "tokens": entry.get("tokens"),
                "detail": entry.get("detail") or entry["state"]}

    def latest(run_dir):
        if not checkpoint_ok:
            return None
        return orig_latest_fn(run_dir)

    base.read_stage_status = read_status
    base.latest_checkpoint = latest
    return base


def test_full_pass_exhaustion_advances_with_actual_tokens(tmp_path):
    """Full-pass data_stop within one batch is expected: done, not failed."""
    _standard_runs(tmp_path)  # tps = 2*1*8 = 16 in fixture config
    plan = _plan_dict(tmp_path, stages=1)
    plan["stages"][0]["target_tokens"] = 1000
    plan["stages"][0]["epoch_label"] = "pass 1"
    plan["effective_tokens"] = 1000
    auto_train.write_plan(plan, tmp_path)
    # 992 = 1000 - 8 tail (< one batch of 16); consumable full-pass boundary.
    ctx = _data_stop_ctx(tmp_path, tokens_list=[
        {"state": "data_stop: SFT train exhausted", "tokens": 992}])
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    done = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert done["stages"][0]["status"] == "done"
    # Actual processed tokens preserved, never the planned target.
    assert done["stages"][0]["tokens"] == 992
    assert done["stages"][0]["checkpoint"] is not None
    assert Path(done["stages"][0]["checkpoint"]).is_file()
    assert "full-pass boundary" in (done["stages"][0].get("note") or "")
    assert done["status"] == "done"


def test_early_exhaustion_partial_stage_stays_failed(tmp_path):
    """Partial-stage data_stop is unexpected even when the shortfall is small."""
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path, stages=1)
    plan["stages"][0]["target_tokens"] = 1000
    plan["stages"][0]["epoch_label"] = "pass 1 (partial, 12% of the mixed file)"
    plan["effective_tokens"] = 1000
    auto_train.write_plan(plan, tmp_path)
    ctx = _data_stop_ctx(tmp_path, tokens_list=[
        {"state": "data_stop: SFT train exhausted", "tokens": 992}])
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    failed = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert failed["stages"][0]["status"] == "failed"
    assert failed["status"] == "failed"
    assert "unexpected exhaustion" in (failed.get("failure") or "")


def test_early_exhaustion_large_shortfall_stays_failed(tmp_path):
    """Full-pass label but far from target is premature exhaustion, not a boundary."""
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path, stages=1)
    plan["stages"][0]["target_tokens"] = 1000
    plan["stages"][0]["epoch_label"] = "pass 1"
    plan["effective_tokens"] = 1000
    auto_train.write_plan(plan, tmp_path)
    # Shortfall 200 (>= one batch of 16) cannot be a tail-batch boundary.
    ctx = _data_stop_ctx(tmp_path, tokens_list=[
        {"state": "data_stop: SFT train exhausted", "tokens": 800}])
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    failed = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert failed["stages"][0]["status"] == "failed"
    assert failed["status"] == "failed"


def test_full_pass_exhaustion_requires_checkpoint(tmp_path):
    """No checkpoint file means unsaved work cannot be claimed: stay failed."""
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path, stages=1)
    plan["stages"][0]["target_tokens"] = 1000
    plan["stages"][0]["epoch_label"] = "pass 1"
    plan["effective_tokens"] = 1000
    auto_train.write_plan(plan, tmp_path)
    ctx = _data_stop_ctx(tmp_path, tokens_list=[
        {"state": "data_stop: SFT train exhausted", "tokens": 992}],
        checkpoint_ok=False)
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    failed = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert failed["stages"][0]["status"] == "failed"
    assert failed["stages"][0]["checkpoint"] is None


def test_checkpoint_handoff_does_not_carry_unbatchable_tail(tmp_path):
    """Stage 2 inherits stage-1 weights and keeps its own batch-rounded target."""
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path, stages=2)
    plan["stages"][0]["target_tokens"] = 1000
    plan["stages"][0]["epoch_label"] = "pass 1"
    plan["stages"][1]["target_tokens"] = 500
    plan["stages"][1]["epoch_label"] = "pass 2 (partial, 21% of the mixed file); rows repeat after each full pass"
    plan["effective_tokens"] = 1500
    plan["requested_tokens"] = 1500
    auto_train.write_plan(plan, tmp_path)
    # Legacy stage 1 has an 8-token tail. It is not debt in the fresh stage.
    ctx = _data_stop_ctx(tmp_path, tokens_list=[
        {"state": "data_stop: SFT train exhausted", "tokens": 992},
        {"state": "target_reached", "tokens": 500}])
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    done = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert [s["status"] for s in done["stages"]] == ["done", "done"]
    # Stage 1 keeps actual tokens, not target.
    assert done["stages"][0]["tokens"] == 992
    s1_ckpt = done["stages"][0]["checkpoint"]
    assert s1_ckpt is not None and Path(s1_ckpt).is_file()
    assert done["stages"][1]["target_tokens"] == 500
    # Handoff resolves to the same latest file the supervisor will launch from.
    assert done["stages"][1]["checkpoint"] is not None
    assert len(ctx.launched) == 2
    # No duplicate run dir: same dirs, stage 2 launched once from stage-1 weights.
    assert done["stages"][0]["run_dir"].endswith("-s1")
    assert done["stages"][1]["run_dir"].endswith("-s2")


def test_resuming_persisted_plan_skips_done_stage_without_duplicate(tmp_path):
    """Repaired plan shape (s1 done / s2 pending) resumes without replanning."""
    _standard_runs(tmp_path)
    plan = _plan_dict(tmp_path, stages=2)
    # Mirror the repaired 10M plan: s1 done with actual tokens, s2 pending
    # with remaining budget and explicit parent to s1 weights.
    s1_dir = Path(tmp_path) / plan["stages"][0]["run_dir"]
    s1_ckpt = s1_dir / "checkpoints" / "step_00000100.pt"
    s1_ckpt.parent.mkdir(parents=True, exist_ok=True)
    s1_ckpt.write_bytes(b"ckpt-s1")
    plan["stages"][0]["status"] = "done"
    plan["stages"][0]["tokens"] = 992
    plan["stages"][0]["target_tokens"] = 1000
    plan["stages"][0]["checkpoint"] = str(s1_ckpt)
    plan["stages"][0]["epoch_label"] = "pass 1"
    plan["stages"][1]["status"] = "pending"
    plan["stages"][1]["tokens"] = None
    plan["stages"][1]["target_tokens"] = 508  # remaining = 1500 - 992
    plan["stages"][1]["checkpoint"] = None
    plan["stages"][1]["parent_checkpoint"] = str(s1_ckpt)
    plan["stages"][1]["parent_stage"] = 1
    plan["stages"][1]["epoch_label"] = "pass 2 (partial, 21% of the mixed file); rows repeat after each full pass"
    plan["effective_tokens"] = 1500
    plan["requested_tokens"] = 1500
    plan["status"] = "stopped"
    plan["failure"] = None
    plan["stop_reason"] = "stage 1 full-pass boundary; resume continues stage 2"
    auto_train.write_plan(plan, tmp_path)
    ctx = FakeCtx(exit_state="instant",
                  eval_rows=_eval_rows("red apple green banana"))
    # Resume exactly as the dashboard does: stopped -> running, same plan id.
    resumed = auto_train.read_plan(plan["plan_id"], tmp_path)
    resumed["status"] = "running"
    auto_train.touch(resumed, tmp_path)
    Supervisor(plan["plan_id"], tmp_path, ctx).run()
    done = auto_train.read_plan(plan["plan_id"], tmp_path)
    assert [s["status"] for s in done["stages"]] == ["done", "done"]
    # Same dirs, no duplicate run, no restart from the original head.
    assert done["stages"][0]["run_dir"] == plan["stages"][0]["run_dir"]
    assert done["stages"][1]["run_dir"] == plan["stages"][1]["run_dir"]
    assert done["stages"][0]["tokens"] == 992
    assert len(ctx.launched) == 1  # only stage 2 launched
    assert done["plan_id"] == plan["plan_id"]
