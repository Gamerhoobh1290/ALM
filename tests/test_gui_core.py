from pathlib import Path

from adamlm.gui_core import RunInfo, build_plan, parameter_count, validate_mixture


def test_parameter_count_matches_production_model():
    model = {"vocab_size": 4096, "block_size": 512, "n_layer": 4, "n_head": 6, "n_embd": 384}
    assert parameter_count(model) == 8_868_096


def test_resume_preserves_immutable_target(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "step_00000010.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    run = RunInfo(run_dir, {"target_tokens": 1_000_000, "dataset": "tinystories"},
                  {"tokens": 400_000, "target_tokens": 1_000_000}, checkpoint, False)
    plan = build_plan(mode="resume", dataset="tinystories", additional_tokens=10_000_000,
                      duration_seconds=60, selected_run=run, root=tmp_path)
    assert plan.additional_tokens == plan.remaining_tokens == 600_000
    assert "--target-tokens" not in plan.command
    assert "--auto-resume" in plan.command


def test_continuation_uses_additional_tokens_as_new_stage_target(tmp_path):
    checkpoint = tmp_path / "parent" / "checkpoints" / "step_00000010.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    run_dir = tmp_path / "results" / "next"
    plan = build_plan(mode="continuation", dataset="mixture", additional_tokens=1_000_000,
                      duration_seconds=None, parent_checkpoint=checkpoint,
                      mixture="tinystories=0.2,wikitext103=0.8", run_dir=run_dir, root=tmp_path)
    assert plan.run_dir == run_dir
    assert plan.remaining_tokens == 1_000_000
    assert plan.effective_target >= 1_000_000
    assert "--parent-checkpoint" in plan.command
    assert "--run-until-stopped" in plan.command


def test_continuation_reports_parent_lineage_tokens(tmp_path):
    parent_dir = tmp_path / "results" / "parent"
    checkpoint = parent_dir / "checkpoints" / "step_00000010.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (parent_dir / "launcher-config.json").write_text(
        '{"target_tokens": 1000000, "dataset": "tinystories"}', encoding="utf-8"
    )
    (parent_dir / "status.json").write_text(
        '{"tokens": 400000, "target_tokens": 1000000}', encoding="utf-8"
    )

    plan = build_plan(mode="continuation", dataset="tinystories", additional_tokens=1000000,
                      duration_seconds=60, parent_checkpoint=checkpoint,
                      run_dir=tmp_path / "results" / "next", root=tmp_path)

    assert plan.cumulative_tokens == 400000
    assert plan.effective_target > plan.cumulative_tokens
    assert "--target-tokens" in plan.command


def test_mixture_validation_rejects_nonpositive_or_duplicate_weights():
    validate_mixture("tinystories=0.2,wikitext103=0.8")
    for value in ("tinystories=0", "tinystories=1,tinystories=2", "unknown=1"):
        try:
            validate_mixture(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
