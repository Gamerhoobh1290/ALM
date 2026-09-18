"""Storage-budget changes must stay resume-compatible; lineage changes must not.

Regression test for the disk_pause run whose project outgrew its 15 GiB cap:
raising project_budget_gb (operational, like max_session_seconds/run_dir)
must not trip the resume compatibility check, while tokenizer, validation,
or training-hyperparameter drift still refuses to resume.
"""
import copy
from pathlib import Path

import pytest
import torch

from adamlm import checkpoint as checkpoint_module
from adamlm.bpe_train import protocol_config
from adamlm.checkpoint import CheckpointManager
from adamlm.storage import StorageBudget


def _saved_config():
    return {
        "seed": 1337, "stage": "sft", "dataset": "dailydialog",
        "model": {"vocab_size": 4096, "block_size": 512},
        "tokenizer": "tokenizers/bpe4096/tokenizer.json",
        "run_dir": "results/sft-500m-dailydialog-1m-continuation-20260917191622",
        "target_tokens": 10000000, "learning_rate": 0.0001,
        "storage": {"project_budget_gb": 15, "cache_limit_gb": 2,
                    "minimum_free_disk_gb": 5, "checkpoint_keep": 3,
                    "disk_poll_seconds": 30},
    }


def _compatible(saved, current):
    return protocol_config(saved) == protocol_config(current)


def test_storage_cap_raise_stays_resume_compatible():
    saved = _saved_config()
    current = copy.deepcopy(saved)
    current["storage"] = dict(saved["storage"], project_budget_gb=25)
    assert _compatible(saved, current)


def test_lineage_changes_still_refuse_resume():
    saved = _saved_config()
    for mutate in (lambda c: c.update(learning_rate=0.0002),
                   lambda c: c.update(tokenizer="tokenizers/other.json"),
                   lambda c: c.update(dataset="dolly")):
        current = copy.deepcopy(saved)
        mutate(current)
        assert not _compatible(saved, current)


def test_storage_scan_survives_file_deleted_mid_scan(tmp_path, monkeypatch):
    """A lock file vanishing between is_file() and stat() must not abort the scan.

    This crashed a real run: the guard runs inside the trainer's checkpoint
    save, so FileNotFoundError there ended training at step 232.
    """
    (tmp_path / "keep.bin").write_bytes(b"x" * 100)
    (tmp_path / ".training.lock").write_bytes(b"lock")
    real_stat = Path.stat
    seen = {"n": 0}

    def racing_stat(self, *args, **kwargs):
        if self.name == ".training.lock":
            seen["n"] += 1
            if seen["n"] > 1:  # is_file() succeeds, the size lookup races
                raise FileNotFoundError(2, "The system cannot find the file specified")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", racing_stat)
    assert StorageBudget(tmp_path).project_bytes() == 100


def test_operational_fields_stay_excluded():
    saved = _saved_config()
    current = copy.deepcopy(saved)
    current.update(run_dir="results/elsewhere", max_session_seconds=60)
    assert _compatible(saved, current)
    assert "storage" not in protocol_config(saved)


def test_interrupted_checkpoint_save_preserves_last_intact_file(tmp_path, monkeypatch):
    class Model(torch.nn.Linear):
        def config_dict(self):
            return {"kind": "test"}

    model = Model(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(tmp_path, keep=2)
    intact = manager.save(1, model, optimizer, {"cursor": 7})
    intact_bytes = intact.read_bytes()

    def interrupted_save(state, path):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated interrupted save")

    monkeypatch.setattr(checkpoint_module.torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        manager.save(2, model, optimizer, {"cursor": 8})
    assert manager.latest() == intact
    assert intact.read_bytes() == intact_bytes
    assert not list(tmp_path.glob("*.tmp"))


def test_retention_protects_approved_or_selected_milestone(tmp_path):
    class Model(torch.nn.Linear):
        def config_dict(self):
            return {"kind": "test"}

    model = Model(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    protected = tmp_path / "step_00000001.pt"
    manager = CheckpointManager(tmp_path, keep=2, protected=[protected])
    for step in range(1, 5):
        manager.save(step, model, optimizer, {"cursor": step})
    assert protected.is_file()
    assert (tmp_path / "step_00000004.pt").is_file()
