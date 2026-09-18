import json
from filelock import FileLock
from adamlm.control import active, write_status


def test_status_updates_preserve_checkpoint(tmp_path):
    write_status(tmp_path, state="running", checkpoint="step_00000010.pt", checkpoint_step=10)
    write_status(tmp_path, state="stopped", step=12, tokens=49152)
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["checkpoint_step"] == 10
    assert state["tokens"] == 49152
    assert state["state"] == "stopped"
    assert not (tmp_path / "status.tmp").exists()


def test_active_uses_lock_not_stale_status(tmp_path):
    write_status(tmp_path, state="running")
    assert not active(tmp_path)


def test_status_preserves_and_displays_dataset(capsys, tmp_path, monkeypatch):
    write_status(tmp_path, state="saving", dataset="wikitext103", checkpoint_step=7)
    monkeypatch.setattr("sys.argv", ["control", "status", "--run-dir", str(tmp_path)])
    from adamlm.control import main
    main()
    assert "Dataset/source: wikitext103" in capsys.readouterr().out
    with FileLock(str(tmp_path / ".training.lock")):
        assert active(tmp_path)
    assert not active(tmp_path)
