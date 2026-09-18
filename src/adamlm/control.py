"""Small run-local stop request and status files; no resident service."""
import argparse
import json
import os
import time
from pathlib import Path
from filelock import FileLock, Timeout


def active(out):
    try:
        with FileLock(str(out / ".training.lock"), timeout=0):
            return False
    except Timeout:
        return True


def write_status(out, **updates):
    path = out / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(updates, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stop", "status"])
    parser.add_argument("--config", default="config/bpe512-local.json")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.run_dir or cfg["run_dir"])
    if not out.exists():
        print("No training run exists yet.")
        return
    running = active(out)
    if args.action == "stop":
        if not running:
            print("Training is not running; existing checkpoints are unchanged.")
            return
        (out / "STOP_REQUESTED").touch()
        print("Graceful stop requested. Wait for status 'stopped' and a saved checkpoint before closing the training window.")
        return
    path = out / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    launcher_path = out / "launcher-config.json"
    launcher = json.loads(launcher_path.read_text()) if launcher_path.exists() else {}
    print(f"Run: {out.resolve()}\nTrainer active: {running}\nState: {state.get('state', 'unknown')}")
    print(f"Stage: {state.get('stage', launcher.get('stage', 'not yet available'))}")
    print(f"Dataset/source: {state.get('dataset', state.get('source', 'not yet available'))}")
    for key in ("step", "tokens", "target_tokens", "train_loss", "validation_loss", "checkpoint", "checkpoint_step", "updated_at", "detail"):
        print(f"{key}: {state.get(key, 'not yet available')}")
    checkpoint = Path(state["checkpoint"]) if state.get("checkpoint") else None
    if checkpoint and not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    print(f"Latest checkpoint status: {'complete' if checkpoint and checkpoint.is_file() else 'not found'}")
    print(f"Stop pending: {(out / 'STOP_REQUESTED').exists()}")
    if not running and state.get("state") in ("initializing", "verifying_data", "running", "saving"):
        print("Trainer exited unexpectedly; displayed progress may be stale. Resume uses only the latest complete checkpoint.")


if __name__ == "__main__":
    main()
