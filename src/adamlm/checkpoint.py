"""Atomic checkpoints containing model, optimizer, RNG, and dataset cursor state."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, directory: str | Path, keep=3, protected=()):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = max(1, int(keep))
        self.protected = {Path(path).resolve() for path in protected}

    def save(self, step, model, optimizer, dataset_state, extra=None) -> Path:
        state = {
            "model_config": model.config_dict(),
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dataset": dataset_state,
            "rng": {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "extra": extra or {},
        }
        target = self.directory / f"step_{step:08d}.pt"
        temporary = target.with_suffix(".tmp")
        try:
            torch.save(state, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        checkpoints = sorted(self.directory.glob("step_*.pt"))
        removable = [path for path in checkpoints[:-self.keep]
                     if path.resolve() not in self.protected]
        for old in removable:
            old.unlink(missing_ok=True)
        return target

    def pin(self, source: str | Path, name="best.pt") -> Path:
        """Atomically preserve a checkpoint outside rolling step retention."""
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = self.directory / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def latest(self) -> Path | None:
        paths = sorted(self.directory.glob("step_*.pt"))
        return paths[-1] if paths else None

    @staticmethod
    def load(path, model, optimizer, dataset_stream) -> int:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        dataset_stream.load_state_dict(state["dataset"])
        random.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        return int(state["step"])
