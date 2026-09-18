"""Project-local storage budget and low-disk pause guard."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


class LowDiskSpace(RuntimeError):
    pass


def _tree_bytes(directory: Path) -> int:
    """Sum file sizes under a directory, tolerating files that vanish mid-scan.

    Transient files (a run's .training.lock, temp files from a concurrent
    stage) can disappear between enumeration and stat. A raced file must not
    abort a storage check, because check() runs from the trainer's checkpoint
    save path and an exception there kills an otherwise healthy run.
    """
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


class StorageBudget:
    def __init__(self, project_root: str | Path, budget_gb=15.0, cache_limit_gb=2.0, minimum_free_gb=5.0):
        self.root = Path(project_root).resolve()
        self.budget_bytes = int(budget_gb * 1024**3)
        self.cache_limit_bytes = int(cache_limit_gb * 1024**3)
        self.minimum_free_bytes = int(minimum_free_gb * 1024**3)

    def project_bytes(self) -> int:
        return _tree_bytes(self.root)

    def cache_bytes(self) -> int:
        cache = self.root / ".cache"
        return _tree_bytes(cache) if cache.exists() else 0

    def snapshot(self) -> dict:
        usage = shutil.disk_usage(self.root)
        return {
            "project_bytes": self.project_bytes(),
            "cache_bytes": self.cache_bytes(),
            "disk_free_bytes": usage.free,
            "budget_bytes": self.budget_bytes,
            "cache_limit_bytes": self.cache_limit_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
        }

    def check(self, reserve_bytes: int = 0) -> None:
        state = self.snapshot()
        if state["project_bytes"] + reserve_bytes > self.budget_bytes:
            raise LowDiskSpace("project storage budget exceeded")
        if state["cache_bytes"] > self.cache_limit_bytes:
            raise LowDiskSpace("project cache limit exceeded; clean .cache before continuing")
        if state["disk_free_bytes"] - reserve_bytes < self.minimum_free_bytes:
            raise LowDiskSpace("drive free space is below the configured safety floor")

    def check_free_disk(self, reserve_bytes: int = 0) -> None:
        """Cheap per-step check; perform full project/cache checks periodically and before writes."""
        if shutil.disk_usage(self.root).free - reserve_bytes < self.minimum_free_bytes:
            raise LowDiskSpace("drive free space is below the configured safety floor")

    def wait_until_safe(self, poll_seconds=30, pause_file: str | Path | None = None) -> None:
        marker = Path(pause_file) if pause_file else self.root / "PAUSED_LOW_DISK"
        while True:
            try:
                self.check()
                if marker.exists():
                    marker.unlink()
                return
            except LowDiskSpace as exc:
                marker.write_text(f"{exc}\n", encoding="utf-8")
                time.sleep(poll_seconds)
