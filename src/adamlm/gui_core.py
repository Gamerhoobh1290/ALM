"""Read-only run inspection and launch planning for the AdamLM desktop UI."""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout


ROOT = Path(__file__).resolve().parents[2]
TERMINAL_STATES = {"stopped", "target_reached", "session_limit", "disk_pause", "error", "interrupted"}
# States the trainer writes while it still holds the run lock.
LIVE_STATES = {"initializing", "verifying_data", "running", "saving"}
SESSION_FILE = "web-session.json"


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def parameter_count(model: dict) -> int:
    """Exact count for DecoderTransformer, whose token embedding and head are tied."""
    vocab, context = int(model["vocab_size"]), int(model["block_size"])
    layers, width = int(model["n_layer"]), int(model["n_embd"])
    return vocab * width + context * width + layers * (12 * width * width + 13 * width) + 2 * width


def format_number(value) -> str:
    if value is None:
        return "Unavailable"
    value = float(value)
    for suffix, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{suffix}"
    return f"{int(value):,}"


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "Unavailable"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h {minutes:02d}m" if hours else (f"{minutes}m {secs:02d}s" if minutes else f"{secs}s")


def trainer_active(run_dir: Path) -> bool:
    try:
        with FileLock(str(run_dir / ".training.lock"), timeout=0):
            return False
    except Timeout:
        return True


def stop_pending(run_dir: Path) -> bool:
    """Whether a graceful stop has been requested but the trainer still holds the lock."""
    try:
        return (run_dir / "STOP_REQUESTED").is_file()
    except OSError:
        return False


def file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def read_session(run_dir: Path) -> dict:
    """Metadata the dashboard wrote when it launched this run's session (if any)."""
    return read_json(run_dir / SESSION_FILE, {})


def write_session(run_dir: Path, value: dict) -> None:
    atomic_json(run_dir / SESSION_FILE, value)


def session_max_seconds(run_dir: Path) -> float | None:
    """Maximum session time in seconds, or None for run-until-stopped / unknown.

    Prefers the dashboard-written session file; falls back to the trainer's own
    ``start`` event in metrics.jsonl so CLI-launched runs still report a limit.
    """
    session = read_session(run_dir)
    value = session.get("max_seconds")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    try:
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "start" and isinstance(row.get("max_session_seconds"), (int, float)):
                return float(row["max_session_seconds"])
    except OSError:
        pass
    return None


def session_started_at(run_dir: Path) -> float | None:
    """Wall-clock start of the current session, or None when unknown.

    Only the dashboard-written session file carries a trustworthy start time;
    anything else would be invented, so callers must render None as unavailable.
    """
    value = read_session(run_dir).get("started_at")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def latest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    return paths[-1] if paths else None


@dataclass
class RunInfo:
    directory: Path
    launcher: dict
    status: dict
    checkpoint: Path | None
    active: bool

    @property
    def name(self):
        return self.directory.name

    @property
    def target(self):
        return int(self.launcher.get("target_tokens") or self.status.get("target_tokens") or 0)

    @property
    def tokens(self):
        return int(self.status.get("tokens") or 0)

    @property
    def remaining(self):
        return max(0, self.target - self.tokens)


def display_state(run: RunInfo) -> str:
    """User-facing training state, independent of which run is selected.

    Combines the trainer-written ``status.json`` state with the lock-based
    liveness probe and the pending stop request, so the dashboard can show
    running / stopping / stopped / completed / failed accurately.
    """
    raw = str(run.status.get("state") or "unknown")
    if run.active:
        if stop_pending(run.directory) or raw == "saving":
            return "stopping" if stop_pending(run.directory) else "saving"
        return raw if raw in LIVE_STATES else raw
    if raw in LIVE_STATES:
        # The trainer held the lock while writing this state but is gone now:
        # it crashed or was killed before saving a terminal state.
        return "interrupted"
    return raw


def checkpoint_category(run: RunInfo, *, production_name: str = "bpe512-local-run",
                        smoke_markers: tuple[str, ...] = ("smoke", "benchmark", "test", "web-ui", "web-stage", "gui-", "launcher-")) -> str:
    """Playground grouping: production, general-pretraining, sft, or smoke-test.

    Smoke-test artifacts are always labeled as such even when they belong to
    an SFT or pretraining stage, so experiments are never mistaken for
    production or serious stage checkpoints.
    """
    name = run.name.lower()
    if any(marker in name for marker in smoke_markers):
        return "smoke-test"
    if run.name == production_name:
        return "production"
    stage = run.launcher.get("stage") or run.status.get("stage")
    if stage == "sft":
        return "sft"
    return "general-pretraining"


def active_runs(runs: list[RunInfo] | None = None, root: Path = ROOT) -> list[RunInfo]:
    """Runs whose trainer lock is currently held, rediscovered from disk.

    This uses only the established lock mechanism, so it keeps working after
    a backend restart when in-memory process handles are gone.
    """
    return [run for run in (runs if runs is not None else discover_runs(root)) if run.active]


def discover_runs(root: Path = ROOT) -> list[RunInfo]:
    results = root / "results"
    found = []
    if not results.exists():
        return found
    for directory in results.iterdir():
        if not directory.is_dir():
            continue
        launcher = read_json(directory / "launcher-config.json", {})
        status = read_json(directory / "status.json", {})
        checkpoint = latest_checkpoint(directory)
        if launcher or status or checkpoint:
            found.append(RunInfo(directory, launcher, status, checkpoint, trainer_active(directory)))
    return sorted(found, key=lambda run: run.directory.stat().st_mtime, reverse=True)


def load_metrics(run_dir: Path) -> list[dict]:
    rows = []
    try:
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return rows


def measured_throughput(run_dir: Path | None = None, root: Path = ROOT) -> float | None:
    candidates = [run_dir] if run_dir else []
    candidates.extend(run.directory for run in discover_runs(root) if run.directory != run_dir)
    for directory in candidates:
        if not directory:
            continue
        value = read_json(directory / "summary.json", {}).get("end_to_end_tokens_per_second")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def default_config(dataset: str) -> Path:
    if dataset == "tinystories":
        name = "bpe512-local.json"
    elif dataset == "dolly":
        name = "sft-dolly.json"
    elif dataset == "dailydialog":
        name = "sft-dailydialog.json"
    else:
        name = "pretrain-general.json"
    return ROOT / "config" / name


def config_for_run(run: RunInfo, root: Path = ROOT) -> Path:
    recorded = run.launcher.get("config")
    if recorded:
        path = Path(recorded)
        path = path if path.is_absolute() else root / path
        if path.is_file():
            return path
    for path in (root / "config").glob("*.json"):
        cfg = read_json(path, {})
        configured = cfg.get("run_dir")
        if configured and (root / configured).resolve() == run.directory.resolve():
            return path
    return default_config(run.launcher.get("dataset") or run.status.get("dataset") or "tinystories")


def validate_mixture(value: str) -> None:
    items = value.split(",")
    if not items:
        raise ValueError("Enter custom mixture weights")
    allowed = {"tinystories", "wikitext103", "fineweb_edu"}
    seen = set()
    for item in items:
        try:
            name, raw_weight = item.split("=", 1)
            weight = float(raw_weight)
        except (ValueError, TypeError):
            raise ValueError("Use mixture syntax such as wikitext103=0.7,fineweb_edu=0.3") from None
        if name not in allowed or name in seen or weight <= 0:
            raise ValueError("Mixture datasets must be unique and weights must be positive")
        seen.add(name)


def unique_stage_dir(parent_name: str, root: Path = ROOT) -> Path:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", parent_name).strip("-") or "adamlm"
    base = root / "results" / f"{stem}-continuation-{time.strftime('%Y%m%d-%H%M%S')}"
    candidate, index = base, 2
    while candidate.exists():
        candidate = Path(f"{base}-{index}")
        index += 1
    return candidate


@dataclass
class TrainingPlan:
    mode: str
    run_dir: Path
    dataset: str
    checkpoint: Path | None
    additional_tokens: int
    cumulative_tokens: int
    remaining_tokens: int
    effective_target: int
    updates: int
    total_updates: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_steps: int
    throughput: float | None
    estimated_seconds: float | None
    params: int
    checkpoint_bytes: int
    command: list[str]


def build_plan(*, mode: str, dataset: str, additional_tokens: int, duration_seconds: float | None,
               selected_run: RunInfo | None = None, parent_checkpoint: Path | None = None,
               mixture: str | None = None, run_dir: Path | None = None, root: Path = ROOT,
               config_path: Path | str | None = None) -> TrainingPlan:
    if mode not in {"resume", "continuation"}:
        raise ValueError("Choose resume or continuation mode")
    if additional_tokens <= 0:
        raise ValueError("Additional tokens must be positive")
    if mode == "resume":
        if not selected_run or not selected_run.checkpoint:
            raise ValueError("Select an existing run with a checkpoint")
        dataset = selected_run.launcher.get("dataset") or selected_run.status.get("dataset") or dataset
        run_dir = selected_run.directory
        config_path = config_for_run(selected_run, root)
        config = read_json(config_path)
        target = selected_run.target
        cumulative = selected_run.tokens
        remaining = max(0, target - cumulative)
        if remaining == 0:
            raise ValueError("This target is complete. Create a continuation stage for more tokens.")
        additional_tokens = remaining
        checkpoint = selected_run.checkpoint
        command = [sys.executable, "-u", "-m", "adamlm.bpe_train", "--auto-resume", "--config", str(config_path), "--run-dir", str(run_dir)]
    else:
        checkpoint = Path(parent_checkpoint) if parent_checkpoint else None
        if not checkpoint or not checkpoint.is_file():
            raise ValueError("Choose a compatible parent checkpoint")
        config_path = Path(config_path) if config_path else default_config(dataset)
        config = read_json(config_path)
        run_dir = Path(run_dir) if run_dir else unique_stage_dir(checkpoint.parent.parent.name, root)
        parent_run = next((run for run in discover_runs(root)
                           if run.checkpoint and run.checkpoint.resolve() == checkpoint.resolve()), None)
        cumulative, remaining, target = (parent_run.tokens if parent_run else 0), additional_tokens, additional_tokens
        command = [sys.executable, "-u", "-m", "adamlm.bpe_train", "--auto-resume", "--config", str(config_path),
                   "--run-dir", str(run_dir), "--dataset", dataset, "--target-tokens", str(target),
                   "--parent-checkpoint", str(checkpoint)]
        if dataset == "mixture":
            if not mixture:
                raise ValueError("Enter positive custom mixture weights")
            validate_mixture(mixture)
            command += ["--mixture", mixture]
    if duration_seconds is None:
        command.append("--run-until-stopped")
    else:
        if duration_seconds <= 0:
            raise ValueError("Session duration must be positive")
        command += ["--max-seconds", str(float(duration_seconds))]
    model = config["model"]
    tokens_per_update = int(config["batch_size"]) * int(config["accumulation"]) * int(model["block_size"])
    updates = math.ceil(remaining / tokens_per_update)
    total_updates = math.ceil(target / tokens_per_update)
    throughput = measured_throughput(selected_run.directory if selected_run else None, root)
    estimate = remaining / throughput if throughput else None
    params = parameter_count(model)
    checkpoint_bytes = params * 16 + 32 * 2**20
    effective_target = total_updates * tokens_per_update if mode == "resume" else cumulative + total_updates * tokens_per_update
    return TrainingPlan(mode, Path(run_dir), dataset, checkpoint, additional_tokens, cumulative, remaining,
                        effective_target, updates, total_updates, float(config["learning_rate"]),
                        float(config["minimum_learning_rate"]), int(config["warmup_steps"]), throughput,
                        estimate, params, checkpoint_bytes, command)


def request_stop(run_dir: Path) -> None:
    if trainer_active(run_dir):
        (run_dir / "STOP_REQUESTED").touch()


def gpu_telemetry() -> dict:
    command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            return {}
        values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
        return dict(utilization=values[0], memory_used_mib=values[1], memory_total_mib=values[2], temperature_c=values[3])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def disk_telemetry(root: Path = ROOT) -> dict:
    usage = shutil.disk_usage(root)
    return {"free": usage.free, "total": usage.total}
