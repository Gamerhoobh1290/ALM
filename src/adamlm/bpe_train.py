"""Explicitly launched BPE training; legacy byte training remains unchanged."""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import requests
import torch
from filelock import FileLock
from .bpe import BPETokenizer
from .checkpoint import CheckpointManager
from .local_data import LocalStoriesStream
from .dataset_selection import build_stream, load_validation_texts
from .sft_data import SFTStream, epoch_capacity, manifest_epoch_tokens
from .model import DecoderTransformer, ModelConfig
from .storage import StorageBudget, LowDiskSpace
from .training import autocast, update, generate
from .control import write_status


def learning_rate(step, total_steps, warmup, peak, floor):
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = min(1.0, (step-warmup) / max(1, total_steps-warmup))
    return floor + (peak-floor)*0.5*(1+math.cos(math.pi*progress))


def monitor_policy(cfg):
    """Stopping policy for a stage, resolved from config with safe defaults.

    ``warmup_steps`` and ``settled_lr`` exist because a stage's learning rate
    warms to a peak and then anneals: validation taken while the schedule is
    still moving is not comparable with the start-of-stage baseline, and the
    excursion reverses on its own. Verdicts wait for a comparable point.
    """
    monitor_cfg = cfg.get("validation_monitor", {}) or {}
    return {
        "patience": max(0, int(monitor_cfg.get("patience", 0))),
        "min_delta": max(0.0, float(monitor_cfg.get("min_delta", 0.0))),
        "max_regression": max(0.0, float(monitor_cfg.get("max_regression", math.inf))),
        "regression_patience": max(1, int(monitor_cfg.get("regression_patience", 2))),
        "warmup_steps": max(int(monitor_cfg.get("warmup_steps", cfg["warmup_steps"])),
                            int(cfg["warmup_steps"])),
        "settled_lr": float(cfg["minimum_learning_rate"]) * max(
            1.0, float(monitor_cfg.get("settled_lr_factor", 2.0))),
    }


def validation_verdict(monitor, val_loss, step, lr, policy):
    """What one validation check means. Pure: returns a new monitor and action.

    Actions are "improved" (a new best worth pinning), "deferred" (recorded,
    but the schedule has not settled enough to judge), "bad" (a settled check
    that did not improve), and "pause" (sustained evidence of regression).
    """
    monitor = dict(monitor)
    best = float(monitor.get("best_loss", val_loss))
    if val_loss < best - policy["min_delta"]:
        monitor.update(best_loss=val_loss, bad_validations=0, best_step=step,
                       regression_streak=0)
        return monitor, "improved", None
    if step < policy["warmup_steps"] or lr > policy["settled_lr"]:
        monitor["deferred_checks"] = int(monitor.get("deferred_checks", 0)) + 1
        return monitor, "deferred", None
    monitor["bad_validations"] = int(monitor.get("bad_validations", 0)) + 1
    breached = val_loss > best + policy["max_regression"]
    monitor["regression_streak"] = (int(monitor.get("regression_streak", 0)) + 1
                                    if breached else 0)
    sustained = policy["patience"] and monitor["bad_validations"] >= policy["patience"]
    regressed = monitor["regression_streak"] >= policy["regression_patience"]
    if not (sustained or regressed):
        return monitor, "bad", None
    cause = (f"exceeded the best by more than {policy['max_regression']:g} on "
             f"{monitor['regression_streak']} consecutive settled checks" if regressed else
             f"has not improved by {policy['min_delta']:g} for "
             f"{monitor['bad_validations']} settled checks")
    detail = (f"Validation {cause}; best {best:.6f} at step {monitor.get('best_step')}, "
              f"current {val_loss:.6f} at learning rate {lr:.2e}. "
              "Best and latest checkpoints are preserved for review.")
    return monitor, "pause", detail


def protocol_config(cfg):
    """Return only lineage-defining settings; paths, session limits, and storage caps are not lineage."""
    result = dict(cfg)
    result.pop("run_dir", None)
    result.pop("max_session_seconds", None)
    # Storage budgets are operational environment limits (like session limits):
    # raising them must never invalidate an otherwise compatible checkpoint.
    result.pop("storage", None)
    # The validation monitor is a stopping policy, not lineage: retuning when
    # a run pauses must never strand an existing checkpoint as unresumable.
    result.pop("validation_monitor", None)
    if result.get("stage", "pretrain") == "pretrain":
        result.pop("stage", None)
    return result


def protected_checkpoints(root: Path, cfg: dict) -> set[Path]:
    """Approved and operator-listed milestones are exempt from rolling cleanup."""
    protected = set()
    pointer = root / "results" / "assistant_default.json"
    try:
        value = json.loads(pointer.read_text(encoding="utf-8")).get("checkpoint")
        if value:
            path = Path(value)
            protected.add((path if path.is_absolute() else root / path).resolve())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    for value in cfg.get("storage", {}).get("protected_checkpoints", []):
        path = Path(value)
        protected.add((path if path.is_absolute() else root / path).resolve())
    return protected


def validation_batches(texts, tokenizer, context, device):
    # Keep story boundaries and all targets; pad only the last chunk of each story.
    result = []
    for text in texts:
        ids = tokenizer.encode_story(text)
        for offset in range(0, len(ids)-1, context):
            chunk = ids[offset:offset+context+1]
            n = len(chunk)-1
            x = torch.tensor([chunk[:-1] + [tokenizer.boundary_id]*(context-n)], device=device)
            y = torch.tensor([chunk[1:] + [-100]*(context-n)], device=device)
            result.append((x, y))
    return result


@torch.no_grad()
def validation_loss(model, batches):
    model.eval()
    total, count = 0.0, 0
    for x, y in batches:
        with autocast("bf16"):
            _, loss = model(x, y)
        n = int((y != -100).sum())
        total += loss.item()*n
        count += n
    model.train()
    return total/count


def sft_validation_batches(manifest, split, tokenizer, context, device, count=8, batch_size=4):
    stream = SFTStream(manifest, tokenizer, split=split)
    result = []
    for _ in range(count):
        try:
            x, y = stream.next_batch(batch_size, context)
        except StopIteration:
            break
        result.append((torch.tensor(x, device=device), torch.tensor(y, device=device)))
    if not result:
        raise ValueError(f"SFT {split} split has no validation batches")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/bpe512-local.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--target-tokens", type=int, help="Total target for a fresh run; must match on resume")
    parser.add_argument("--run-dir")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--duration", type=float, help="Alias for --max-seconds")
    parser.add_argument("--run-until-stopped", action="store_true", help="No session timer; stop remains graceful")
    parser.add_argument("--stage", choices=["pretrain", "sft"])
    parser.add_argument("--dataset", choices=["tinystories", "wikitext103", "fineweb_edu", "mixture", "dolly", "dailydialog"])
    parser.add_argument("--mixture", help="Explicit mixture, e.g. wikitext103=0.7,fineweb_edu=0.3")
    parser.add_argument("--parent-checkpoint", help="Trusted compatible checkpoint used to initialize a new stage")
    parser.add_argument("--repeat-epoch", action="store_true",
                        help="Approve a repeat pass: rewind the SFT corpus to its start and record "
                             "the stage as repeated data. Without this a new stage continues the "
                             "parent's pass and refuses to run past the end of the data.")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    saved_run_config = None
    if args.run_dir is not None:
        saved_path = Path(args.run_dir) / "launcher-config.json"
        if saved_path.exists():
            saved_run_config = json.loads(saved_path.read_text(encoding="utf-8"))
    dataset = args.dataset or (saved_run_config or {}).get("dataset") or cfg.get("dataset", "tinystories")
    stage = args.stage or (saved_run_config or {}).get("stage") or cfg.get("stage", "pretrain")
    if stage == "sft" and dataset not in ("dolly", "dailydialog"):
        parser.error("The prepared SFT stages currently use --dataset dolly or dailydialog")
    if stage == "pretrain" and dataset in ("dolly", "dailydialog"):
        parser.error("Dolly/DailyDialog require --stage sft")
    cfg["stage"] = stage
    if args.duration is not None:
        args.max_seconds = args.duration
    if args.run_until_stopped and args.max_seconds is not None:
        parser.error("--run-until-stopped cannot be combined with a duration")
    if dataset != "tinystories" and args.run_dir is None and not cfg.get("run_dir"):
        args.run_dir = f"results/{dataset}-stage"
    if args.run_dir is not None:
        cfg["run_dir"] = args.run_dir
    if dataset != "tinystories":
        cfg["dataset"] = dataset
    if args.mixture:
        components = []
        for item in args.mixture.split(","):
            name, weight = item.split("=", 1)
            if name not in ("tinystories", "wikitext103", "fineweb_edu"):
                parser.error("Mixture components must be tinystories, wikitext103, or fineweb_edu")
            components.append({"dataset": name, "weight": float(weight)})
        cfg["mixture"] = {"components": components}
    if args.target_tokens is not None:
        if args.target_tokens <= 0:
            parser.error("--target-tokens must be positive")
        cfg["target_tokens"] = args.target_tokens
    out = Path(args.run_dir or cfg["run_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # Shared GPU-session ownership: a CLI trainer must also respect the claim
    # held by the dashboard, Auto Train, or the Research Agent. Adoption
    # succeeds for a free GPU, a stale claim, the startup-grace window, or a
    # pre-claim naming this run; a live foreign claim refuses with a clear
    # error instead of competing for the GPU.
    from .gpu_session import adopt as _gpu_adopt, current as _gpu_current
    _root = Path(__file__).resolve().parents[2]
    _run_name = out.name
    _claim = _gpu_adopt(_root, kind="training", label=f"training:{_run_name}", run=_run_name)
    if _claim is None:
        _owner = _gpu_current(_root)
        detail = f" ({_owner.get('kind')}: {_owner.get('label')})" if _owner else ""
        parser.error(f"GPU is busy{detail}. "
                     "Stop the active session before starting another trainer.")
    with FileLock(str(out / ".training.lock"), timeout=0):
        if args.auto_resume:
            args.resume = any((out / "checkpoints").glob("step_*.pt"))
            # Reuse a launcher's saved target and mixture unless overridden.
            saved_config = out / "launcher-config.json"
            if args.resume and saved_config.exists():
                saved = json.loads(saved_config.read_text(encoding="utf-8"))
                if args.target_tokens is None:
                    cfg["target_tokens"] = saved["target_tokens"]
                if args.mixture is None and saved.get("mixture") is not None:
                    cfg["mixture"] = saved["mixture"]
                if args.parent_checkpoint is None and saved.get("parent_checkpoint") is not None:
                    args.parent_checkpoint = saved["parent_checkpoint"]
        (out / "STOP_REQUESTED").unlink(missing_ok=True)
        write_status(out, state="initializing", stage=stage, dataset=dataset, source=dataset, detail=None)
        try:
            run(cfg, args)
        except Exception as exc:
            write_status(out, state="error", detail=str(exc))
            raise
        finally:
            try:
                from .gpu_session import release as _gpu_release

                _gpu_release(Path(__file__).resolve().parents[2], kind="training")
            except Exception:
                pass


def run(cfg, args):
    start = time.perf_counter()
    seconds = math.inf if args.run_until_stopped else (args.max_seconds if args.max_seconds is not None else cfg["max_session_seconds"])
    if seconds <= 0 or cfg["precision"] != "bf16":
        raise ValueError("Positive session limit and BF16 precision required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a BF16-capable CUDA GPU")
    torch.set_num_threads(4)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    dataset = cfg.get("dataset", "tinystories")
    stage = cfg.get("stage", "pretrain")
    tok = BPETokenizer(cfg["tokenizer"])
    if stage == "pretrain" and cfg["data"]["split"] != "train":
        raise ValueError("Training source must use the train split")
    out = Path(args.run_dir or cfg["run_dir"])
    out.mkdir(parents=True, exist_ok=True)
    configured_validation = Path(cfg["validation"])
    if stage == "sft":
        val_path = configured_validation
        if not val_path.is_file():
            raise FileNotFoundError("Prepare the SFT dataset first (scripts/prepare_dolly.py or scripts/prepare_dailydialog.py)")
        validation = {"dataset": dataset}
    else:
        val_path = configured_validation if dataset == "tinystories" else out / "validation.json"
    if stage != "sft" and dataset != "tinystories" and val_path.exists():
        validation = json.loads(val_path.read_text(encoding="utf-8"))
    elif stage != "sft" and dataset != "tinystories":
        validation = {"dataset": dataset, "texts": load_validation_texts(dataset, config=cfg.get("mixture"))}
        val_path.write_text(json.dumps(validation, ensure_ascii=False), encoding="utf-8")
    elif stage != "sft":
        validation = json.loads(val_path.read_text(encoding="utf-8"))
    if stage == "pretrain" and dataset == "tinystories" and validation.get("split") != "validation":
        raise ValueError("Separate TinyStories validation split required")
    if dataset != "tinystories" and validation.get("dataset") != dataset:
        raise ValueError("Training and validation dataset selection mismatch")
    protocol = dict(config=protocol_config(cfg), tokenizer_sha256=tok.sha256,
                    validation_sha256=hashlib.sha256(val_path.read_bytes()).hexdigest())
    root = Path(__file__).resolve().parents[2]
    manager = CheckpointManager(out / "checkpoints", cfg["storage"]["checkpoint_keep"],
                                protected=protected_checkpoints(root, cfg))
    latest = manager.latest()
    requested_parent = args.parent_checkpoint or cfg.get("parent_checkpoint")
    stage_init = latest is None and not args.resume and (dataset != "tinystories" or requested_parent is not None)
    if args.resume and latest is None:
        raise ValueError("No compatible checkpoint exists for this dataset stage")
    if not args.resume and latest is not None:
        raise ValueError("Existing run requires --resume; use a new --run-dir for a new stage")
    state = None
    if args.resume:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        saved_protocol = state["extra"].get("bpe_protocol", {})
        if (saved_protocol.get("tokenizer_sha256") != protocol["tokenizer_sha256"] or
                saved_protocol.get("validation_sha256") != protocol["validation_sha256"] or
                protocol_config(saved_protocol.get("config", {})) != protocol["config"]):
            raise ValueError("Incompatible checkpoint: tokenizer, validation or training configuration changed")
    (out / "launcher-config.json").write_text(json.dumps({
        "config": str(Path(args.config)),
        "target_tokens": cfg["target_tokens"],
        "stage": stage,
        "dataset": dataset,
        "mixture": cfg.get("mixture"),
        "parent_checkpoint": args.parent_checkpoint or cfg.get("parent_checkpoint"),
        # Provenance only: an interrupted stage resumes from its checkpoint,
        # which already carries the cursor this flag chose at stage init.
        "repeat_epoch": bool(args.repeat_epoch),
    }, indent=2), encoding="utf-8")
    model = DecoderTransformer(ModelConfig(**cfg["model"])).cuda()
    if model.config.vocab_size != tok.vocab_size:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    write_status(out,state="verifying_data",dataset=dataset,source=dataset,detail="Verifying selected local source; no network access")
    if stage == "sft":
        storage = cfg["storage"]
        guard = StorageBudget(".", storage["project_budget_gb"], storage["cache_limit_gb"], storage["minimum_free_disk_gb"])
        stream = SFTStream(cfg["data"]["manifest"], tok, split="train", page_size=cfg["data"].get("page_size", 64))
    elif dataset == "tinystories":
        if not cfg.get("local_manifest"):
            raise ValueError("TinyStories requires the verified local corpus configuration")
        stream = LocalStoriesStream(cfg["local_manifest"],**cfg["data"], tokenizer=tok)
    else:
        storage = cfg["storage"]
        guard = StorageBudget(".", storage["project_budget_gb"], storage["cache_limit_gb"], storage["minimum_free_disk_gb"])
        stream = build_stream(dataset, tok, guard, config=cfg.get("mixture") if dataset == "mixture" else cfg.get("data"))
    if stage == "pretrain":
        stream.excluded_hashes = {hashlib.sha256(t.strip().encode()).hexdigest() for t in validation["texts"]}
    source_transition = None
    if stage_init:
        if requested_parent:
            parent = Path(requested_parent)
            if parent.is_dir():
                parents = sorted(parent.glob("step_*.pt"))
                parent = parents[-1] if parents else parent / "missing"
        else:
            parents = sorted(Path("results/bpe512-local-run/checkpoints").glob("step_*.pt"))
            parent = parents[-1] if parents else Path("missing")
        if not parent.is_file():
            raise ValueError("No compatible AdamLM parent checkpoint found")
        parent_state = torch.load(parent, map_location="cpu", weights_only=False)
        if parent_state["model_config"] != model.config_dict() or parent_state["extra"].get("tokenizer_sha256") not in (None, tok.sha256):
            raise ValueError("Latest parent checkpoint is incompatible with this model/tokenizer")
        model.load_state_dict(parent_state["model"])
        source_transition = dict(exact_dataset_resume=False, parent_checkpoint=str(parent), parent_step=parent_state["step"],
            parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(), optimizer_reset=True,
            reason="new dataset stage uses a fresh optimizer and schedule")
        if stage == "sft":
            # A new stage must say which data it is about to use. Continuing
            # the parent's pass is the default; repeating rows it already
            # trained on requires --repeat-epoch and is recorded as such.
            source_transition["dataset_cursor"] = stream.continue_from(
                parent_state.get("dataset"), repeat_epoch=args.repeat_epoch)
        else:
            stream.source_start_tokens = parent_state["step"] * cfg["batch_size"] * cfg["accumulation"] * model.config.block_size
        step = 0
        del parent_state
    else:
        step = manager.load(latest, model, optimizer, stream) if args.resume else 0
    previous_seconds = state["extra"]["elapsed_seconds"] if state else 0.0
    saved_monitor = state["extra"].get("validation_monitor") if state else None
    if state is not None:
        source_transition = state["extra"].get("source_transition")
    del state
    storage = cfg["storage"]
    guard = guard if 'guard' in locals() else StorageBudget(".", storage["project_budget_gb"], storage["cache_limit_gb"], storage["minimum_free_disk_gb"])
    reserve = model.parameter_count()*16 + 32*2**20
    guard.check(reserve)
    tokens_per_step = cfg["batch_size"]*cfg["accumulation"]*model.config.block_size
    total_steps = math.ceil(cfg["target_tokens"]/tokens_per_step)
    initial_tokens = stream.stats.tokens_consumed
    if stage == "sft":
        # Refuse to schedule past the end of the pass rather than discovering
        # it as an end-of-data stop partway through. Same arithmetic the
        # planner used, so a plan that passed preflight passes here too.
        epoch_tokens = manifest_epoch_tokens(cfg["data"]["manifest"])
        if epoch_tokens:
            capacity = epoch_capacity(epoch_tokens, tokens_per_step, stream.epoch_tokens)
            wanted = max(0, total_steps - step) * tokens_per_step
            if wanted > capacity["remaining_tokens"]:
                raise ValueError(
                    f"This stage asks for {wanted} tokens but only {capacity['remaining_tokens']} "
                    f"remain in pass {stream.epoch + 1} ({capacity['consumed_tokens']} of "
                    f"{capacity['batchable_tokens']} batchable tokens already trained; "
                    f"{capacity['discarded_tail_tokens']} tail tokens below one batch are never "
                    "trained). Lower the target to fit the remaining data, or approve a repeat "
                    "pass explicitly with --repeat-epoch, which records the rows as repeated.")
    if step >= total_steps:
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "already_target_reached", "step": step, "tokens": initial_tokens}) + "\n")
        write_status(out, state="target_reached", step=step, tokens=initial_tokens,
                     target_tokens=cfg["target_tokens"], detail="Target was already reached; checkpoint and summary unchanged.")
        return
    if stage == "sft":
        batches = sft_validation_batches(cfg["data"]["manifest"], "validation", tok, model.config.block_size, "cuda")
        holdout = sft_validation_batches(cfg["data"]["manifest"], "holdout", tok, model.config.block_size, "cuda")
    else:
        batches = validation_batches(validation["texts"][:32], tok, model.config.block_size, "cuda")
        holdout = validation_batches(validation["texts"][32:64], tok, model.config.block_size, "cuda")
    initial_validation = validation_loss(model, batches)
    monitor_cfg = cfg.get("validation_monitor", {}) if stage == "sft" else {}
    policy = monitor_policy(cfg) if stage == "sft" else None
    monitor = dict(saved_monitor or {"best_loss": initial_validation, "bad_validations": 0,
                                     "best_step": step, "regression_streak": 0,
                                     "deferred_checks": 0})
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    compute_seconds = data_seconds = validation_seconds = checkpoint_seconds = 0.0
    next_check = 0.0
    last_loss = None
    reason = "target_reached"
    pause_detail = None

    def emit(row):
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)

    def save():
        nonlocal checkpoint_seconds
        save_start = time.perf_counter()
        write_status(out, state="saving")
        guard.check(reserve)
        path = manager.save(step, model, optimizer, stream.state_dict(), dict(
            bpe_protocol=protocol, tokenizer_path=cfg["tokenizer"], tokenizer_sha256=tok.sha256,
            source_transition=source_transition,
            validation_monitor=monitor,
            elapsed_seconds=previous_seconds+time.perf_counter()-start))
        write_status(out, state="running", checkpoint=str(path), checkpoint_step=step)
        checkpoint_seconds += time.perf_counter() - save_start
        return path

    def sample():
        prompt = "User:\nWhat does the Windows Recycle Bin do?\n\nAssistant:\n" if stage == "sft" else "Once upon a time,"
        text = generate(model, prompt, new_tokens=100, tokenizer=tok)
        with (out / "samples.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(step=step, text=text))+"\n")

    emit(dict(event="start", step=step, params=model.parameter_count(), validation_loss=initial_validation,
              tokenizer_sha256=tok.sha256, max_session_seconds=None if math.isinf(seconds) else seconds))
    if not args.resume:
        save()
    write_status(out, state="running", step=step, tokens=stream.stats.tokens_consumed,
                 target_tokens=cfg["target_tokens"], train_loss=None, validation_loss=initial_validation,
                 checkpoint=str(manager.latest()), checkpoint_step=step, detail=None)
    (out / "PAUSED_LOW_DISK").unlink(missing_ok=True)
    next_status = 0.0
    try:
        while step < total_steps:
            if (out / "STOP_REQUESTED").exists():
                reason = "graceful_stop"
                break
            if time.perf_counter()-start >= seconds:
                reason = "session_limit"
                break
            guard.check_free_disk(reserve)
            if time.perf_counter() >= next_check:
                guard.check(reserve)
                next_check = time.perf_counter()+storage["disk_poll_seconds"]
            cursor = copy.deepcopy(stream.state_dict())
            fetch_start = time.perf_counter()
            microbatches = []
            try:
                for _ in range(cfg["accumulation"]):
                    x, y = stream.next_batch(cfg["batch_size"], model.config.block_size)
                    microbatches.append((torch.tensor(x, device="cuda"), torch.tensor(y, device="cuda")))
            except (requests.RequestException, StopIteration, ValueError, OSError):
                stream.load_state_dict(cursor)
                data_seconds += time.perf_counter()-fetch_start
                raise
            torch.cuda.synchronize()
            data_seconds += time.perf_counter()-fetch_start
            lr = learning_rate(step+1, total_steps, cfg["warmup_steps"], cfg["learning_rate"], cfg["minimum_learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            compute_start = time.perf_counter()
            loss = update(model, optimizer, microbatches, "bf16")
            torch.cuda.synchronize()
            compute_seconds += time.perf_counter()-compute_start
            step += 1
            last_loss = loss.item()
            if time.perf_counter() >= next_status:
                write_status(out, state="running", step=step, tokens=stream.stats.tokens_consumed,
                             train_loss=last_loss, learning_rate=lr)
                next_status = time.perf_counter()+1
            validation_every = cfg.get("validation_every", cfg["checkpoint_every"])
            if step % validation_every == 0:
                validation_start = time.perf_counter()
                val_loss = validation_loss(model, batches)
                validation_seconds += time.perf_counter() - validation_start
                write_status(out, validation_loss=val_loss)
                previous_best = float(monitor.get("best_loss", val_loss))
                if policy is None:
                    action, verdict_detail = "bad", None
                else:
                    monitor, action, verdict_detail = validation_verdict(
                        monitor, val_loss, step, lr, policy)
                emit(dict(event="validation", step=step, tokens=stream.stats.tokens_consumed,
                          train_loss=last_loss, validation_loss=val_loss, learning_rate=lr,
                          monitor_action=action))
                if action == "improved":
                    best_path = manager.pin(save(), "best.pt")
                    (out / "best-checkpoint.json").write_text(json.dumps({
                        "checkpoint": str(best_path), "source_step": step,
                        "validation_loss": val_loss, "updated_at": time.time(),
                    }, indent=2), encoding="utf-8")
                elif action == "pause":
                    reason = "regression_pause"
                    pause_detail = verdict_detail
                    write_status(out, state=reason, detail=pause_detail)
                    emit(dict(event=reason, step=step, validation_loss=val_loss,
                              best_validation_loss=previous_best, detail=pause_detail))
                    break
            if step % cfg["checkpoint_every"] == 0:
                save()
            if step % cfg["sample_every"] == 0:
                sample()
    except LowDiskSpace as exc:
        (out / "PAUSED_LOW_DISK").write_text(str(exc))
        emit(dict(event="disk_pause", detail=str(exc), resume_from=str(manager.latest())))
        write_status(out, state="disk_pause", detail=str(exc))
        return
    except KeyboardInterrupt:
        # An interrupt could occur mid-update: use only the last atomic checkpoint.
        emit(dict(event="interrupted", resume_from=str(manager.latest())))
        write_status(out, state="interrupted", detail="Resume uses the last atomic checkpoint; use stop-training.cmd for a complete graceful stop.")
        return
    except (requests.RequestException, StopIteration, ValueError, OSError) as exc:
        reason = f"data_stop: {exc}"
    checkpoint = save()
    if reason == "graceful_stop":
        write_status(out, state="stopped", step=step, tokens=stream.stats.tokens_consumed,
                     train_loss=last_loss, detail="Graceful stop completed; full checkpoint saved.")
        (out / "STOP_REQUESTED").unlink(missing_ok=True)
        emit(dict(event="graceful_stop", step=step, tokens=stream.stats.tokens_consumed, checkpoint=str(checkpoint)))
        return
    validation_start = time.perf_counter()
    final_validation = validation_loss(model, batches)
    final_holdout = validation_loss(model, holdout)
    validation_seconds += time.perf_counter() - validation_start
    sample()
    elapsed = time.perf_counter()-start
    tokens = stream.stats.tokens_consumed-initial_tokens
    summary = dict(reason=reason, step=step, params=model.parameter_count(), new_tokens=tokens,
        total_tokens=stream.stats.tokens_consumed, seconds=elapsed, end_to_end_tokens_per_second=tokens/elapsed,
        compute_tokens_per_second=tokens/compute_seconds if compute_seconds else None,
        compute_seconds=compute_seconds, data_seconds=data_seconds,
        validation_seconds=validation_seconds, checkpoint_seconds=checkpoint_seconds,
        allocated_mib=torch.cuda.max_memory_allocated()/2**20, reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        initial_validation_loss=initial_validation, final_validation_loss=final_validation, final_holdout_loss=final_holdout,
        last_train_loss=last_loss, checkpoint=str(checkpoint), tokenizer_sha256=tok.sha256,
        validation_monitor=monitor, detail=pause_detail)
    if stage == "sft":
        summary["dataset_position"] = {"epoch_index": stream.epoch, "pass_number": stream.epoch + 1,
                                       "epoch_tokens": stream.epoch_tokens,
                                       "byte_offset": stream.byte_offset,
                                       "repeated_data": stream.epoch > 0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    emit(summary)
    write_status(out, state=reason, step=step, tokens=stream.stats.tokens_consumed,
                 train_loss=last_loss, validation_loss=final_validation,
                 detail=pause_detail or reason)
    if reason.startswith("data_stop"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
