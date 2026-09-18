"""Bounded TinyStories experiment. Run artifacts live outside baseline checkpoints."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
from adamlm.checkpoint import CheckpointManager
from adamlm.data import TinyStoriesStream
from adamlm.model import DecoderTransformer, ModelConfig
from adamlm.storage import StorageBudget, LowDiskSpace
from adamlm.training import update, evaluate, generate


def log(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def packed(texts, context, batch=8):
    ids = list(("\n\n".join(texts) + "\n\n").encode("utf-8"))
    count = len(ids) // ((context + 1)*batch)
    tensor = torch.tensor(ids[:count*(context+1)*batch], device="cuda").view(count, batch, context+1)
    return [(part[:, :-1].contiguous(), part[:, 1:].contiguous()) for part in tensor]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2097152)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", default="results/first-run")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600 or args.tokens < 1 or min(args.batch, args.accumulation) < 1:
        parser.error("Positive arguments required; wall-clock cap is 3600 seconds")
    started = time.perf_counter()
    deadline = started + args.seconds
    root = Path.cwd()
    out = root / args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    manager = CheckpointManager(out / "checkpoints", keep=3)
    if manager.latest() and not args.resume:
        raise ValueError("Run already has checkpoints; use --resume or a new run directory")
    cfg = json.loads((root / "config/default.json").read_text())
    guard = StorageBudget(root, cfg["storage"]["project_budget_gb"], cfg["storage"]["cache_limit_gb"], cfg["storage"]["minimum_free_disk_gb"])
    torch.set_num_threads(4)
    random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.reset_peak_memory_stats()
    model = DecoderTransformer(ModelConfig(**cfg["model"])).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None
    reserve = model.parameter_count()*16 + 16*2**20
    guard.check(reserve)
    val_path = out / "validation.json"
    if val_path.exists():
        validation = json.loads(val_path.read_text(encoding="utf-8"))
    else:
        stream = TinyStoriesStream(split="validation", page_size=64, timeout=20)
        texts = stream._fetch_page()
        validation = {"source": "roneneldan/TinyStories", "split": "validation", "offset": 0, "texts": texts}
        val_path.write_text(json.dumps(validation, ensure_ascii=False), encoding="utf-8")
    texts = validation["texts"]
    validation_hash = hashlib.sha256(val_path.read_bytes()).hexdigest()
    val_batches = packed(texts[:32], model.config.block_size)
    test_batches = packed(texts[32:], model.config.block_size)
    train = TinyStoriesStream(page_size=100, timeout=20)
    train.excluded_hashes = {hashlib.sha256(text.strip().encode("utf-8")).hexdigest() for text in texts}
    protocol = dict(model=model.config_dict(), tokenizer="utf8-byte-v1", precision=args.precision,
                    batch=args.batch, accumulation=args.accumulation, validation_sha256=validation_hash,
                    lr=3e-4, seed=1337, dataset=train.state_dict()["source"])
    step, previous_seconds, data_seconds, compute_seconds = 0, 0.0, 0.0, 0.0
    if args.resume:
        latest = manager.latest()
        if latest is None:
            raise ValueError("No checkpoint to resume")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["extra"]["protocol"] != protocol:
            raise ValueError("Resume configuration or validation snapshot changed")
        step = manager.load(latest, model, opt, train)
        previous_seconds = state["extra"]["elapsed_seconds"]
        deadline -= previous_seconds
        data_seconds = state["extra"]["data_seconds"]
        compute_seconds = state["extra"]["compute_seconds"]
        if scaler:
            scaler.load_state_dict(state["extra"]["scaler"])
        del state
    initial_tokens = train.stats.tokens_consumed
    metrics_path = out / "metrics.jsonl"
    initial_val = evaluate(model, val_batches, args.precision)
    log(metrics_path, dict(event="start", step=step, tokens=initial_tokens, validation_loss=initial_val, protocol=protocol))
    losses, last_loss, reason = [], None, "token_target"
    telemetry = []

    def save():
        guard.check(reserve)
        return manager.save(step, model, opt, train.state_dict(), dict(
            protocol=protocol, elapsed_seconds=previous_seconds+time.perf_counter()-started,
            data_seconds=data_seconds, compute_seconds=compute_seconds,
            scaler=scaler.state_dict() if scaler else None))

    model.train()
    next_budget_check = 0.0
    while train.stats.tokens_consumed < args.tokens:
        # Reserve time for final evaluation, serialization, and sampling.
        if time.perf_counter() >= deadline - 60:
            reason = "wall_clock_limit"
            break
        try:
            guard.check_free_disk(reserve)
            if time.perf_counter() >= next_budget_check:
                guard.check(reserve)
                next_budget_check = time.perf_counter() + cfg["storage"]["disk_poll_seconds"]
        except LowDiskSpace as exc:
            (root / "PAUSED_LOW_DISK").write_text(str(exc))
            reason = "disk_pause_resume_from_latest_checkpoint"
            break
        before = train.state_dict()
        before["buffer"] = list(before["buffer"])
        batches = []
        fetch_start = time.perf_counter()
        try:
            for _ in range(args.accumulation):
                x, y = train.next_batch(args.batch, model.config.block_size)
                batches.append((torch.tensor(x, device="cuda"), torch.tensor(y, device="cuda")))
        except Exception as exc:
            train.load_state_dict(before)
            reason = f"data_error: {type(exc).__name__}: {exc}"
            break
        torch.cuda.synchronize()
        data_seconds += time.perf_counter()-fetch_start
        compute_start = time.perf_counter()
        loss = update(model, opt, batches, args.precision, scaler)
        torch.cuda.synchronize()
        compute_seconds += time.perf_counter()-compute_start
        step += 1
        last_loss = loss.item()
        losses.append(last_loss)
        if step % 128 == 0 or train.stats.tokens_consumed >= args.tokens:
            val_loss = evaluate(model, val_batches, args.precision)
            save()
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
            telemetry.append(gpu)
            elapsed = previous_seconds+time.perf_counter()-started
            log(metrics_path, dict(event="progress", step=step, tokens=train.stats.tokens_consumed,
                train_loss=sum(losses)/len(losses), validation_loss=val_loss, elapsed_seconds=elapsed,
                end_to_end_tokens_per_second=train.stats.tokens_consumed/elapsed, gpu=gpu))
            losses.clear()
    checkpoint = manager.latest()
    if not reason.startswith("disk_pause"):
        checkpoint = save()
    final_val = evaluate(model, val_batches, args.precision)
    holdout_loss = evaluate(model, test_batches, args.precision)
    # Reload the saved model: samples prove standalone checkpoint inference.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    del saved
    prompts = ["Once upon a time, a little girl", "The dog found a red ball. He", "Lily was sad because"]
    samples = [generate(model, p, new_tokens=300) for p in prompts]
    elapsed = previous_seconds+time.perf_counter()-started
    byte_count = sum(len(t.encode("utf-8")) for t in texts)
    word_count = sum(len(t.split()) for t in texts)
    result = dict(reason=reason, step=step, tokens=train.stats.tokens_consumed,
        invocation_tokens=train.stats.tokens_consumed-initial_tokens,
        invocation_tokens_per_second=(train.stats.tokens_consumed-initial_tokens)/(time.perf_counter()-started),
        elapsed_seconds=elapsed, invocation_seconds=time.perf_counter()-started,
        end_to_end_tokens_per_second=train.stats.tokens_consumed/elapsed,
        compute_tokens_per_second=train.stats.tokens_consumed/compute_seconds if compute_seconds else None,
        data_seconds=data_seconds, compute_seconds=compute_seconds,
        final_training_batch_loss=last_loss, validation_loss=final_val, holdout_loss=holdout_loss,
        holdout_byte_perplexity=math.exp(holdout_loss), validation_tokens=sum(x.numel() for x,y in val_batches),
        holdout_tokens=sum(x.numel() for x,y in test_batches), allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        reserved_mib=torch.cuda.max_memory_reserved()/2**20, gpu_snapshots=telemetry,
        stories_read=train.stats.stories_read, requests=train.stats.requests, checkpoint=str(checkpoint),
        bytes_per_word=byte_count/word_count, mean_story_bytes=byte_count/len(texts), samples=samples, protocol=protocol)
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
