"""Sequential response-masked instruction data for supervised fine-tuning."""
import json
from pathlib import Path

from .data import StreamStats
from .downloads import sha256


def encode_example(row, tokenizer, block_size):
    instruction = row["instruction"].strip()
    context = row.get("context", "").strip()
    response = row["response"].strip()
    prompt = f"User:\n{instruction}\n"
    if context:
        prompt += f"\nContext:\n{context}\n"
    prompt += "\nAssistant:\n"
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(response) + [tokenizer.boundary_id]
    maximum = block_size + 1
    if len(prompt_ids) + len(answer_ids) > maximum:
        answer_budget = min(len(answer_ids), max(32, maximum // 2))
        prompt_budget = maximum - answer_budget
        marker = tokenizer.encode("\nAssistant:\n")
        if prompt_budget <= len(marker):
            raise ValueError("Context is too small for the assistant marker")
        prompt_ids = prompt_ids[:prompt_budget-len(marker)] + marker
        if len(answer_ids) > answer_budget:
            answer_ids = answer_ids[:answer_budget-1] + [tokenizer.boundary_id]
    sequence = prompt_ids + answer_ids
    inputs = sequence[:-1]
    targets = sequence[1:]
    targets[:max(0, len(prompt_ids)-1)] = [-100] * max(0, len(prompt_ids)-1)
    if not any(target >= 0 for target in targets):
        raise ValueError("SFT example has no response target")
    return inputs, targets


class SFTStream:
    def __init__(self, manifest_path, tokenizer, split="train", page_size=64, verify=True):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "adamlm-response-sft-v1" or split not in manifest.get("files", {}):
            raise ValueError("Unsupported SFT manifest or split")
        spec = manifest["files"][split]
        self.path = Path(spec["path"])
        if not self.path.is_file() or self.path.stat().st_size != spec["size"] or (verify and sha256(self.path) != spec["sha256"]):
            raise ValueError(f"SFT {split} integrity failure")
        self.identity = {"format": manifest["format"], "dataset": manifest["dataset"], "revision": manifest["revision"],
                         "split": split, "sha256": spec["sha256"], "tokenizer": tokenizer.sha256}
        self.tokenizer = tokenizer
        self.page_size = page_size
        self.byte_offset = 0
        self.input_buffer = []
        self.target_buffer = []
        self.stats = StreamStats()
        self.excluded_hashes = set()
        # Position within the corpus as a whole, carried across stage
        # boundaries. ``epoch`` is the 0-based pass currently being consumed;
        # ``epoch_tokens`` counts tokens trained within that pass by every
        # stage so far, not just this one. ``stats`` stays per-stage.
        self.epoch = 0
        self.epoch_tokens = 0

    def _fill(self, needed, block_size):
        with self.path.open("rb") as handle:
            handle.seek(self.byte_offset)
            while len(self.input_buffer) < needed:
                line = handle.readline(4 * 1024 * 1024 + 1)
                if not line:
                    self.byte_offset = handle.tell()
                    raise StopIteration(f"SFT {self.identity['split']} exhausted")
                if len(line) > 4 * 1024 * 1024:
                    raise ValueError("SFT row exceeds 4 MiB")
                self.byte_offset = handle.tell()
                try:
                    row = json.loads(line.decode("utf-8"))
                    inputs, targets = encode_example(row, self.tokenizer, block_size)
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    self.stats.documents_rejected += 1
                    continue
                self.input_buffer.extend(inputs)
                self.target_buffer.extend(targets)
                self.stats.stories_read += 1

    def next_batch(self, batch_size, block_size):
        needed = batch_size * block_size
        self._fill(needed, block_size)
        inputs, targets = self.input_buffer[:needed], self.target_buffer[:needed]
        self.input_buffer, self.target_buffer = self.input_buffer[needed:], self.target_buffer[needed:]
        self.stats.tokens_consumed += needed
        self.epoch_tokens += needed
        return ([inputs[i:i+block_size] for i in range(0, needed, block_size)],
                [targets[i:i+block_size] for i in range(0, needed, block_size)])

    def state_dict(self):
        # Copies, not references: a captured cursor must describe the moment
        # it was taken even if this stream keeps consuming afterwards.
        return {"sft_source": self.identity, "byte_offset": self.byte_offset,
                "buffer": list(self.input_buffer), "target_buffer": list(self.target_buffer),
                "stats": self.stats.__dict__.copy(),
                "epoch": self.epoch, "epoch_tokens": self.epoch_tokens}

    def load_state_dict(self, state):
        if state.get("sft_source") != self.identity:
            raise ValueError("SFT source, split, tokenizer, or revision changed")
        offset = state.get("byte_offset")
        if not isinstance(offset, int) or not 0 <= offset <= self.path.stat().st_size:
            raise ValueError("Invalid SFT byte cursor")
        inputs = state.get("buffer", [])
        targets = state.get("target_buffer", [])
        if len(inputs) != len(targets):
            raise ValueError("Mismatched SFT token and target buffers")
        self.byte_offset = offset
        self.input_buffer = [int(value) for value in inputs]
        self.target_buffer = [int(value) for value in targets]
        self.stats = StreamStats(**state.get("stats", {}))
        # Checkpoints written before the corpus position was tracked carry
        # neither key. Those runs only ever consumed a first pass, so the
        # stage's own token count is exactly its position within epoch 0.
        self.epoch = int(state.get("epoch", 0))
        self.epoch_tokens = int(state.get("epoch_tokens", self.stats.tokens_consumed))

    def continue_from(self, cursor, *, repeat_epoch=False):
        """Position a NEW stage against the data a parent stage already used.

        This is the only supported way to start a stage that is not an exact
        resume, and it makes the choice explicit rather than implicit:

        - No parent cursor for this corpus (a pretrain parent, or the first
          SFT stage ever): begin pass 1 at the start of the file.
        - ``repeat_epoch=False``: continue at the parent's byte offset,
          carrying its partial token buffer, so no row is replayed and none
          is skipped.
        - ``repeat_epoch=True``: an explicitly approved repeat. Rewind to the
          start of the file and advance the epoch counter, so the repetition
          is recorded as repeated data instead of being counted as new.

        Returns a description of the transition for the run's provenance
        record. Raises when the parent consumed a different corpus.
        """
        source = (cursor or {}).get("sft_source")
        if source is None:
            if repeat_epoch:
                raise ValueError(
                    "Cannot repeat an epoch: this parent never consumed "
                    f"{self.identity['dataset']}/{self.identity['split']}, so there is "
                    "no completed pass to repeat. Start a first pass instead.")
            self.byte_offset, self.input_buffer, self.target_buffer = 0, [], []
            self.epoch, self.epoch_tokens = 0, 0
            self.stats = StreamStats()
            return {"mode": "fresh_pass", "epoch_index": 0, "repeated_data": False,
                    "start_byte_offset": 0, "epoch_tokens_before": 0,
                    "detail": "first pass over this corpus; cursor starts at the beginning"}
        if source != self.identity:
            raise ValueError(
                "SFT source, split, tokenizer, or revision changed between the parent "
                "checkpoint and this stage; refusing to reinterpret its cursor")
        parent_offset = cursor.get("byte_offset")
        size = self.path.stat().st_size
        if not isinstance(parent_offset, int) or not 0 <= parent_offset <= size:
            raise ValueError("Invalid SFT byte cursor on the parent checkpoint")
        inputs = cursor.get("buffer", [])
        targets = cursor.get("target_buffer", [])
        if len(inputs) != len(targets):
            raise ValueError("Mismatched SFT token and target buffers on the parent checkpoint")
        parent_stats = StreamStats(**cursor.get("stats", {}))
        parent_epoch = int(cursor.get("epoch", 0))
        parent_epoch_tokens = int(cursor.get("epoch_tokens", parent_stats.tokens_consumed))
        self.stats = StreamStats()
        if repeat_epoch:
            self.byte_offset, self.input_buffer, self.target_buffer = 0, [], []
            self.epoch, self.epoch_tokens = parent_epoch + 1, 0
            return {"mode": "repeat_epoch", "epoch_index": self.epoch, "repeated_data": True,
                    "start_byte_offset": 0, "epoch_tokens_before": 0,
                    "detail": (f"explicitly approved repeat: pass {self.epoch + 1} over rows already "
                               f"trained in {self.epoch} earlier pass(es); these tokens are not new data")}
        self.byte_offset = parent_offset
        self.input_buffer = [int(value) for value in inputs]
        self.target_buffer = [int(value) for value in targets]
        self.epoch, self.epoch_tokens = parent_epoch, parent_epoch_tokens
        return {"mode": "continue_epoch", "epoch_index": self.epoch,
                "repeated_data": self.epoch > 0, "start_byte_offset": self.byte_offset,
                "epoch_tokens_before": self.epoch_tokens,
                "detail": (f"continuing pass {self.epoch + 1} at byte {self.byte_offset} of {size} "
                           f"after {self.epoch_tokens} tokens already trained in this pass")}


def epoch_capacity(epoch_tokens, tokens_per_step, consumed=0):
    """Batchable capacity of one pass, and what is left of it.

    The planner and the trainer both call this so they round batches and
    discard the sub-batch file tail identically. A pass supplies only whole
    optimizer batches; the remainder below one batch can never be trained.
    """
    epoch_tokens, tokens_per_step, consumed = int(epoch_tokens), int(tokens_per_step), int(consumed)
    if tokens_per_step <= 0:
        raise ValueError("Training batch size must be positive")
    usable = (epoch_tokens // tokens_per_step) * tokens_per_step
    if consumed < 0 or consumed > usable:
        raise ValueError(
            f"Cursor reports {consumed} tokens consumed in a pass that supplies {usable} "
            "batchable tokens; refusing to plan past the end of the dataset")
    remaining = ((usable - consumed) // tokens_per_step) * tokens_per_step
    return {"source_tokens": epoch_tokens, "batchable_tokens": usable,
            "discarded_tail_tokens": epoch_tokens - usable,
            "consumed_tokens": consumed, "remaining_tokens": remaining}


def manifest_epoch_tokens(manifest_path):
    """Measured tokens in one pass over a corpus, or None when unrecorded.

    Read from the manifest's per-source measurements so the planner and the
    trainer size a pass from the same number rather than each estimating.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        return None
    total = 0
    for spec in sources.values():
        try:
            total += int(spec["tokens"])
        except (KeyError, TypeError, ValueError):
            return None
    return total or None


def read_dataset_cursor(checkpoint_path):
    """The dataset cursor a checkpoint carries, or None when it has none.

    The checkpoint is the source of truth for run position, so planner and
    trainer both read the position from here rather than from any mirror.
    """
    import torch

    state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    cursor = state.get("dataset")
    return cursor if isinstance(cursor, dict) else None
