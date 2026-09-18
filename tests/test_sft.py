import hashlib
import json

import pytest

from adamlm.bpe import BPETokenizer
from adamlm.sft_data import (
    SFTStream,
    encode_example,
    epoch_capacity,
    manifest_epoch_tokens,
)


TOKENIZER = "tokenizers/bpe4096/tokenizer.json"


def make_corpus(tmp_path, rows=24, name="corpus"):
    """A tiny SFT corpus whose every row is individually identifiable."""
    tokenizer = BPETokenizer(TOKENIZER)
    data = [{"instruction": f"State record {index}.", "context": "",
             "response": f"Record {index} of the synthetic corpus."} for index in range(rows)]
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
    tokens = sum(len(encode_example(row, tokenizer, 32)[0]) for row in data)
    spec = {"path": str(path), "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "examples": len(data)}
    manifest = tmp_path / f"{name}-manifest.json"
    manifest.write_text(json.dumps({
        "format": "adamlm-response-sft-v1", "dataset": name, "revision": "1",
        "sources": {name: {"rows": len(data), "tokens": tokens}},
        "files": {split: spec for split in ("train", "validation", "holdout")},
    }), encoding="utf-8")
    return manifest


def drain(stream, batch_size=2, block_size=32):
    """Consume a whole pass; return the batches it actually supplied."""
    batches = []
    while True:
        try:
            batches.append(stream.next_batch(batch_size, block_size))
        except StopIteration:
            return batches


def make_manifest(tmp_path):
    rows = [
        {"instruction": "Reply with one word.", "context": "", "response": "blue", "category": "test"},
        {"instruction": "Add the numbers.", "context": "Two and three.", "response": "Five.", "category": "test"},
    ] * 20
    files = {}
    for split in ("train", "validation", "holdout"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        files[split] = {"path": str(path), "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "examples": len(rows)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": "adamlm-response-sft-v1", "dataset": "toy", "revision": "1", "files": files}))
    return manifest


def test_response_mask_keeps_only_assistant_targets():
    tokenizer = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    row = {"instruction": "Repeat the color.", "context": "The color is blue.", "response": "blue"}
    inputs, targets = encode_example(row, tokenizer, 64)
    first_target = next(index for index, value in enumerate(targets) if value != -100)
    assert all(value == -100 for value in targets[:first_target])
    assert tokenizer.decode([value for value in targets[first_target:] if value >= 0]).startswith("blue")
    assert len(inputs) == len(targets) <= 64


def test_sft_stream_resume_replays_exact_batch(tmp_path):
    tokenizer = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    manifest = make_manifest(tmp_path)
    stream = SFTStream(manifest, tokenizer)
    stream.next_batch(1, 32)
    state = stream.state_dict()
    expected = stream.next_batch(2, 32)
    restored = SFTStream(manifest, tokenizer)
    restored.load_state_dict(state)
    assert restored.next_batch(2, 32) == expected


def test_five_fresh_passes_have_identical_batchable_capacity(tmp_path):
    """A freshly constructed stream is always pass 1: same capacity, same start.

    Advancing between passes is the caller's explicit decision (see
    ``continue_from``); construction alone never carries a position.
    """
    tokenizer = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    manifest = make_manifest(tmp_path)
    per_pass = []
    first_batches = []
    for _ in range(5):
        stream = SFTStream(manifest, tokenizer)
        consumed = 0
        first = None
        while True:
            try:
                batch = stream.next_batch(2, 32)
                first = first or batch
                consumed += 64
            except StopIteration:
                break
        per_pass.append(consumed)
        first_batches.append(first)
    assert len(set(per_pass)) == 1 and per_pass[0] > 0
    assert first_batches[4] == first_batches[0]


# --------------------------------------------------------------------------
# Stage boundaries: interrupted resume, continuation, approved repeat.
# --------------------------------------------------------------------------

def test_old_failure_reproduced_then_fixed_by_continuing_the_parent_cursor(tmp_path):
    """The regression itself: a new stage used to restart at row 1.

    Building a stream from the parent's *weights* while leaving its cursor
    behind replays the rows the parent just trained on. Carrying the cursor
    forward is what makes the next stage see new data.
    """
    tokenizer = BPETokenizer(TOKENIZER)
    manifest = make_corpus(tmp_path)
    parent = SFTStream(manifest, tokenizer)
    first_batch = parent.next_batch(2, 32)
    parent_cursor = parent.state_dict()

    replayed = SFTStream(manifest, tokenizer)  # what the old stage init did
    assert replayed.next_batch(2, 32) == first_batch

    advanced = SFTStream(manifest, tokenizer)
    info = advanced.continue_from(parent_cursor)
    assert info["mode"] == "continue_epoch" and info["repeated_data"] is False
    assert advanced.next_batch(2, 32) != first_batch


def test_chained_stages_reproduce_one_uninterrupted_pass_exactly(tmp_path):
    """Three chained stages must equal one continuous pass: no gap, no overlap."""
    tokenizer = BPETokenizer(TOKENIZER)
    manifest = make_corpus(tmp_path)
    solo = SFTStream(manifest, tokenizer)
    expected = [solo.next_batch(2, 32) for _ in range(6)]

    chained, cursor = [], None
    for _ in range(3):
        stage = SFTStream(manifest, tokenizer)
        if cursor is not None:
            stage.continue_from(cursor)
        chained += [stage.next_batch(2, 32) for _ in range(2)]
        cursor = stage.state_dict()
    assert chained == expected
    assert cursor["epoch"] == 0 and cursor["epoch_tokens"] == 6 * 2 * 32


def test_interrupted_stage_resumes_its_exact_cursor_and_position(tmp_path):
    """Case 1: an interrupted stage resumes exactly, and keeps its corpus position."""
    tokenizer = BPETokenizer(TOKENIZER)
    manifest = make_corpus(tmp_path)
    stream = SFTStream(manifest, tokenizer)
    stream.next_batch(2, 32)
    saved = stream.state_dict()
    expected = [stream.next_batch(2, 32) for _ in range(2)]

    restored = SFTStream(manifest, tokenizer)
    restored.load_state_dict(saved)
    assert restored.epoch == saved["epoch"]
    assert restored.epoch_tokens == saved["epoch_tokens"]
    assert [restored.next_batch(2, 32) for _ in range(2)] == expected
    assert restored.state_dict()["epoch_tokens"] == stream.state_dict()["epoch_tokens"]


def test_pass_four_to_pass_five_needs_approval_and_is_recorded_as_repeated(tmp_path):
    """Case 3: crossing a pass boundary is explicit, labeled, and counted."""
    tokenizer = BPETokenizer(TOKENIZER)
    manifest = make_corpus(tmp_path)
    cursor, opening_batches = None, []
    for index in range(5):
        stage = SFTStream(manifest, tokenizer)
        info = stage.continue_from(cursor, repeat_epoch=index > 0)
        assert info["epoch_index"] == index
        assert info["repeated_data"] is (index > 0)
        assert info["start_byte_offset"] == 0
        batches = drain(stage)
        opening_batches.append(batches[0])
        cursor = stage.state_dict()
    # Pass 5 exists as its own pass, and re-reads the corpus from the start.
    assert cursor["epoch"] == 4
    assert opening_batches[4] == opening_batches[0]


def test_exhausted_pass_does_not_rewind_without_approval(tmp_path):
    """Case 2 at the end of the data: continuing an exhausted pass supplies nothing."""
    tokenizer = BPETokenizer(TOKENIZER)
    manifest = make_corpus(tmp_path)
    first = SFTStream(manifest, tokenizer)
    drain(first)
    cursor = first.state_dict()

    following = SFTStream(manifest, tokenizer)
    following.continue_from(cursor)
    assert following.epoch == 0
    with pytest.raises(StopIteration):
        following.next_batch(2, 32)

    repeating = SFTStream(manifest, tokenizer)
    repeating.continue_from(cursor, repeat_epoch=True)
    assert repeating.epoch == 1
    assert repeating.next_batch(2, 32)  # the approved repeat does supply data


def test_continue_from_rejects_a_foreign_or_absent_cursor(tmp_path):
    tokenizer = BPETokenizer(TOKENIZER)
    mine = make_corpus(tmp_path, name="mine")
    theirs = make_corpus(tmp_path, rows=12, name="theirs")
    other = SFTStream(theirs, tokenizer)
    other.next_batch(2, 32)

    stream = SFTStream(mine, tokenizer)
    with pytest.raises(ValueError, match="changed between the parent"):
        stream.continue_from(other.state_dict())

    # A pretrain parent carries no SFT cursor: that is a first pass, and it
    # cannot be a repeat of something that never happened.
    fresh = SFTStream(mine, tokenizer)
    assert fresh.continue_from({"local_source": "pretrain"})["mode"] == "fresh_pass"
    with pytest.raises(ValueError, match="never consumed"):
        SFTStream(mine, tokenizer).continue_from(None, repeat_epoch=True)


def test_epoch_capacity_rounds_batches_and_discards_the_sub_batch_tail():
    capacity = epoch_capacity(8_291_299, 4096)
    assert capacity["batchable_tokens"] == 8_290_304
    assert capacity["discarded_tail_tokens"] == 995
    assert capacity["remaining_tokens"] == 8_290_304

    part = epoch_capacity(8_291_299, 4096, consumed=4_096_000)
    assert part["remaining_tokens"] == 8_290_304 - 4_096_000
    # A partly-used pass still reports whole batches only.
    assert epoch_capacity(8_291_299, 4096, consumed=8_290_304)["remaining_tokens"] == 0
    with pytest.raises(ValueError, match="past the end of the dataset"):
        epoch_capacity(8_291_299, 4096, consumed=8_290_305)


def test_manifest_epoch_tokens_reads_the_measured_total(tmp_path):
    manifest = make_corpus(tmp_path)
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_epoch_tokens(manifest) == recorded["sources"]["corpus"]["tokens"]
