import json
from pathlib import Path

import torch
import pytest

from adamlm.checkpoint import CheckpointManager
from adamlm.model import DecoderTransformer, ModelConfig
from adamlm.storage import LowDiskSpace, StorageBudget
from adamlm.tokenizer import ByteTokenizer


def test_byte_tokenizer_is_lossless_for_ascii():
    tokenizer = ByteTokenizer()
    text = "hello AdamLM"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_model_shapes_and_loss():
    model = DecoderTransformer(ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=32))
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)


def test_checkpoint_round_trip(tmp_path):
    model = DecoderTransformer(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path / "checkpoints", keep=2)
    path = manager.save(1, model, optimizer, {"next_offset": 4, "buffer": []})
    restored = DecoderTransformer(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32))
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    class Stream:
        def load_state_dict(self, state):
            self.state = state
    stream = Stream()
    assert CheckpointManager.load(path, restored, restored_optimizer, stream) == 1
    assert stream.state["next_offset"] == 4
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))


def test_storage_budget_snapshot(tmp_path):
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "x").write_bytes(b"123")
    state = StorageBudget(tmp_path, budget_gb=1, cache_limit_gb=1, minimum_free_gb=0).snapshot()
    assert state["project_bytes"] >= 3
    assert state["cache_bytes"] == 3


def test_storage_guard_detects_cache_limit(tmp_path):
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "x").write_bytes(b"123")
    with pytest.raises(LowDiskSpace):
        StorageBudget(tmp_path, budget_gb=1, cache_limit_gb=0, minimum_free_gb=0).check()
