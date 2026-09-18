import copy
from pathlib import Path
import pytest
import torch
from adamlm.bpe import BPETokenizer, BOUNDARY
from adamlm.bpe_train import learning_rate, validation_batches
from adamlm.data import TinyStoriesStream
from adamlm.model import DecoderTransformer, ModelConfig
from adamlm.training import generate


@pytest.fixture
def tokenizer():
    return BPETokenizer(Path(__file__).parents[1] / "tokenizers/bpe4096/tokenizer.json")


def test_byte_fallback_and_special_token_roundtrip(tokenizer):
    from tokenizers.pre_tokenizers import ByteLevel
    assert all(tokenizer.backend.token_to_id(symbol) is not None for symbol in ByteLevel.alphabet())
    texts = ["", "hello\n world", "مرحبا 世界 🧬\u0000", "".join(chr(n) for n in range(256)),
             "literal " + BOUNDARY + " stays text", "𐍈🦄\u200d"]
    for text in texts:
        ids = tokenizer.encode(text)
        assert tokenizer.boundary_id not in ids
        assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_size == 4096
    assert tokenizer.encode_story("Hi")[-1] == tokenizer.boundary_id


def test_stream_uses_boundary_and_rejects_byte_checkpoint(tokenizer):
    stream = TinyStoriesStream(tokenizer=tokenizer)
    stream._fetch_page = lambda: ["One story", "Another story"]
    stream._fill(1)
    assert stream.buffer == tokenizer.encode_story("One story") + tokenizer.encode_story("Another story")
    state = copy.deepcopy(stream.state_dict())
    other = TinyStoriesStream(tokenizer=tokenizer)
    other.load_state_dict(state)
    assert stream.next_batch(1, 2) == other.next_batch(1, 2)
    with pytest.raises(ValueError, match="tokenizer"):
        stream.load_state_dict(TinyStoriesStream().state_dict())
    with pytest.raises(ValueError, match="tokenizer"):
        TinyStoriesStream().load_state_dict(state)


def test_warmup_decay_is_resumable():
    assert learning_rate(1, 100, 10, 1.0, 0.1) == 0.1
    assert learning_rate(10, 100, 10, 1.0, 0.1) == 1.0
    assert learning_rate(100, 100, 10, 1.0, 0.1) == 0.1
    original = [learning_rate(s, 100, 10, 1.0, 0.1) for s in range(1, 101)]
    resumed = [learning_rate(s, 100, 10, 1.0, 0.1) for s in range(51, 101)]
    assert original[50:] == resumed


def test_validation_keeps_all_targets(tokenizer):
    texts = ["A story. "*100, "A cat."]
    batches = validation_batches(texts, tokenizer, 512, "cpu")
    assert sum(int((y != -100).sum()) for x, y in batches) == sum(len(tokenizer.encode_story(t))-1 for t in texts)
    assert all(x.shape == y.shape == (1,512) for x,y in batches)


def test_bpe_generation_stops_at_boundary(tokenizer):
    class BoundaryModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
            self.config = ModelConfig(vocab_size=4096, block_size=512)
        def forward(self, ids):
            logits = torch.full((*ids.shape,4096), -100.0)
            logits[:,:,tokenizer.boundary_id] = 100.0
            return logits, None
    assert generate(BoundaryModel(), "Hello", tokenizer=tokenizer) == "Hello"
