"""UTF-8-safe inference: sampling must never emit undecodable bytes.

Covers the reported glitch where replies contained U+FFFD (e.g. around
apostrophes): multi-byte characters fall back to single-byte BPE tokens,
so unconstrained sampling or a tight token budget could produce invalid
UTF-8, which the decoder renders as the replacement character.
"""
import codecs
import random

import torch

from adamlm.bpe import BPETokenizer
from adamlm.model import ModelConfig
from adamlm.training import (
    generate,
    incomplete_utf8_tail,
    token_byte_table,
)


def _bpe_tokenizer():
    from pathlib import Path
    return BPETokenizer(Path(__file__).parents[1] / "tokenizers/bpe4096/tokenizer.json")


def test_byte_table_matches_encoding():
    tokenizer = _bpe_tokenizer()
    table = token_byte_table(tokenizer)
    assert len(table) == tokenizer.vocab_size == 4096
    assert table[159] + table[223] + table[248] == "’".encode("utf-8")
    for text in ["Hello!", "It’s great", "café “hi”", "User:\nHi\n\nAssistant:\n"]:
        ids = tokenizer.encode(text)
        assert b"".join(table[i] for i in ids) == text.encode("utf-8")


def test_incomplete_tail_matches_incremental_decoder():
    random.seed(0)
    alphabet = "héllo’“” café\nUser:,.!?"
    for _ in range(300):
        text = "".join(random.choice(alphabet) for _ in range(random.randint(0, 12)))
        raw = text.encode("utf-8")
        cut = random.randint(0, len(raw))
        prefix = raw[:cut]
        expected_tail = codecs.getincrementaldecoder("utf-8")()
        expected_tail.decode(prefix, False)
        leftover = expected_tail.getstate()[0]
        assert incomplete_utf8_tail(prefix) == bytes(leftover or b""), (text, cut)


def _logit_model(vocab_size, block_size, forced_id):
    class Forced(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
            self.config = ModelConfig(vocab_size=vocab_size, block_size=block_size)

        def forward(self, ids):
            logits = torch.full((*ids.shape, vocab_size), -100.0)
            logits[:, :, forced_id] = 100.0
            return logits, None

    return Forced()


def test_forced_lone_continuation_byte_never_decodes():
    tokenizer = _bpe_tokenizer()
    lone_continuation = tokenizer.encode("’")[1]  # middle byte 0x80 of U+2019
    assert token_byte_table(tokenizer)[lone_continuation] == b"\x80"
    text = generate(_logit_model(4096, 512, lone_continuation), "Hi",
                    new_tokens=8, tokenizer=tokenizer)
    assert "�" not in text and text.startswith("Hi")


def test_truncated_multibyte_tail_is_dropped_not_replaced():
    tokenizer = _bpe_tokenizer()
    first_byte = tokenizer.encode("’")[0]  # 0xE2 without its continuation bytes
    text = generate(_logit_model(4096, 512, first_byte), "Hi",
                    new_tokens=3, tokenizer=tokenizer)
    # The model can only emit 0xE2 bytes; the trailing partial character is
    # dropped instead of surfacing as U+FFFD.
    assert "�" not in text and text.startswith("Hi")
    text.encode("utf-8")


def test_boundary_stop_returns_prompt_verbatim():
    tokenizer = _bpe_tokenizer()
    text = generate(_logit_model(4096, 512, tokenizer.boundary_id), "Hello",
                    tokenizer=tokenizer)
    assert text == "Hello"
