"""Frozen byte-level BPE with all 256 byte symbols and explicit story termination."""
import hashlib
from pathlib import Path
from tokenizers import Tokenizer

BOUNDARY = "<|endofstory|>"


class BPETokenizer:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.backend = Tokenizer.from_file(str(self.path))
        # Literal special-token strings in user text remain ordinary text.
        self.backend.encode_special_tokens = True
        self.vocab_size = self.backend.get_vocab_size()
        self.boundary_id = self.backend.token_to_id(BOUNDARY)
        if self.vocab_size != 4096 or self.boundary_id is None:
            raise ValueError("Expected 4096-token AdamLM BPE with story boundary")

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def encode_story(self, text):
        return self.encode(text) + [self.boundary_id]

    def decode(self, ids):
        return self.backend.decode(ids, skip_special_tokens=True)
