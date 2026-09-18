"""Bounded, page-based TinyStories access through the Hugging Face viewer API."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import requests
import truststore

from .tokenizer import ByteTokenizer

truststore.inject_into_ssl()


@dataclass
class StreamStats:
    stories_read: int = 0
    tokens_consumed: int = 0
    requests: int = 0
    raw_rows_read: int = 0
    documents_rejected: int = 0


class TinyStoriesStream:
    """Read stories by page; no corpus-sized local download or cache is used."""

    def __init__(self, dataset="roneneldan/TinyStories", config="default", split="train", page_size=8, timeout=30, tokenizer=None):
        self.dataset, self.config, self.split = dataset, config, split
        self.page_size, self.timeout = page_size, timeout
        self.next_offset = 0
        self.buffer: list[int] = []
        self.stats = StreamStats()
        self.tokenizer = tokenizer or ByteTokenizer()
        self.session = requests.Session()
        self.excluded_hashes = set()
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")

    def _fetch_page(self) -> list[str]:
        url = "https://datasets-server.huggingface.co/rows"
        params = {"dataset": self.dataset, "config": self.config, "split": self.split, "offset": self.next_offset, "length": self.page_size}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        rows = response.json().get("rows", [])
        self.stats.requests += 1
        if not rows:
            raise StopIteration("TinyStories stream returned no more rows")
        if any(row.get("truncated_cells") for row in rows):
            raise ValueError("Dataset viewer truncated a story; refusing silently incomplete data")
        if any(row["row_idx"] != self.next_offset + i for i, row in enumerate(rows)):
            raise ValueError("Unexpected dataset row order")
        texts = [row["row"]["text"] for row in rows]
        self.next_offset += len(texts)
        self.stats.stories_read += len(texts)
        return texts

    def _fill(self, minimum_tokens: int) -> None:
        while len(self.buffer) < minimum_tokens:
            for text in self._fetch_page():
                if hashlib.sha256(text.strip().encode("utf-8")).hexdigest() in self.excluded_hashes:
                    continue
                if hasattr(self.tokenizer, "encode_story"):
                    self.buffer.extend(self.tokenizer.encode_story(text))
                else:
                    self.buffer.extend(self.tokenizer.encode(text + "\n\n"))

    def next_batch(self, batch_size: int, block_size: int) -> tuple[list[list[int]], list[list[int]]]:
        needed = batch_size * (block_size + 1)
        self._fill(needed)
        ids = self.buffer[:needed]
        self.buffer = self.buffer[needed:]
        self.stats.tokens_consumed += batch_size * block_size
        x = [ids[i * (block_size + 1): i * (block_size + 1) + block_size] for i in range(batch_size)]
        y = [ids[i * (block_size + 1) + 1: (i + 1) * (block_size + 1)] for i in range(batch_size)]
        return x, y

    def state_dict(self) -> dict:
        return {"tokenizer": getattr(self.tokenizer, "sha256", "utf8-byte-v1"), "source": [self.dataset, self.config, self.split], "next_offset": self.next_offset, "buffer": self.buffer, "stats": self.stats.__dict__.copy()}

    def load_state_dict(self, state: dict) -> None:
        if state.get("tokenizer", "utf8-byte-v1") != getattr(self.tokenizer, "sha256", "utf8-byte-v1"):
            raise ValueError("Checkpoint tokenizer does not match")
        if state.get("source", [self.dataset, self.config, self.split]) != [self.dataset, self.config, self.split]:
            raise ValueError("Checkpoint dataset source does not match")
        self.next_offset = int(state["next_offset"])
        self.buffer = [int(x) for x in state.get("buffer", [])]
        self.stats = StreamStats(**state.get("stats", {}))
