"""Explicit dataset selection and separated validation material."""
import json
from pathlib import Path
from .local_data import LocalStoriesStream
from .parquet_data import ParquetDocumentStream
from .data import TinyStoriesStream


class MixtureStream(TinyStoriesStream):
    def __init__(self, streams, weights):
        if len(streams) != len(weights) or not streams or any(w <= 0 for w in weights):
            raise ValueError("Mixture needs equally sized positive weights")
        super().__init__(tokenizer=streams[0].tokenizer)
        self.streams = streams
        total = sum(weights)
        normalized = [weight/total for weight in weights]
        scale = 1000
        credit = [0.0] * len(normalized)
        self.schedule = []
        for _ in range(scale):
            credit = [value + weight for value, weight in zip(credit, normalized)]
            index = max(range(len(credit)), key=credit.__getitem__)
            credit[index] -= 1.0
            self.schedule.append(index)
        self.schedule_index = 0
        self.identity = dict(kind="token_batch_mixture_v2", components=[getattr(s, "dataset_name", "tinystories") for s in streams], weights=weights)

    def _fetch_page(self):
        index = self.schedule[self.schedule_index % len(self.schedule)]
        self.schedule_index += 1
        texts = self.streams[index]._fetch_page()
        self.stats.stories_read += len(texts)
        return texts

    def next_batch(self, batch_size, block_size):
        """Mix equal-sized token batches so configured weights are token weights."""
        index = self.schedule[self.schedule_index % len(self.schedule)]
        self.schedule_index += 1
        x, y = self.streams[index].next_batch(batch_size, block_size)
        self.stats.tokens_consumed += batch_size * block_size
        self.stats.stories_read = sum(stream.stats.stories_read for stream in self.streams)
        self.stats.raw_rows_read = sum(stream.stats.raw_rows_read for stream in self.streams)
        self.stats.documents_rejected = sum(stream.stats.documents_rejected for stream in self.streams)
        return x, y

    def state_dict(self):
        return {"tokenizer": getattr(self.tokenizer, "sha256", "utf8-byte-v1"), "mixture": self.identity,
                "schedule_index": self.schedule_index, "buffer": self.buffer,
                "stats": self.stats.__dict__.copy(), "components":[s.state_dict() for s in self.streams]}

    def load_state_dict(self, state):
        if state.get("mixture") != self.identity or state.get("tokenizer") != getattr(self.tokenizer, "sha256", "utf8-byte-v1"):
            raise ValueError("Mixture configuration or tokenizer changed")
        if len(state.get("components", [])) != len(self.streams):
            raise ValueError("Mixture component count changed")
        for stream, saved in zip(self.streams, state["components"]):
            stream.load_state_dict(saved)
        self.schedule_index = int(state["schedule_index"])
        self.buffer = [int(x) for x in state.get("buffer", [])]
        from .data import StreamStats
        self.stats = StreamStats(**state.get("stats", {}))


def load_registry(path="config/extra-datasets.json"):
    return json.loads(Path(path).read_text())


def build_stream(name, tokenizer, guard, config=None):
    if name == "tinystories":
        cfg = dict(config or {})
        return LocalStoriesStream("config/tinystories-v2.json", tokenizer=tokenizer, **cfg)
    registry = load_registry()
    if name == "mixture":
        specs = (config or {}).get("components", [])
        if not specs:
            raise ValueError("Mixture selection requires explicit components")
        streams = [build_stream(spec["dataset"], tokenizer, guard, config=spec.get("data")) for spec in specs]
        return MixtureStream(streams, [float(spec["weight"]) for spec in specs])
    if name not in registry:
        raise ValueError(f"Dataset unavailable: {name}")
    spec = registry[name]
    row_end = None
    if spec.get("validation_from_tail"):
        row_end = sum(x["rows"] for x in spec["train"]) - spec["validation_rows"]
    return ParquetDocumentStream("config/extra-datasets.json", name, tokenizer=tokenizer, row_end=row_end, **(config or {}))


def load_validation_texts(name, limit=None, config=None):
    if name == "tinystories":
        texts = json.loads(Path("data/tinystories-v2/validation-snapshot.json").read_text(encoding="utf-8"))["texts"]
    elif name == "mixture":
        groups = []
        for component in (config or {}).get("components", []):
            groups.append(load_validation_texts(component["dataset"], limit=64, config=component.get("data")))
        texts = []
        for index in range(max((len(group) for group in groups), default=0)):
            texts.extend(group[index] for group in groups if index < len(group))
    else:
        spec = load_registry()[name]
        if spec.get("validation_from_tail"):
            total = sum(x["rows"] for x in spec["train"])
            stream = ParquetDocumentStream("config/extra-datasets.json", name, split="train", row_start=total-spec["validation_rows"], tokenizer=__import__("adamlm.bpe",fromlist=["BPETokenizer"]).BPETokenizer("tokenizers/bpe4096/tokenizer.json"))
        else:
            stream = ParquetDocumentStream("config/extra-datasets.json", name, split="validation", tokenizer=__import__("adamlm.bpe",fromlist=["BPETokenizer"]).BPETokenizer("tokenizers/bpe4096/tokenizer.json"))
        texts = []
        while True:
            try: texts.extend(stream._fetch_page())
            except StopIteration: break
        
    return texts[:limit] if limit else texts
