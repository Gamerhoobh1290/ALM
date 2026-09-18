"""Sequential UTF-8 TinyStories text, with exact byte/story/token-buffer positions."""
import json
from pathlib import Path
from .data import TinyStoriesStream
from .downloads import sha256

PARSER = "endoftext-line-utf8-strip-v1"


def read_stories(path, offset=0, limit=100):
    texts = []
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        while len(texts) < limit:
            chunks = []
            size = 0
            while True:
                line = handle.readline(4*1024*1024+1)
                if not line or line.strip() == b"<|endoftext|>":
                    break
                size += len(line)
                if size > 4*1024*1024:
                    raise ValueError("Story exceeds 4 MiB bound or missing story delimiter")
                chunks.append(line)
            text = b"".join(chunks).decode("utf-8").strip()
            if text:
                texts.append(text)
            if not line:
                break
        return texts, handle.tell()


class LocalStoriesStream(TinyStoriesStream):
    def __init__(self, manifest_path, root="data/tinystories-v2", verify=True, **kwargs):
        super().__init__(**kwargs)
        manifest = json.loads(Path(manifest_path).read_text())
        if self.split != "train":
            raise ValueError("Training stream only; validation uses its separate file")
        spec = manifest["train"]
        self.path = Path(root) / spec["name"]
        self.identity = dict(dataset=manifest["dataset"], revision=manifest["revision"],
            filename=spec["name"], sha256=spec["sha256"], parser=PARSER)
        if self.path.stat().st_size != spec["size"] or (verify and sha256(self.path) != spec["sha256"]):
            raise ValueError("Local corpus integrity failure; no automatic redownload or source substitution")
        self.byte_offset = 0
        self.source_start_tokens = 0

    def _fetch_page(self):
        texts, offset = read_stories(self.path,self.byte_offset,self.page_size)
        if not texts:
            raise StopIteration("Local training corpus exhausted; no automatic rewind")
        self.byte_offset = offset
        self.next_offset += len(texts)
        self.stats.stories_read += len(texts)
        return texts

    def state_dict(self):
        return {**super().state_dict(), "local_source":self.identity, "byte_offset":self.byte_offset,
                "source_start_tokens":self.source_start_tokens}

    def load_state_dict(self,state):
        if state.get("local_source") != self.identity:
            raise ValueError("Source changed: requires an explicit recorded continuation, not cursor resume")
        offset = state["byte_offset"]
        if not isinstance(offset,int) or not 0 <= offset <= self.path.stat().st_size:
            raise ValueError("Invalid local byte cursor")
        super().load_state_dict(state)
        self.byte_offset = offset
        self.source_start_tokens = state["source_start_tokens"]
