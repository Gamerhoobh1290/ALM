"""Deterministic Parquet streams with source-aware document assembly."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import pyarrow.parquet as pq
from .data import TinyStoriesStream
from .downloads import sha256

PARSER = "parquet-document-v2"
ARTICLE_TITLE = re.compile(r"^= [^=].* =$", re.DOTALL)


def clean_document(value, quality=None):
    """Return normalized training text, or None for clearly unusable content."""
    if not isinstance(value, str):
        return None
    quality = quality or {}
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or "\x00" in text:
        return None
    controls = sum(unicodedata.category(char) == "Cc" and char not in "\n\t" for char in text)
    if controls / len(text) > quality.get("max_control_ratio", 0.001):
        return None
    if text.count("\ufffd") / len(text) > quality.get("max_replacement_ratio", 0.001):
        return None
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(words) < quality.get("min_words", 1):
        return None
    visible = sum(not char.isspace() for char in text)
    letters = sum(char.isalpha() for char in text)
    if visible and letters / visible < quality.get("min_letter_ratio", 0.2):
        return None
    lowered = [word.casefold() for word in words]
    if len(lowered) >= 100 and len(set(lowered)) / len(lowered) < quality.get("min_unique_word_ratio", 0.08):
        return None
    lines = [re.sub(r"\s+", " ", line).strip().casefold() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4 and 1 - len(set(lines)) / len(lines) > quality.get("max_duplicate_line_ratio", 0.5):
        return None
    return text


class ParquetDocumentStream(TinyStoriesStream):
    def __init__(self, manifest_path, dataset_name, split="train", row_start=0, row_end=None, verify=True, **kwargs):
        super().__init__(split=split, **kwargs)
        raw = Path(manifest_path).read_bytes()
        self.manifest_hash = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)[dataset_name]
        self.dataset_name = dataset_name
        self.files = self.manifest[split]
        self.text_column = self.manifest["text_column"]
        self.boundary = self.manifest.get("boundary", "row_document")
        if self.boundary not in ("row_document", "wikitext_article"):
            raise ValueError(f"Unsupported document boundary mode: {self.boundary}")
        self.quality = self.manifest.get("quality", {})
        self.row_start = row_start
        self.row_cursor = row_start
        self.row_end = row_end if row_end is not None else sum(x["rows"] for x in self.files)
        self._parquet = {}
        self._cached_group = None
        self._cached_values = None
        self._verify = verify
        self._verified = set()
        self._article_parts = []
        self._row_backlog = []
        self._locate = []
        total = 0
        for index, spec in enumerate(self.files):
            path = Path(spec["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != spec["size"]:
                raise ValueError(f"Size mismatch: {path}")
            if verify and sha256(path) != spec["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}; refusing source substitution")
            self._verified.add(spec["sha256"])
            pf = pq.ParquetFile(path)
            if pf.metadata.num_rows != spec["rows"] or self.text_column not in pf.schema_arrow.names:
                raise ValueError(f"Schema/row count mismatch: {path}")
            self._parquet[index] = pf
            self._locate.append((total, total + spec["rows"], index))
            total += spec["rows"]
        if not 0 <= self.row_start <= self.row_end <= total:
            raise ValueError("Invalid row range")

    def _read_rows(self, start, count):
        output = []
        pos = start
        while len(output) < count and pos < self.row_end:
            found = next((x for x in self._locate if x[0] <= pos < x[1]), None)
            if found is None:
                break
            base, end, index = found
            pf = self._parquet[index]
            local = pos - base
            group_start = 0
            for group in range(pf.num_row_groups):
                group_count = pf.metadata.row_group(group).num_rows
                if group_start <= local < group_start + group_count:
                    take = min(count-len(output), group_start+group_count-local, self.row_end-pos)
                    cache_key = (index, group)
                    if self._cached_group != cache_key:
                        table = pf.read_row_group(group, columns=[self.text_column])
                        self._cached_values = table.column(self.text_column).to_pylist()
                        self._cached_group = cache_key
                    values = self._cached_values
                    output.extend(values[local-group_start:local-group_start+take])
                    pos += take
                    break
                group_start += group_count
        return output, pos

    def _fetch_page(self):
        valid = []
        while len(valid) < self.page_size and (self._row_backlog or self.row_cursor < self.row_end):
            if self._row_backlog:
                texts, self._row_backlog = self._row_backlog, []
            else:
                count = (self.page_size-len(valid)) if self.boundary == "row_document" else min(256, self.row_end-self.row_cursor)
                texts, next_pos = self._read_rows(self.row_cursor, count)
                self.stats.raw_rows_read += len(texts)
                self.row_cursor = next_pos
                self.next_offset = next_pos
            if self.boundary == "row_document":
                for value in texts:
                    document = clean_document(value, self.quality)
                    if document is None:
                        self.stats.documents_rejected += 1
                    else:
                        valid.append(document)
                    if len(valid) >= self.page_size:
                        break
                if valid:
                    break
                continue
            for position, value in enumerate(texts):
                stripped = value.strip() if isinstance(value, str) else ""
                if stripped and ARTICLE_TITLE.fullmatch(stripped) and self._article_parts:
                    document = clean_document("\n\n".join(self._article_parts), self.quality)
                    if document is None:
                        self.stats.documents_rejected += 1
                    else:
                        valid.append(document)
                    self._article_parts = [stripped]
                    if len(valid) >= self.page_size:
                        self._row_backlog = texts[position+1:]
                        break
                elif stripped:
                    self._article_parts.append(stripped)
        if self.boundary == "wikitext_article" and self.row_cursor >= self.row_end and not self._row_backlog and self._article_parts:
            document = clean_document("\n\n".join(self._article_parts), self.quality)
            self._article_parts = []
            if document is None:
                self.stats.documents_rejected += 1
            else:
                valid.append(document)
        if not valid:
            raise StopIteration(f"{self.dataset_name} training rows exhausted")
        self.stats.stories_read += len(valid)
        return valid

    def state_dict(self):
        return {**super().state_dict(), "parquet_source":dict(dataset=self.dataset_name,manifest_hash=self.manifest_hash,parser=PARSER,files=[x["sha256"] for x in self.files],row_start=self.row_start,row_end=self.row_end,boundary=self.boundary,quality=self.quality),"row_cursor":self.row_cursor,"article_parts":list(self._article_parts),"row_backlog":list(self._row_backlog)}

    def load_state_dict(self, state):
        source = state.get("parquet_source", {})
        expected = dict(dataset=self.dataset_name,manifest_hash=self.manifest_hash,parser=PARSER,files=[x["sha256"] for x in self.files],row_start=self.row_start,row_end=self.row_end,boundary=self.boundary,quality=self.quality)
        if source != expected:
            raise ValueError("Parquet source, manifest, parser, or row range changed")
        if not isinstance(state.get("row_cursor"), int) or not 0 <= state["row_cursor"] <= self.row_end:
            raise ValueError("Invalid Parquet row cursor")
        super().load_state_dict(state)
        self.row_cursor = state["row_cursor"]
        parts = state.get("article_parts", [])
        if not isinstance(parts, list) or not all(isinstance(part, str) for part in parts):
            raise ValueError("Invalid WikiText article assembly state")
        self._article_parts = list(parts)
        backlog = state.get("row_backlog", [])
        if not isinstance(backlog, list) or not all(value is None or isinstance(value, str) for value in backlog):
            raise ValueError("Invalid Parquet row backlog")
        self._row_backlog = list(backlog)
