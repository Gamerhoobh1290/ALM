"""Reproducible tokenizer and local data-loader measurements."""
import argparse
import json
import statistics
import time
from pathlib import Path

from adamlm.bpe import BPETokenizer
from adamlm.parquet_data import ParquetDocumentStream


parser = argparse.ArgumentParser()
parser.add_argument("--documents", type=int, default=200)
parser.add_argument("--batches", type=int, default=100)
parser.add_argument("--output", default="results/data-profile.json")
args = parser.parse_args()
tokenizer = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
result = {"tokenizer_sha256": tokenizer.sha256, "datasets": {}}

for name in ("wikitext103", "fineweb_edu"):
    stream = ParquetDocumentStream("config/extra-datasets.json", name, tokenizer=tokenizer, page_size=16, verify=False)
    documents = []
    while len(documents) < args.documents:
        documents.extend(stream._fetch_page())
    documents = documents[:args.documents]
    started = time.perf_counter()
    encoded = [tokenizer.encode(document) for document in documents]
    encode_seconds = time.perf_counter() - started
    batch_stream = ParquetDocumentStream("config/extra-datasets.json", name, tokenizer=tokenizer, page_size=16, verify=False)
    timings = []
    for _ in range(args.batches):
        started = time.perf_counter()
        batch_stream.next_batch(8, 512)
        timings.append(time.perf_counter() - started)
    characters = sum(map(len, documents))
    utf8_bytes = sum(len(document.encode("utf-8")) for document in documents)
    tokens = sum(map(len, encoded))
    result["datasets"][name] = {
        "documents": len(documents), "characters": characters, "tokens": tokens,
        "characters_per_token": characters / tokens, "utf8_bytes_per_token": utf8_bytes / tokens,
        "tokenizer_tokens_per_second": tokens / encode_seconds,
        "loader_mean_ms_per_batch": statistics.mean(timings) * 1000,
        "loader_p95_ms_per_batch": sorted(timings)[int(len(timings) * .95) - 1] * 1000,
        "raw_rows_read": batch_stream.stats.raw_rows_read,
        "documents_rejected": batch_stream.stats.documents_rejected,
    }

target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
