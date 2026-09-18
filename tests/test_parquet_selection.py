import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from adamlm.bpe import BPETokenizer
from adamlm.dataset_selection import MixtureStream
from adamlm.parquet_data import ParquetDocumentStream, clean_document
from adamlm.storage import StorageBudget


def make_dataset(tmp_path):
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"text": ["", "First document "*20, None, "Second document "*20, "Third document "*20, "Fourth document "*20]}), path, row_group_size=2)
    spec = {"path": str(path), "rows": 6, "size": path.stat().st_size,
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
    manifest = {"toy": {"kind":"parquet", "train":[spec], "validation":[spec], "text_column":"text"}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_parquet_skips_empty_and_checkpoint_replays(tmp_path):
    path = make_dataset(tmp_path)
    tok = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    a = ParquetDocumentStream(path, "toy", tokenizer=tok, page_size=4)
    a.next_batch(1, 8)
    saved = a.state_dict()
    expected = a.next_batch(1, 8)
    b = ParquetDocumentStream(path, "toy", tokenizer=tok, page_size=4)
    b.load_state_dict(saved)
    assert b.next_batch(1, 8) == expected
    assert a.row_cursor == 4 and a.stats.stories_read == 2


def test_parquet_rejects_changed_file(tmp_path):
    path = make_dataset(tmp_path)
    tok = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    stream = ParquetDocumentStream(path, "toy", tokenizer=tok)
    path_obj = Path(json.loads(path.read_text())['toy']['train'][0]['path'])
    path_obj.write_bytes(path_obj.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Size mismatch"):
        ParquetDocumentStream(path, "toy", tokenizer=tok)


def test_mixture_is_explicit_and_stateful(tmp_path):
    path = make_dataset(tmp_path)
    tok = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    a = ParquetDocumentStream(path, "toy", tokenizer=tok, page_size=2)
    b = ParquetDocumentStream(path, "toy", tokenizer=tok, page_size=2)
    mix = MixtureStream([a,b],[0.5,0.5])
    mix.next_batch(1, 8)
    state = mix.state_dict()
    assert state["mixture"]["components"] == ["toy", "toy"]
    restored = MixtureStream([ParquetDocumentStream(path,"toy",tokenizer=tok,page_size=2),ParquetDocumentStream(path,"toy",tokenizer=tok,page_size=2)],[0.5,0.5])
    restored.load_state_dict(state)
    assert restored.schedule_index == mix.schedule_index
    assert restored.next_batch(1, 8) == mix.next_batch(1, 8)


def test_parquet_rejects_changed_cursor_identity(tmp_path):
    path = make_dataset(tmp_path)
    tok = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    stream = ParquetDocumentStream(path, "toy", tokenizer=tok, row_end=4)
    saved = stream.state_dict()
    changed = ParquetDocumentStream(path, "toy", tokenizer=tok, row_end=5)
    with pytest.raises(ValueError, match="source, manifest, parser, or row range changed"):
        changed.load_state_dict(saved)


def test_wikitext_rows_are_assembled_into_articles_and_resume(tmp_path):
    parquet = tmp_path / "wiki.parquet"
    rows = ["", " = First Article = \n", "First paragraph. " * 20, "Second paragraph. " * 20,
            " = Second Article = \n", "Its paragraph. " * 30]
    pq.write_table(pa.table({"text": rows}), parquet, row_group_size=2)
    spec = {"path": str(parquet), "rows": len(rows), "size": parquet.stat().st_size,
            "sha256": __import__("hashlib").sha256(parquet.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"wiki": {"train": [spec], "text_column": "text",
        "boundary": "wikitext_article", "quality": {"min_words": 10}}}))
    tok = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    stream = ParquetDocumentStream(manifest, "wiki", tokenizer=tok, page_size=1)
    first = stream._fetch_page()
    assert len(first) == 1 and "First paragraph" in first[0] and "Second paragraph" in first[0]
    assert "Second Article" not in first[0]
    state = stream.state_dict()
    assert state["article_parts"] == ["= Second Article ="]
    restored = ParquetDocumentStream(manifest, "wiki", tokenizer=tok, page_size=1)
    restored.load_state_dict(state)
    assert restored._fetch_page() == stream._fetch_page()


def test_quality_filter_rejects_corruption_and_repetition():
    assert clean_document("\x00broken text") is None
    assert clean_document("same line\n" * 10, {"min_words": 2}) is None
    assert clean_document("A useful paragraph with varied words and clear prose.", {"min_words": 5})


def test_mixture_weights_apply_to_interleaved_token_batches(tmp_path):
    path = make_dataset(tmp_path)
    tok = BPETokenizer("tokenizers/bpe4096/tokenizer.json")
    streams = [ParquetDocumentStream(path, "toy", tokenizer=tok, page_size=2) for _ in range(2)]
    mix = MixtureStream(streams, [0.7, 0.3])
    assert set(mix.schedule[:10]) == {0, 1}
    assert mix.schedule.count(0) == 700 and mix.schedule.count(1) == 300
