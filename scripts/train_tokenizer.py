"""Fit BPE exclusively on bounded training rows; report frozen held-out coverage."""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from adamlm.bpe import BPETokenizer, BOUNDARY
from adamlm.data import TinyStoriesStream
from adamlm.storage import StorageBudget

out = Path("tokenizers/bpe4096")
if (out / "tokenizer.json").exists():
    raise SystemExit("Frozen tokenizer exists; refusing to replace it")
StorageBudget(".").check(64*2**20)
out.mkdir(parents=True, exist_ok=True)
validation = json.loads(Path("results/first-run/validation.json").read_text(encoding="utf-8"))["texts"]
excluded = {hashlib.sha256(t.strip().encode()).hexdigest() for t in validation}
stream = TinyStoriesStream(page_size=100, timeout=20)
texts = []
for page in range(50):
    texts.extend(t for t in stream._fetch_page() if hashlib.sha256(t.strip().encode()).hexdigest() not in excluded)
    if (page+1) % 10 == 0:
        print(f"training rows fetched={(page+1)*100}", flush=True)
backend = Tokenizer(models.BPE(byte_fallback=True))
backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
backend.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=4096, min_frequency=2, special_tokens=[BOUNDARY],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
backend.train_from_iterator(texts, trainer=trainer, length=len(texts))
if backend.get_vocab_size() != 4096:
    raise RuntimeError("Insufficient corpus for requested vocabulary")
backend.save(str(out / "tokenizer.json"))
tokenizer = BPETokenizer(out / "tokenizer.json")
lengths = [len(tokenizer.encode(t)) for t in validation]
byte_lengths = [len(t.encode()) for t in validation]
assert all(tokenizer.decode(tokenizer.encode(t)) == t for t in validation)
manifest = dict(source="roneneldan/TinyStories", split="train", first_row=0, rows_fetched=5000,
    stories_used=len(texts), bytes_used=sum(len(t.encode()) for t in texts),
    corpus_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
    tokenizer_sha256=tokenizer.sha256, vocabulary=4096, boundary_id=tokenizer.boundary_id,
    byte_alphabet_size=256, validation_not_used_for_merges=True,
    evaluation=dict(stories=len(validation), bytes=sum(byte_lengths), bpe_text_tokens=sum(lengths),
        bpe_tokens_with_boundaries=sum(lengths)+len(lengths), bytes_per_bpe_token=sum(byte_lengths)/sum(lengths),
        original_context_bytes=256, new_context_tokens=512,
        estimated_new_context_bytes=512*sum(byte_lengths)/(sum(lengths)+len(lengths)),
        original_whole_stories_fitting=sum(n<=256 for n in byte_lengths),
        new_whole_stories_fitting=sum(n+1<=512 for n in lengths), roundtrip_stories=len(validation)))
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)
