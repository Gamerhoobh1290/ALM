"""Download, verify, normalize, deduplicate, and split Dolly 15k locally."""
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

from adamlm.downloads import download, sha256
from adamlm.storage import StorageBudget


ROOT = Path("data/dolly")
CITATION = re.compile(r"\[(?:\d+|citation needed)\]", re.IGNORECASE)


def normalize(value):
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    value = CITATION.sub("", value)
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def prepare(source, root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    outputs = {name: root / f"{name}.jsonl" for name in ("train", "validation", "holdout")}
    temporary = {name: path.with_suffix(".tmp") for name, path in outputs.items()}
    handles = {name: path.open("w", encoding="utf-8", newline="\n") for name, path in temporary.items()}
    seen = set()
    counts = {name: 0 for name in outputs}
    rejected = 0
    try:
        with Path(source).open(encoding="utf-8") as source_handle:
            for line in source_handle:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    rejected += 1
                    continue
                row = {key: normalize(raw.get(key, "")) for key in ("instruction", "context", "response", "category")}
                if not row["instruction"] or not row["response"]:
                    rejected += 1
                    continue
                identity = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if identity in seen:
                    rejected += 1
                    continue
                seen.add(identity)
                bucket = int(identity[:8], 16) % 100
                split = "validation" if bucket < 5 else "holdout" if bucket < 10 else "train"
                handles[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    files = {}
    for name, target in outputs.items():
        os.replace(temporary[name], target)
        files[name] = {"path": str(target).replace("\\", "/"), "size": target.stat().st_size,
                       "sha256": sha256(target), "examples": counts[name]}
    upstream = json.loads(Path("config/dolly.json").read_text(encoding="utf-8"))
    manifest = {"format": "adamlm-response-sft-v1", "dataset": upstream["dataset"],
                "revision": upstream["revision"], "license": upstream["license"],
                "source": upstream["source"], "upstream_sha256": upstream["train"]["sha256"],
                "split_rule": "sha256(canonical row) modulo 100: 0-4 validation, 5-9 holdout, 10-99 train",
                "rejected_or_duplicate": rejected, "files": files}
    target = root / "processed-manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "NOTICE.md").write_text(
        "# Databricks Dolly 15k\n\nSource: https://huggingface.co/datasets/databricks/databricks-dolly-15k\n\n"
        "Revision: bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a\n\nLicense: CC BY-SA 3.0. "
        "The processed files normalize whitespace, remove bracketed citation markers, deduplicate exact canonical rows, and create deterministic splits.\n",
        encoding="utf-8")
    return manifest


if __name__ == "__main__":
    cfg = json.loads(Path("config/bpe512-local.json").read_text())
    guard = StorageBudget(".", cfg["storage"]["project_budget_gb"], cfg["storage"]["cache_limit_gb"], cfg["storage"]["minimum_free_disk_gb"])
    upstream = json.loads(Path("config/dolly.json").read_text(encoding="utf-8"))
    source = download(upstream, "train", guard, root=ROOT)
    print(json.dumps(prepare(source), indent=2))
