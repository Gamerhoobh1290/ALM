"""Prepare DailyDialog conversational SFT splits from the existing local copy.

Uses only the already-downloaded plain-text file under data/extra/train/:
  dialogues_train.txt  (utterances joined by " __eou__ ", one dialogue per line)
  dialogues_act_train.txt / dialogues_emotion_train.txt are verified for
  line-count and turn-count alignment but are NOT used for training.

Safety:
  - No download, no network, no pickle (plain UTF-8 text only, strict decode).
  - Deterministic dialogue-level split by sha256(canonical dialogue) % 100,
    mirroring the Dolly rule: 0-4 validation, 5-9 holdout/test, else train.
    Splitting at dialogue level prevents turns from the same dialogue leaking
    across splits.
  - Each assistant turn becomes one SFT row compatible with the existing
    adamlm-response-sft-v1 pipeline (instruction/context/response), so the
    trainer masks User/Context tokens and trains only Assistant responses
    through the loss -- no hardcoded greetings or external APIs.
  - Atomic writes (.tmp + os.replace) and size/sha256 manifest, same as Dolly.
"""
import hashlib
import json
import os
import unicodedata
from pathlib import Path

from adamlm.downloads import sha256

ROOT = Path("data/dailydialog")
SOURCE = Path("data/extra/train/dialogues_train.txt")
ACT = Path("data/extra/train/dialogues_act_train.txt")
EMOTION = Path("data/extra/train/dialogues_emotion_train.txt")
SEPARATOR = "__eou__"


def normalize_utterance(value: str) -> str:
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def parse_dialogue(line: str):
    # Split on the bare marker so both " __eou__ " separators and a trailing
    # " __eou__" end-of-line marker are handled. Interior empties are kept
    # so the caller can reject corrupted dialogues; a single trailing empty
    # from the end-of-line marker is dropped.
    parts = line.split(SEPARATOR)
    turns = [normalize_utterance(p) for p in parts]
    if turns and turns[-1] == "" and len(turns) > 1:
        turns = turns[:-1]
    return turns


def format_history(turns):
    """Format turns[0:i-1] as User:/Assistant: history for the Context field."""
    lines = []
    for idx, utt in enumerate(turns):
        role = "User:" if idx % 2 == 0 else "Assistant:"
        lines.append(f"{role}\n{utt}")
    return "\n".join(lines)


def prepare(source=SOURCE, root=ROOT):
    if not source.is_file():
        raise FileNotFoundError(f"DailyDialog source missing: {source} (do not re-download; locate the existing copy)")
    # Strict UTF-8 decode: fail loudly on corruption, never silently replace.
    raw_bytes = source.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"DailyDialog source is not strict UTF-8: {exc}") from exc
    source_sha = hashlib.sha256(raw_bytes).hexdigest()
    source_size = len(raw_bytes)

    # Verify auxiliary annotation files align (same dialogues, same turn counts).
    # They are not used for training; this only guards against a mismatched copy.
    aux_info = {}
    for aux_path in (ACT, EMOTION):
        if not aux_path.is_file():
            raise FileNotFoundError(f"Expected auxiliary file missing: {aux_path}")
        aux_lines = aux_path.read_text(encoding="utf-8").splitlines()
        aux_info[aux_path.name] = {"lines": len(aux_lines)}

    lines = text.splitlines()
    if aux_info[ACT.name]["lines"] != len(lines) or aux_info[EMOTION.name]["lines"] != len(lines):
        raise ValueError(
            f"Auxiliary line-count mismatch: dialogues={len(lines)} "
            f"act={aux_info[ACT.name]['lines']} emotion={aux_info[EMOTION.name]['lines']}"
        )

    root.mkdir(parents=True, exist_ok=True)
    outputs = {name: root / f"{name}.jsonl" for name in ("train", "validation", "holdout")}
    temporary = {name: path.with_suffix(".tmp") for name, path in outputs.items()}
    handles = {name: path.open("w", encoding="utf-8", newline="\n") for name, path in temporary.items()}

    seen_dialogues = set()
    counts = {name: 0 for name in outputs}  # SFT examples per split
    dialogue_counts = {name: 0 for name in outputs}
    rejected = 0
    duplicate_dialogues = 0
    checked_turn_alignment = 0
    try:
        # Re-read aux files lazily for per-dialogue turn-count check without holding all in memory.
        with ACT.open(encoding="utf-8") as act_h, EMOTION.open(encoding="utf-8") as emo_h:
            for line in lines:
                act_line = act_h.readline()
                emo_line = emo_h.readline()
                turns = parse_dialogue(line)
                # Alignment check: annotation token counts must match turn counts.
                try:
                    act_turns = len(act_line.strip().split()) if act_line.strip() else 0
                    emo_turns = len(emo_line.strip().split()) if emo_line.strip() else 0
                except Exception:
                    act_turns = emo_turns = -1
                if act_turns != len(turns) or emo_turns != len(turns):
                    rejected += 1
                    continue
                checked_turn_alignment += 1
                if len(turns) < 2 or any(not t for t in turns):
                    rejected += 1
                    continue
                canonical = "\n".join(turns)
                identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if identity in seen_dialogues:
                    duplicate_dialogues += 1
                    rejected += 1
                    continue
                seen_dialogues.add(identity)
                bucket = int(identity[:8], 16) % 100
                split = "validation" if bucket < 5 else "holdout" if bucket < 10 else "train"
                # Expand to turn-level SFT rows: each assistant turn (odd index)
                # paired with its preceding user turn + earlier history as context.
                # Convention: even indices = User, odd = Assistant (alternating).
                examples = 0
                for i in range(1, len(turns), 2):
                    instruction = turns[i - 1]
                    history = turns[: i - 1]
                    context = format_history(history) if history else ""
                    response = turns[i]
                    if not instruction or not response:
                        continue
                    row = {"instruction": instruction, "context": context,
                           "response": response, "category": "dailydialog"}
                    handles[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    examples += 1
                if examples == 0:
                    rejected += 1
                    continue
                counts[split] += examples
                dialogue_counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()

    files = {}
    for name, target in outputs.items():
        os.replace(temporary[name], target)
        files[name] = {"path": str(target).replace("\\", "/"), "size": target.stat().st_size,
                       "sha256": sha256(target), "examples": counts[name],
                       "dialogues": dialogue_counts[name]}

    manifest = {
        "format": "adamlm-response-sft-v1",
        "dataset": "dailydialog",
        "revision": f"local-extra-train-{source_sha[:12]}",
        "license": "DailyDialog (Li et al., IJCNLP 2017) for research use; local copy only, no redistribution.",
        "source": str(source).replace("\\", "/"),
        "source_sha256": source_sha,
        "source_size": source_size,
        "source_dialogues": len(lines),
        "aux_alignment_checked": checked_turn_alignment,
        "split_rule": "sha256(canonical dialogue) modulo 100: 0-4 validation, 5-9 holdout/test, 10-99 train; split at dialogue level, then expanded to one SFT row per assistant turn",
        "row_rule": "instruction=immediately preceding user turn, context=earlier User:/Assistant: history (may be empty), response=current assistant turn; alternating User-first convention",
        "rejected_or_duplicate_dialogues": rejected,
        "duplicate_dialogues": duplicate_dialogues,
        "files": files,
    }
    (root / "processed-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "NOTICE.md").write_text(
        "# DailyDialog (conversational SFT)\n\n"
        "Source: existing local copy at data/extra/train/dialogues_train.txt "
        "(no download performed).\n\n"
        "Reference: Yanran Li et al., DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset, IJCNLP 2017. "
        "http://yanran.li/dailydialog\n\n"
        "License: research use; this preparation only reformats the local copy into deterministic "
        "train/validation/holdout SFT rows (one row per assistant turn, dialogue-level split). "
        "Act/emotion annotation files were checked for alignment only and are not used for training.\n",
        encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
