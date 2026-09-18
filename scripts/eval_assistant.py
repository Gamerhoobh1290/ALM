"""Small held-out evaluation for the assistant experiment.

Compares checkpoints with identical settings (temp 0.8, top-k 40, 60 tokens,
seed 42, CPU) on:
  - greetings: 1 genuinely held-out DailyDialog row + 2 hand-written probes
  - simple instructions: 4 genuinely held-out Dolly rows (one per category:
    classification, open_qa, brainstorming, general_qa) + 1 hand-written probe
  - follow-ups: 1 held-out DailyDialog with-context row + 1 scripted 2-turn probe
  - coherent-English auto-checks on every reply (non-empty, no U+FFFD,
    printable ratio, no prompt echo, no runaway repetition)

Held-out rows come from the never-trained validation-excluded holdout splits
(data/*/holdout.jsonl); hand-written probes are labeled as such in the output.
Prints a readable table and writes JSON. Exit 0 always (judgment is human).

Usage:
  .venv/Scripts/python scripts/eval_assistant.py <label=checkpoint>... [--out path]
"""
import json
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from adamlm.inference import build_chat_prompt, extract_response, load_checkpoint
from adamlm.training import generate as gen_fn

TOKENS, TEMP, TOPK, SEED = 60, 0.8, 40, 42
GREET = re.compile(r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b", re.I)


def load_holdout(name):
    with open(f"data/{name}/holdout.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_eval_set():
    dd = load_holdout("dailydialog")
    dolly = load_holdout("dolly")
    items = []
    held_greet = next(r for r in dd if not r.get("context", "").strip() and GREET.search(r["instruction"]))
    items.append({"id": "greet-heldout-dd", "kind": "greeting", "source": "held-out",
                  "instruction": held_greet["instruction"], "context": "", "reference": held_greet["response"]})
    for i, text in enumerate(["hey", "Good morning! How are you today?"]):
        items.append({"id": f"greet-hand-{i}", "kind": "greeting", "source": "hand-written",
                      "instruction": text, "context": "", "reference": None})
    cats = ["classification", "open_qa", "brainstorming", "general_qa"]
    for cat in cats:
        row = next(r for r in dolly
                   if r.get("category") == cat and len(r["instruction"] + r.get("context", "")) <= 1200)
        items.append({"id": f"instr-heldout-{cat}", "kind": "instruction", "source": "held-out",
                      "instruction": row["instruction"], "context": row.get("context", ""),
                      "reference": row["response"], "category": cat})
    items.append({"id": "instr-hand-fruits", "kind": "instruction", "source": "hand-written",
                  "instruction": "Name three red fruits.", "context": "", "reference": None})
    held_ctx = next(r for r in dd if r.get("context", "").strip()
                    and len(r["instruction"] + r.get("context", "")) <= 1200)
    items.append({"id": "follow-heldout-dd", "kind": "follow-up", "source": "held-out",
                  "instruction": held_ctx["instruction"], "context": held_ctx["context"],
                  "reference": held_ctx["response"]})
    items.append({"id": "follow-hand-pizza", "kind": "follow-up", "source": "hand-written turns",
                  "turns": ["I love pizza.", "What about you?"], "reference": None})
    return items


def format_prompt(item, history=(), tokenizer=None, block_size=512):
    # Same template the trainer uses (sft_data.encode_example): the latest
    # user turn as instruction, everything earlier verbatim in Context:.
    # Dolly reference passages and DailyDialog labeled histories both travel
    # in that block, exactly as seen during training.
    if "turns" in item:
        texts = list(history)  # flat [u, a, ...] ending with the latest turn
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "text": text}
                    for i, text in enumerate(texts)]
        return build_chat_prompt(messages, tokenizer, block_size, TOKENS)["prompt"]
    else:
        latest, hist = item["instruction"], item.get("context", "").strip()
        if not hist and tokenizer is not None:
            return build_chat_prompt([{"role": "user", "text": latest}], tokenizer,
                                     block_size, TOKENS)["prompt"]
    if hist:
        return f"User:\n{latest}\n\nContext:\n{hist}\n\nAssistant:\n"
    return f"User:\n{latest}\n\nAssistant:\n"


def checks(reply, prompt):
    rep = re.findall(r"(\b\w+(?:\s+\w+){4,})(?:\s+\1){1,}", reply)
    printable = sum(c.isprintable() or c.isspace() for c in reply) / max(1, len(reply))
    return {
        "non_empty": bool(reply.strip()),
        "no_replacement_char": "�" not in reply,
        "printable_ratio": round(printable, 4),
        "no_prompt_echo": "User:" not in reply,
        "no_runaway_repeat": not rep,
    }


def run_checkpoint(label, checkpoint):
    model, tok, _ = load_checkpoint(checkpoint, device="cpu")
    rows = []
    for item in build_eval_set():
        if "turns" in item:
            history, replies = [], []
            for turn in item["turns"]:
                prompt = format_prompt(item, history + [turn], tok, model.config.block_size)
                text = gen_fn(model, prompt, TOKENS, temperature=TEMP, top_k=TOPK, seed=SEED,
                              tokenizer=tok, stop_sequences=("\nUser:\n", "\n\nUser:\n"))
                reply = extract_response(text, prompt)
                replies.append(reply)
                history += [turn, reply]
            rows.append({"id": item["id"], "kind": item["kind"], "source": item["source"],
                         "turns": [{"user": u, "reply": r} for u, r in zip(item["turns"], replies)],
                         "checks": [checks(r, "") for r in replies]})
        else:
            prompt = format_prompt(item, tokenizer=tok, block_size=model.config.block_size)
            text = gen_fn(model, prompt, TOKENS, temperature=TEMP, top_k=TOPK, seed=SEED,
                          tokenizer=tok, stop_sequences=("\nUser:\n", "\n\nUser:\n"))
            reply = extract_response(text, prompt)
            rows.append({"id": item["id"], "kind": item["kind"], "source": item["source"],
                         "instruction": item["instruction"],
                         "context": item.get("context", "")[:200],
                         "reference": (item.get("reference") or "")[:200],
                         "reply": reply, "checks": checks(reply, prompt)})
    del model
    return {"label": label, "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "settings": {"tokens": TOKENS, "temperature": TEMP, "top_k": TOPK, "seed": SEED}, "rows": rows}


def main():
    pairs, out = [], None
    for arg in sys.argv[1:]:
        if arg.startswith("--out="):
            out = arg.split("=", 1)[1]
        elif "=" in arg:
            pairs.append(arg.split("=", 1))
    if not pairs:
        raise SystemExit("usage: eval_assistant.py <label=checkpoint>... [--out=path]")
    results = [run_checkpoint(label, ckpt) for label, ckpt in pairs]
    for res in results:
        print(f"===== {res['label']} ({res['checkpoint']}) =====")
        for row in res["rows"]:
            print(f"--- {row['id']} [{row['kind']}/{row['source']}] ---")
            if "turns" in row:
                for t in row["turns"]:
                    print(f"  U: {t['user'][:120]!r}\n  A: {t['reply'][:220]!r}")
            else:
                print(f"  I: {row['instruction'][:150]!r}")
                if row.get("reference"):
                    print(f"  R: {row['reference'][:150]!r}")
                print(f"  A: {row['reply'][:220]!r}")
            print(f"  checks: {row['checks']}")
    if out:
        Path(out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
