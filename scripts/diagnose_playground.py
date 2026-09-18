"""Fair Playground diagnosis: same prompts, same fixed seed, same settings.

Compares (CPU, no training launched):
  A = current default  sft-assistant step_00000245.pt (1.00M tokens)
  B = saved 8.29M     auto-assistant-20260917-215605-s1 step_00002024.pt
  C = earlier         sft-assistant step_00000100.pt

Fixed SEED=42 + TOKENS=60/TEMP=0.8/TOPK=40 mirrors scripts/eval_assistant.py
so checkpoint differences come from weights only — never from sampling luck.
Normal Playground generation stays random-by-default (see web._parse_playground_seed);
training and evaluation seeds are intentionally untouched.

Writes results/playground-diagnosis.md + .json. Exit 0 always (judgment is human).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from adamlm.inference import complete

ROOT = Path(__file__).resolve().parents[1]
TOKENS, TEMP, TOPK, SEED = 60, 0.8, 40, 42
PROMPTS = ["hi", "hello", "Good morning! How are you today?",
           "What does the Windows Recycle Bin do?", "Name three red fruits."]
CHECKPOINTS = [
    ("A-default-1M", "results/sft-assistant/checkpoints/step_00000245.pt"),
    ("B-newer-8.29M", "results/auto-assistant-20260917-215605-s1/checkpoints/step_00002024.pt"),
    ("C-earlier", "results/sft-assistant/checkpoints/step_00000100.pt"),
]


def checks(reply):
    rep = re.findall(r"(\b\w+(?:\s+\w+){4,})(?:\s+\1){1,}", reply or "")
    printable = sum(c.isprintable() or c.isspace() for c in (reply or "")) / max(1, len(reply or ""))
    return {"non_empty": bool((reply or "").strip()), "no_replacement_char": "�" not in (reply or ""),
            "printable_ratio": round(printable, 4), "no_prompt_echo": "User:" not in (reply or ""),
            "no_runaway_repeat": not rep}


def run_meta(ckpt_path):
    run_dir = Path(ckpt_path).parent.parent
    out = {}
    for name in ("status.json", "summary.json"):
        try:
            out[name] = json.loads((ROOT / run_dir / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            out[name] = {}
    return out


def main():
    rows = []
    for label, ckpt in CHECKPOINTS:
        path = ROOT / ckpt
        meta = run_meta(ckpt)
        for prompt in PROMPTS:
            try:
                reply = complete(path, prompt, tokens=TOKENS, temperature=TEMP,
                                 top_k=TOPK, seed=SEED, chat=True, device="cpu")
            except Exception as exc:  # never hide a broken checkpoint
                reply = f"<ERROR: {exc}>"
            rows.append({"checkpoint": label, "path": ckpt, "prompt": prompt,
                         "reply": reply, "checks": checks(reply),
                         "settings": {"tokens": TOKENS, "temperature": TEMP, "top_k": TOPK, "seed": SEED}})
        rows.append({"checkpoint": label, "meta": meta, "kind": "run-meta"})
    out_json = ROOT / "results" / "playground-diagnosis.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Playground diagnosis (fixed seed 42, identical settings, CPU, no training)",
             "", f"Settings: tokens={TOKENS} temp={TEMP} top_k={TOPK} seed={SEED} chat=true",
             "", "## Replies"]
    for r in rows:
        if r.get("kind") == "run-meta":
            continue
        lines += [f"### {r['checkpoint']} :: {r['prompt']!r}", f"A: {r['reply'][:600]!r}", f"checks: {r['checks']}", ""]
    lines += ["## Run validation (status/summary, no new training)"]
    for label, ckpt in CHECKPOINTS:
        meta = run_meta(ckpt)
        st, sm = meta.get("status.json", {}), meta.get("summary.json", {})
        lines += [f"- {label} `{ckpt}`: state={st.get('state')} step={st.get('step')} tokens={st.get('tokens')}/{st.get('target_tokens')} "
                  f"train={st.get('train_loss')} val={st.get('validation_loss')} holdout={sm.get('final_holdout_loss')}"]
    lines += ["", "## Readout (fill after inspecting replies above)",
              "- Does B beat A on greetings? …", "- Does B beat A on basic instructions? …",
              "- Recommendation for one targeted next experiment (no auto-launch): …", ""]
    (ROOT / "results" / "playground-diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_json} + results/playground-diagnosis.md")


if __name__ == "__main__":
    main()
