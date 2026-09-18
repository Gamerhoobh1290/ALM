"""Evaluate one or more trusted local AdamLM checkpoints on the frozen suite."""

import argparse
import json

from adamlm.evaluation import evaluate_checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic AdamLM checkpoint evaluation. Checkpoints use trusted torch.load state."
    )
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files or directories; directories use their latest step")
    parser.add_argument("--suite", default="config/eval-suite.json")
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, help="Override the suite generation limits")
    args = parser.parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    report = evaluate_checkpoints(
        args.checkpoints,
        args.suite,
        args.output,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    for result in report["checkpoints"]:
        checkpoint = result["checkpoint"]
        summary = result["summary"]
        print(json.dumps({
            "checkpoint": checkpoint["path"],
            "step": checkpoint["step"],
            "correct": summary["correct"],
            "scored": summary["scored"],
            "accuracy": summary["accuracy"],
        }))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
