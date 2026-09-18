"""Generate from a trusted local checkpoint; legacy checkpoints use default config."""
import argparse
from adamlm.inference import complete

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--prompt", default="Once upon a time, a little girl")
parser.add_argument("--tokens", type=int, default=300)
parser.add_argument("--interactive", action="store_true", help="Enter independent story prompts; /quit exits")
parser.add_argument("--chat", action="store_true", help="Format each prompt as a Dolly-style user/assistant turn")
parser.add_argument("--messages-json", help="JSON array of role/text messages for reproducible multi-turn chat")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=40)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stream", action="store_true",
                    help="Emit incremental reply text as JSON lines (one per update) for live UI streaming")
args = parser.parse_args()


def _emit_stream(text):
    import json as _json
    import sys as _sys
    _sys.stdout.write(_json.dumps({"text": text}, ensure_ascii=False) + "\n")
    _sys.stdout.flush()


def answer(user_prompt):
    import json as _json
    messages = _json.loads(args.messages_json) if args.messages_json else None
    if args.stream:
        return complete(args.checkpoint, user_prompt, tokens=args.tokens, temperature=args.temperature,
                        top_k=args.top_k, seed=args.seed, chat=args.chat, messages=messages,
                        stream=_emit_stream)
    return complete(args.checkpoint, user_prompt, tokens=args.tokens, temperature=args.temperature,
                    top_k=args.top_k, seed=args.seed, chat=args.chat, messages=messages)


if args.interactive:
    while True:
        try:
            prompt = input(("Question" if args.chat else "Prompt") + " (/quit to exit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip() == "/quit":
            break
        if prompt.strip():
            print(answer(prompt))
elif args.stream:
    # Streaming already emitted the reply as JSON lines; print nothing extra
    # so the backend's incremental parser sees only data lines.
    answer(args.prompt)
else:
    print(answer(args.prompt))
