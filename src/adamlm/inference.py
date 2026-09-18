"""Checkpoint-backed AdamLM inference shared by the CLI and desktop UI."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import DecoderTransformer, ModelConfig
from .training import generate


CHAT_STOP_SEQUENCES = ("\nUser:\n", "\n\nUser:\n")


def extract_response(full_text: str, prompt: str) -> str:
    """Extract only the assistant continuation and remove leaked next turns."""
    reply = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    stops = [reply.find(stop) for stop in CHAT_STOP_SEQUENCES if stop in reply]
    if stops:
        reply = reply[:min(stops)]
    return reply.lstrip()


def build_chat_prompt(messages: list[dict], tokenizer, block_size: int,
                      max_new_tokens: int) -> dict:
    """Build the SFT chat template using exact tokenizer based truncation."""
    cleaned = []
    for message in messages or []:
        role = str((message or {}).get("role") or "").lower()
        text = str((message or {}).get("text") or "")
        if role in {"user", "assistant"} and text.strip():
            cleaned.append({"role": role, "text": text.strip()})
    if not cleaned or cleaned[-1]["role"] != "user":
        raise ValueError("Chat needs a user message to answer")
    prompt_budget = int(block_size) - int(max_new_tokens)
    if prompt_budget <= 16:
        raise ValueError("Requested reply leaves too little model context for the chat prompt")

    latest = cleaned[-1]["text"]
    history = cleaned[:-1]

    def render(previous, instruction):
        prompt = f"User:\n{instruction}\n"
        if previous:
            context = "\n".join(
                f"{'User' if item['role'] == 'user' else 'Assistant'}:\n{item['text']}"
                for item in previous)
            prompt += f"\nContext:\n{context}\n"
        return prompt + "\nAssistant:\n"

    prompt = render(history, latest)
    trimmed = False
    while history and len(tokenizer.encode(prompt)) > prompt_budget:
        history = history[1:]
        trimmed = True
        prompt = render(history, latest)
    if len(tokenizer.encode(prompt)) > prompt_budget:
        # Keep the longest suffix of the latest message that fits exactly.
        lo, hi, best = 0, len(latest), None
        while lo <= hi:
            size = (lo + hi) // 2
            candidate = latest[-size:] if size else ""
            rendered = render([], candidate)
            if len(tokenizer.encode(rendered)) <= prompt_budget:
                best = rendered
                lo = size + 1
            else:
                hi = size - 1
        if best is None:
            raise ValueError("Model context is too small for the chat template")
        prompt = best
        trimmed = True
    prompt_tokens = len(tokenizer.encode(prompt))
    return {"prompt": prompt, "kept_turns": len(history) + 1,
            "total_turns": len(cleaned), "trimmed": trimmed,
            "prompt_tokens": prompt_tokens, "context_limit": int(block_size),
            "max_new_tokens": int(max_new_tokens)}


def resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        paths = sorted(path.glob("step_*.pt"))
        if not paths:
            raise ValueError("Checkpoint directory is empty")
        path = paths[-1]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_checkpoint(value: str | Path, device: str | None = None):
    path = resolve_checkpoint(value)
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state.get("model_config") or json.loads(Path("config/default.json").read_text())["model"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderTransformer(ModelConfig(**config)).to(device)
    model.load_state_dict(state["model"])
    tokenizer = None
    if "tokenizer_sha256" in state.get("extra", {}):
        from .bpe import BPETokenizer
        configured = Path(state["extra"]["tokenizer_path"])
        if not configured.is_file():
            project = Path(__file__).resolve().parents[2]
            configured = project / configured
        tokenizer = BPETokenizer(configured)
        if tokenizer.sha256 != state["extra"]["tokenizer_sha256"]:
            raise ValueError("Tokenizer hash does not match checkpoint")
    elif model.config.vocab_size != 256:
        raise ValueError("Non-byte checkpoint is missing its tokenizer identity")
    return model, tokenizer, path


def complete(value: str | Path, prompt: str = "", *, messages: list[dict] | None = None,
             tokens=300, temperature=0.8, top_k=40, seed=42, chat=False,
             device: str | None = None, stream=None) -> str:
    model, tokenizer, _ = load_checkpoint(value, device)
    chat_mode = chat or messages is not None
    if chat_mode:
        chat_messages = messages if messages is not None else [{"role": "user", "text": prompt}]
        formatted = build_chat_prompt(chat_messages, tokenizer, model.config.block_size, tokens)["prompt"]
    else:
        formatted = prompt
    if stream is None:
        text = generate(model, formatted, tokens, temperature=temperature, top_k=top_k,
                        seed=seed, tokenizer=tokenizer,
                        stop_sequences=CHAT_STOP_SEQUENCES if chat_mode else ())
    else:
        def _emit(full):
            stream(extract_response(full, formatted) if chat_mode else full)
        text = generate(model, formatted, tokens, temperature=temperature, top_k=top_k,
                        seed=seed, tokenizer=tokenizer, on_text=_emit,
                        stop_sequences=CHAT_STOP_SEQUENCES if chat_mode else ())
    return extract_response(text, formatted) if chat_mode else text
