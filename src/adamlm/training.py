"""Shared precision, accumulation, evaluation, and sampling operations."""
from contextlib import nullcontext

import torch


def autocast(precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def update(model, optimizer, batches, precision="fp32", scaler=None):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for x, y in batches:
        with autocast(precision):
            _, loss = model(x, y)
        losses.append(loss.detach())
        loss = loss / len(batches)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=scaler is None)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, batches, precision="fp32"):
    was_training = model.training
    model.eval()
    losses = []
    for x, y in batches:
        with autocast(precision):
            _, loss = model(x, y)
        losses.append(loss.float())
    result = torch.stack(losses).mean().item()
    model.train(was_training)
    return result


def _gpt2_byte_alphabet():
    """Byte-to-character alphabet used by the frozen byte-level BPE files.

    Mirrors the standard GPT-2 bytes_to_unicode table so each vocabulary
    surface can be mapped back to the exact bytes it stands for. Used only
    to validate emitted byte streams at inference time; the tokenizer
    itself is never modified.
    """
    byte_list = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    char_list = byte_list[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_list:
            byte_list.append(byte)
            char_list.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(byte_list, char_list)}


_BYTE_ALPHABET = _gpt2_byte_alphabet()
_CHAR_TO_BYTE = {char: byte for byte, char in _BYTE_ALPHABET.items()}


def token_byte_table(tokenizer):
    """Map every vocabulary id to the exact bytes it emits on decode."""
    backend = getattr(tokenizer, "backend", None)
    if backend is not None and hasattr(backend, "get_vocab"):
        vocab = backend.get_vocab()
        surfaces = [None] * len(vocab)
        for surface, index in vocab.items():
            surfaces[index] = surface
        return [bytes(_CHAR_TO_BYTE[char] for char in surface) for surface in surfaces]
    size = getattr(tokenizer, "vocab_size", 256)
    return [bytes((index,)) for index in range(size)]


def incomplete_utf8_tail(data: bytes) -> bytes:
    """Trailing bytes of a valid UTF-8 prefix that do not yet form a character."""
    n = len(data)
    if n == 0:
        return b""
    count = 0
    while count < n and count < 3 and data[n - 1 - count] & 0xC0 == 0x80:
        count += 1
    if count + 1 > n:
        return data
    lead = data[n - 1 - count]
    if lead & 0x80 == 0:
        return b""
    if lead & 0xE0 == 0xC0:
        need = 2
    elif lead & 0xF0 == 0xE0:
        need = 3
    elif lead & 0xF8 == 0xF0:
        need = 4
    else:
        return data
    have = count + 1
    return data[n - have:] if have < need else b""


def _utf8_prefix_allowed(table, boundary, tail):
    """Boolean mask of ids that keep the emitted stream a valid UTF-8 prefix."""
    import codecs
    flags = []
    for index, token_bytes in enumerate(table):
        if index == boundary:
            flags.append(True)
            continue
        probe = codecs.getincrementaldecoder("utf-8")()
        try:
            probe.decode(tail + token_bytes, False)
            flags.append(True)
        except UnicodeDecodeError:
            flags.append(False)
    return torch.tensor(flags)


@torch.no_grad()
def generate(model, prompt, new_tokens=300, temperature=0.8, top_k=40, seed=42, tokenizer=None,
             on_text=None, stream_every=8, stop_sequences=()):
    """Sample a continuation. When ``on_text`` is given it is called with the
    current decoded text every ``stream_every`` tokens and once at the end;
    training/eval callers leave it unset for identical single-shot behavior."""
    from .tokenizer import ByteTokenizer
    if not prompt or temperature <= 0:
        raise ValueError("Nonempty prompt and positive temperature required")
    tokenizer = tokenizer or ByteTokenizer()
    table = token_byte_table(tokenizer)
    boundary = getattr(tokenizer, "boundary_id", None)
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    was_training = model.training
    model.eval()
    # Constrain sampling only when the table covers the model vocabulary;
    # otherwise keep the historical decode path unchanged.
    constrain = len(table) == model.config.vocab_size
    emitted = bytearray()
    tail = b""
    stopped_text = None
    mask_cache = {}
    def _current_text():
        if stopped_text is not None:
            return stopped_text
        if not constrain:
            return tokenizer.decode(ids[0].tolist())
        clean_now = bytes(emitted)
        cur_tail = incomplete_utf8_tail(clean_now)
        if cur_tail:
            clean_now = clean_now[:len(clean_now) - len(cur_tail)]
        return prompt + clean_now.decode("utf-8")

    for step_i in range(new_tokens):
        logits, _ = model(ids[:, -model.config.block_size:])
        scores = logits[:, -1].float() / temperature
        if constrain:
            # Never let sampling produce an undecodable byte stream: mask any
            # token that would break UTF-8 prefix validity. The stop token
            # always stays available so generation can still terminate.
            allowed = mask_cache.get(tail)
            if allowed is None:
                allowed = _utf8_prefix_allowed(table, boundary, tail)
                mask_cache[tail] = allowed
            scores = scores.masked_fill(~allowed.to(scores.device), -float("inf"))
            if not bool(allowed.any()):
                break
        if top_k:
            threshold = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, -float("inf"))
        token = torch.multinomial(scores.softmax(-1), 1, generator=rng)
        token_id = token.item()
        if token_id == boundary:
            break
        ids = torch.cat((ids, token), dim=1)
        if constrain:
            emitted += table[token_id]
            # Only the last bytes can be incomplete: the stream is a valid
            # prefix by construction, so this scans at most 4 bytes.
            tail = incomplete_utf8_tail(bytes(emitted))
        if stop_sequences:
            current = _current_text()
            completion = current[len(prompt):] if current.startswith(prompt) else current
            hits = [completion.find(stop) for stop in stop_sequences if stop and stop in completion]
            if hits:
                stopped_text = prompt + completion[:min(hits)]
                break
        if on_text is not None and (step_i + 1) % max(1, stream_every) == 0:
            try:
                on_text(_current_text())
            except Exception:
                pass
    if on_text is not None:
        try:
            on_text(_current_text())
        except Exception:
            pass
    model.train(was_training)
    if not constrain:
        return tokenizer.decode(ids[0].tolist())
    clean = bytes(emitted)
    if tail:
        clean = clean[:len(clean) - len(tail)]
    return prompt + clean.decode("utf-8")
