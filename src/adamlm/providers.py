"""Supervisor provider interface for the Research Agent.

External models SUPERVISE Research planning only: they propose one validated
experiment at a time. They never answer Chat (always the local approved
AdamLM) and never train weights. All provider output is untrusted JSON
validated by ``research.validate_proposal`` — never shell commands.

Free-only, zero-cost protections (enforced here, not in callers):
- every supervisor model ID must end in ``:free`` (checked on save AND on
  every call), so fallback can never switch to a paid model;
- ``max_api_cost_usd`` is forced to 0.0 on save;
- API keys live in ``config/.research-api-key`` and are never returned to
  the browser (see ``public_provider_config``).

Only the ``openrouter`` provider is enabled in AdamLM 2.0. The interface
reserves ``groq``, ``unorouter``, ``apinex``, ``xkiro`` and ``ollama``
(local) for later work: adding one means writing a new
``SupervisorProvider`` subclass and registering it in ``_PROVIDERS`` — no
changes to ``research.py`` are needed. Unknown provider IDs are rejected.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from .gui_core import ROOT, atomic_json, read_json

SESSIONS_DIR = "results/research"
PROVIDER_CONFIG = "config/research-provider.json"
PROVIDER_KEY_FILE = "config/.research-api-key"

#: The only supervisor provider enabled in AdamLM 2.0.
KNOWN_PROVIDERS = ("openrouter",)
#: Reserved for later work. Accepted nowhere yet; requesting one raises.
RESERVED_PROVIDERS = ("groq", "unorouter", "apinex", "xkiro", "ollama")
#: Sensible bounded default: sessions stop asking the supervisor after this
#: many HTTP attempts instead of retrying (near-)forever.
DEFAULT_MAX_API_REQUESTS = 25


class ProviderError(ValueError):
    """A single supervisor-model call failed, with a machine-readable kind.

    kinds: rate_limited (per-model 429 — try the next approved model),
    quota_exhausted (account-wide free quota — stop trying, pause),
    auth (bad key — fix config, do not burn the list), unavailable
    (5xx/timeout/network/model-missing — try next), bad_response
    (unusable shape — try next), not_configured, api_cap.
    """

    def __init__(self, message: str, *, kind: str = "unavailable",
                 model: str | None = None, retryable: bool = True):
        super().__init__(message)
        self.kind = kind
        self.model = model
        self.retryable = retryable


class ProviderExhausted(ProviderError):
    """Every approved free model failed (or the cap/quota stopped the walk).

    Carries the per-model attempts so the session can record exactly what
    was tried and pause safely instead of retrying blindly.
    """

    def __init__(self, message: str, *, attempts: list, kind: str = "exhausted"):
        super().__init__(message, kind=kind, retryable=False)
        self.attempts = attempts


class SupervisorProvider:
    """Interface every supervisor backend implements.

    A provider speaks one OpenAI-compatible-style chat API (or, later, a
    local Ollama endpoint) and enforces the free-only gate for every model
    it will call. Research planning calls ``complete_with_fallback``; the
    provider UI calls ``probe``. Nothing here touches training or Chat.
    """

    provider_id: str = "openrouter"
    display_name: str = "OpenRouter"

    def is_free_model(self, model_id: str) -> bool:
        raise NotImplementedError

    def ordered_models(self, config: dict) -> list[str]:
        raise NotImplementedError

    def complete(self, root: Path, *, messages: list[dict], max_tokens: int = 800,
                 temperature: float = 0.2, model: str | None = None) -> str:
        raise NotImplementedError

    def complete_with_fallback(self, root: Path, *, messages: list[dict],
                               max_tokens: int = 800, temperature: float = 0.2,
                               requests_used: int = 0, requests_cap: int = 0) -> dict:
        raise NotImplementedError

    def probe(self, root: Path = ROOT) -> dict:
        raise NotImplementedError


class OpenRouterProvider(SupervisorProvider):
    """The enabled supervisor backend: OpenRouter free models only."""

    provider_id = "openrouter"
    display_name = "OpenRouter"

    def is_free_model(self, model_id: str) -> bool:
        # Free-only gate: OpenRouter free models carry the ':free' suffix.
        # Anything else (paid IDs, empty strings, hosted paths) is refused
        # so the agent can never switch to a paid model or incur charges.
        return bool(model_id) and str(model_id).strip().lower().endswith(":free")

    def ordered_models(self, config: dict) -> list[str]:
        ordered: list[str] = []
        for candidate in [config.get("model")] + list(config.get("fallback_models") or []):
            text = str(candidate or "").strip()
            if text and text not in ordered:
                ordered.append(text)
        return ordered

    def complete(self, root: Path, *, messages: list[dict], max_tokens: int = 800,
                 temperature: float = 0.2, model: str | None = None) -> str:
        return _chat_completion(root, messages=messages, max_tokens=max_tokens,
                                temperature=temperature, model=model)

    def complete_with_fallback(self, root: Path, *, messages: list[dict],
                               max_tokens: int = 800, temperature: float = 0.2,
                               requests_used: int = 0, requests_cap: int = 0) -> dict:
        return _chat_completion_with_fallback(
            root, messages=messages, max_tokens=max_tokens,
            temperature=temperature, requests_used=requests_used,
            requests_cap=requests_cap)

    def probe(self, root: Path = ROOT) -> dict:
        return test_provider_connection(root)


_PROVIDERS: dict[str, SupervisorProvider] = {
    OpenRouterProvider.provider_id: OpenRouterProvider(),
}


def get_provider(provider_id: str | None = None) -> SupervisorProvider:
    """Resolve an enabled supervisor provider. Unknown IDs raise (no guessing)."""
    name = str(provider_id or "openrouter").strip().lower() or "openrouter"
    if name in _PROVIDERS:
        return _PROVIDERS[name]
    if name in RESERVED_PROVIDERS:
        raise ValueError(
            f"Supervisor provider '{name}' is reserved for later work and not "
            "enabled in AdamLM 2.0 (OpenRouter only). No request was made.")
    raise ValueError(
        f"Unknown supervisor provider '{name}'. Enabled: {', '.join(KNOWN_PROVIDERS)}; "
        f"reserved for later: {', '.join(RESERVED_PROVIDERS)}.")


# --------------------------------------------------------------------------
# Provider configuration (keys never leave the backend).
# --------------------------------------------------------------------------

def default_provider_config() -> dict:
    return {
        "enabled": False,
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "",
        # Ordered fallback list: tried in order after the primary model when
        # it is rate-limited or unavailable. FREE MODELS ONLY — every entry
        # must end in ":free" or it is rejected on save and never called.
        "fallback_models": [],
        "allow_external_eval": False,
        "send_categories": ["eval-summaries", "dataset-summaries", "failure-labels"],
        "max_api_requests": DEFAULT_MAX_API_REQUESTS,
        "max_api_cost_usd": 0.0,
        "timeout_seconds": 60,
    }


def is_free_model_id(model_id: str) -> bool:
    """Free-only gate, delegated to the enabled provider (OpenRouter rules)."""
    return _PROVIDERS["openrouter"].is_free_model(model_id)


def supervisor_models(config: dict) -> list[str]:
    """Ordered, de-duplicated supervisor model list: primary first."""
    return get_provider(config.get("provider")).ordered_models(config)


def _normalize_model_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n,]+", value)
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError("Fallback models must be a list or newline-separated text.")
    cleaned = [str(p).strip() for p in parts if str(p or "").strip()]
    if len(cleaned) > 8:
        raise ValueError("At most 8 fallback models are supported.")
    return cleaned


def _normalize_api_cap(value) -> int:
    try:
        cap = int(value)
    except (TypeError, ValueError):
        raise ValueError("Maximum API requests must be a whole number.") from None
    if cap <= 0:
        raise ValueError("Maximum API requests must stay positive (bounded operation).")
    if cap > 1000:
        raise ValueError("Maximum API requests must stay bounded (at most 1000).")
    return cap


def read_provider_config(root: Path = ROOT) -> dict:
    merged = default_provider_config()
    stored = read_json(Path(root) / PROVIDER_CONFIG, {})
    if isinstance(stored, dict):
        for key in merged:
            if key in stored:
                merged[key] = stored[key]
    return merged


def write_provider_config(root: Path, values: dict) -> dict:
    """Store non-secret provider settings. The API key is handled separately."""
    root = Path(root)
    current = read_provider_config(root)
    if "provider" in values:
        get_provider(values.get("provider"))  # rejects unknown/reserved IDs
        current["provider"] = str(values.get("provider") or "openrouter").strip().lower()
    for key in ("enabled", "base_url", "model", "allow_external_eval",
                "send_categories", "timeout_seconds"):
        if key in values:
            current[key] = values[key]
    if "max_api_requests" in values:
        current["max_api_requests"] = _normalize_api_cap(values["max_api_requests"])
    if "max_api_cost_usd" in values:
        current["max_api_cost_usd"] = values["max_api_cost_usd"]
    if "fallback_models" in values:
        current["fallback_models"] = _normalize_model_list(values["fallback_models"])
    current["enabled"] = bool(current.get("enabled"))
    current["allow_external_eval"] = bool(current.get("allow_external_eval"))
    # Free-only operation: every configured supervisor model must be a free
    # model, and the cost cap must stay at zero. Paid IDs are rejected here —
    # and re-checked on every call — so fallback can never incur charges.
    for model_id in supervisor_models(current):
        if not is_free_model_id(model_id):
            raise ValueError(
                f"Refusing non-free supervisor model '{model_id}': "
                "the research agent is free-models-only (IDs must end in ':free').")
    try:
        cost_cap = float(current.get("max_api_cost_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        raise ValueError("Maximum API cost must be a number.") from None
    if cost_cap != 0.0:
        raise ValueError("Maximum API cost must stay 0 (free-models-only operation).")
    current["max_api_cost_usd"] = 0.0
    atomic_json(root / PROVIDER_CONFIG, current)
    return public_provider_config(root)


def public_provider_config(root: Path = ROOT) -> dict:
    """Provider settings safe to send to the browser (key status only)."""
    config = read_provider_config(root)
    key_path = Path(root) / PROVIDER_KEY_FILE
    config["key_configured"] = bool(key_path.is_file() and key_path.stat().st_size > 0)
    return config


def store_api_key(root: Path, key: str) -> dict:
    key = (key or "").strip()
    if not key:
        raise ValueError("API key is empty.")
    path = Path(root) / PROVIDER_KEY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key, encoding="utf-8")
    try:
        import os

        os.chmod(path, 0o600)
    except OSError:
        pass
    return {"key_configured": True}


def clear_api_key(root: Path) -> dict:
    try:
        (Path(root) / PROVIDER_KEY_FILE).unlink()
    except OSError:
        pass
    return {"key_configured": False}


def _read_api_key(root: Path) -> str:
    try:
        return (Path(root) / PROVIDER_KEY_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def test_provider_connection(root: Path = ROOT) -> dict:
    """Tiny unauthenticated-shape-checked probe: GET <base>/models."""
    config = read_provider_config(root)
    get_provider(config.get("provider"))  # unknown/reserved IDs never reach the network
    base = str(config.get("base_url") or "").rstrip("/")
    if not base:
        raise ValueError("Provider base URL is not configured.")
    key = _read_api_key(root)
    if not key:
        raise ValueError("No API key is stored. Add one before testing.")
    request = urllib.request.Request(
        f"{base}/models",
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=min(30, float(config.get("timeout_seconds") or 30))) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except Exception as exc:
        raise ValueError(f"Provider connection failed: {exc}") from None
    models = [m.get("id") for m in (payload.get("data") or []) if isinstance(m, dict)][:50]
    configured = str(config.get("model") or "")
    checked = [{"model": mid, "listed": mid in models, "free": is_free_model_id(mid)}
               for mid in supervisor_models(config)]
    return {"ok": True, "model_count": len(models),
            "configured_model_listed": configured in models if configured else None,
            "checked": checked,
            "models": models[:20]}


_QUOTA_PATTERNS = re.compile(
    r"free.models.per.day|daily[^.]{0,40}free|free[^.]{0,40}daily|account[^.]{0,40}quota|"
    r"quota[^.]{0,40}exhausted|insufficient[^.]{0,40}credit|payment required|"
    r"upgrade[^.]{0,40}(plan|account)|billing",
    re.IGNORECASE,
)


def _classify_http_error(code: int, body: str) -> str:
    if code == 401:
        return "auth"
    if code == 402:
        return "quota_exhausted"
    if code == 429:
        return "quota_exhausted" if _QUOTA_PATTERNS.search(body or "") else "rate_limited"
    if code == 404:
        return "unavailable"
    return "unavailable"


def _chat_completion(root: Path, *, messages: list[dict], max_tokens: int = 800,
                     temperature: float = 0.2, model: str | None = None) -> str:
    """One completion from one explicit supervisor model (free-only)."""
    config = read_provider_config(root)
    get_provider(config.get("provider"))  # unknown/reserved IDs never reach the network
    base = str(config.get("base_url") or "").rstrip("/")
    model_id = str(model if model is not None else config.get("model") or "").strip()
    key = _read_api_key(root)
    if not base or not model_id or not key:
        raise ProviderError(
            "Provider is not fully configured (base URL, model, and API key are all required).",
            kind="not_configured", model=model_id or None, retryable=False)
    if not is_free_model_id(model_id):
        raise ProviderError(
            f"Refusing non-free supervisor model '{model_id}' (free-models-only operation).",
            kind="not_configured", model=model_id, retryable=False)
    body = json.dumps({
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(config.get("timeout_seconds") or 60)) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", "replace")
        except Exception:
            err_body = ""
        kind = _classify_http_error(exc.code, err_body)
        detail = err_body.strip()[:300]
        raise ProviderError(
            f"Supervisor model '{model_id}' failed (HTTP {exc.code}"
            f"{': ' + detail if detail else ''}).",
            kind=kind, model=model_id,
            retryable=kind not in ("auth", "quota_exhausted", "not_configured")) from None
    except Exception as exc:
        raise ProviderError(
            f"Supervisor model '{model_id}' unreachable ({exc}).",
            kind="unavailable", model=model_id, retryable=True) from None
    if isinstance(payload, dict) and payload.get("error"):
        err_text = str(payload["error"].get("message") if isinstance(payload["error"], dict)
                       else payload["error"])[:300]
        kind = ("quota_exhausted" if _QUOTA_PATTERNS.search(err_text) else "unavailable")
        raise ProviderError(
            f"Supervisor model '{model_id}' returned an error"
            f"{': ' + err_text if err_text else ''}.",
            kind=kind, model=model_id,
            retryable=kind not in ("quota_exhausted",))
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProviderError(
            f"Supervisor model '{model_id}' returned an unusable response shape.",
            kind="bad_response", model=model_id, retryable=True) from None


def _chat_completion_with_fallback(root: Path, *, messages: list[dict],
                                   max_tokens: int = 800, temperature: float = 0.2,
                                   requests_used: int = 0, requests_cap: int = 0) -> dict:
    """Walk the ordered approved free-model list until one answers.

    Returns {"text", "model", "attempts", "api_requests"} where attempts is
    the per-model audit trail [{model, ok, kind, detail}]. Raises
    ProviderExhausted (with .attempts) when every model fails, the
    account-wide quota is hit, auth fails, or the API request cap is
    reached. Never touches a paid model: the free gate is enforced per call.
    """
    config = read_provider_config(root)
    get_provider(config.get("provider"))  # unknown/reserved IDs never reach the network
    models = supervisor_models(config)
    if not models:
        raise ProviderExhausted("No supervisor models are configured.",
                                attempts=[], kind="not_configured")
    attempts: list[dict] = []
    made = 0
    for model_id in models:
        if requests_cap > 0 and requests_used + made >= requests_cap:
            raise ProviderExhausted(
                f"API request cap reached after {made} attempt(s); "
                f"model '{model_id}' and later models were not tried.",
                attempts=attempts, kind="api_cap")
        try:
            text = _chat_completion(root, messages=messages, max_tokens=max_tokens,
                                    temperature=temperature, model=model_id)
        except ProviderError as exc:
            made += 1
            attempts.append({"model": model_id, "ok": False, "kind": exc.kind,
                             "detail": str(exc)[:300]})
            if exc.kind in ("auth", "not_configured"):
                raise ProviderExhausted(
                    f"Supervisor auth/config problem at '{model_id}' ({exc}); "
                    "not trying further models with the same credentials.",
                    attempts=attempts, kind=exc.kind) from None
            if exc.kind == "quota_exhausted":
                raise ProviderExhausted(
                    f"Account-wide free quota exhausted at '{model_id}' ({exc}); "
                    "pausing instead of burning the remaining list.",
                    attempts=attempts, kind="quota_exhausted") from None
            continue  # rate_limited / unavailable / bad_response -> next model
        made += 1
        attempts.append({"model": model_id, "ok": True, "kind": "ok", "detail": ""})
        return {"text": text, "model": model_id, "attempts": attempts,
                "api_requests": made}
    kinds = sorted({a.get("kind") for a in attempts})
    raise ProviderExhausted(
        f"All {len(models)} approved free model(s) failed "
        f"({', '.join(kinds)}); pausing at this AI decision.",
        attempts=attempts, kind="exhausted")
