"""Thin, provider-agnostic LLM client built on httpx (no vendor SDKs needed).

Supports OpenAI, Anthropic and Gemini REST APIs, retries transient failures with backoff,
optionally falls back to a second provider, and always returns parsed JSON plus token usage.
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import Settings
from app.errors import LLMError, LLMNotConfiguredError, LLMOutputError

logger = logging.getLogger(__name__)

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResult:
    data: dict
    input_tokens: int
    output_tokens: int
    latency_ms: int
    provider: str
    model: str


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and stray prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError("The model did not return JSON.")
        try:
            obj = json.loads(t[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"The model returned malformed JSON ({exc.msg}); output may have been truncated.")
    if not isinstance(obj, dict):
        raise LLMOutputError("The model's JSON was not an object.")
    return obj


class _Provider:
    name = ""

    def __init__(self, model: str, api_key: str, settings: Settings):
        self.model = model
        self.api_key = api_key
        self.settings = settings

    async def _post(self, client: httpx.AsyncClient, url: str, headers: dict, payload: dict) -> dict:
        last = "unknown error"
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                resp = await client.post(url, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = f"network error ({type(exc).__name__})"
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (401, 403):
                    raise LLMError(f"{self.name}: authentication failed - check the API key in your .env file.")
                if resp.status_code in RETRY_STATUS:
                    last = f"HTTP {resp.status_code}"
                else:
                    raise LLMError(f"{self.name} returned HTTP {resp.status_code}: {resp.text[:200]}")
            if attempt < self.settings.llm_max_retries:
                await asyncio.sleep(min(2**attempt, 8))
        raise LLMError(f"{self.name} unavailable after retries ({last}).")

    async def generate(self, client: httpx.AsyncClient, system: str, user: str, max_tokens: int):
        """Return (text, input_tokens, output_tokens)."""
        raise NotImplementedError


class OpenAIProvider(_Provider):
    name = "openai"

    async def generate(self, client, system, user, max_tokens):
        data = await self._post(
            client,
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": max_tokens,
            },
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMOutputError("Unexpected response shape from OpenAI.")
        usage = data.get("usage", {})
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


class AnthropicProvider(_Provider):
    name = "anthropic"

    async def generate(self, client, system, user, max_tokens):
        data = await self._post(
            client,
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


class GeminiProvider(_Provider):
    name = "gemini"

    async def generate(self, client, system, user, max_tokens):
        data = await self._post(
            client,
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": self.api_key},
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "maxOutputTokens": max_tokens,
                },
            },
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            raise LLMOutputError("Gemini returned no content (possibly blocked or truncated).")
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata", {})
        # thinking tokens are billed as output tokens
        out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return text, usage.get("promptTokenCount", 0), out


_PROVIDERS = {"openai": OpenAIProvider, "anthropic": AnthropicProvider, "gemini": GeminiProvider}


def _build(provider: str, model: str, settings: Settings) -> Optional[_Provider]:
    provider = (provider or "").strip().lower()
    if not provider:
        return None
    if provider not in _PROVIDERS:
        raise LLMNotConfiguredError(f"Unknown LLM provider '{provider}'. Use one of: {', '.join(_PROVIDERS)}.")
    key = getattr(settings, f"{provider}_api_key", "")
    return _PROVIDERS[provider](model, key, settings)


class LLMService:
    def __init__(self, settings: Settings, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.settings = settings
        self.transport = transport
        self.primary = _build(settings.llm_provider, settings.llm_model, settings)
        self.fallback = _build(settings.fallback_provider, settings.fallback_model, settings)

    @property
    def provider_name(self) -> str:
        return self.primary.name if self.primary else ""

    @property
    def model_name(self) -> str:
        return self.primary.model if self.primary else ""

    def ensure_configured(self) -> None:
        if not self.primary or not self.primary.api_key:
            raise LLMNotConfiguredError(
                f"No API key configured for LLM provider '{self.settings.llm_provider}'. "
                "Copy .env.example to .env and set the key."
            )

    async def complete_json(self, system: str, user: str, max_tokens: Optional[int] = None) -> LLMResult:
        self.ensure_configured()
        max_tokens = max_tokens or self.settings.llm_max_output_tokens
        providers = [p for p in (self.primary, self.fallback) if p and p.api_key]
        last_error: Optional[LLMError] = None

        for provider in providers:
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.settings.llm_timeout_seconds, connect=10.0),
                    transport=self.transport,
                ) as client:
                    text, tin, tout = await provider.generate(client, system, user, max_tokens)
            except LLMError as exc:
                logger.warning("LLM provider %s failed: %s", provider.name, exc.message)
                last_error = exc
                continue
            latency = int((time.perf_counter() - started) * 1000)
            logger.info("LLM %s/%s: %d in / %d out tokens, %d ms", provider.name, provider.model, tin, tout, latency)
            return LLMResult(extract_json(text), tin, tout, latency, provider.name, provider.model)

        raise last_error or LLMError("No LLM provider available.")
