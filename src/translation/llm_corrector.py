"""
LLM Corrector — configurable multi-provider LLM for sentence correction.
Supports: OpenRouter, OpenAI, Anthropic, Ollama (local), custom endpoints.
"""
import os
import asyncio
import logging
import urllib.parse
import httpx
from typing import Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.OPENROUTER
    api_key: str = ""
    model: str = "meta-llama/llama-3.1-8b-instruct:free"
    base_url: str = ""
    temperature: float = 0.3
    max_tokens: int = 256
    timeout: int = 10

    # Pre-configured provider defaults
    PROVIDER_DEFAULTS = {
        LLMProvider.OPENROUTER: {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "meta-llama/llama-3.1-8b-instruct:free",
        },
        LLMProvider.OPENAI: {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        },
        LLMProvider.ANTHROPIC: {
            "base_url": "https://api.anthropic.com",
            "model": "claude-3-haiku-20240307",
        },
        LLMProvider.OLLAMA: {
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.2:3b",
        },
        LLMProvider.CUSTOM: {
            "base_url": "",
            "model": "",
        },
    }

    def apply_defaults(self):
        defaults = self.PROVIDER_DEFAULTS.get(self.provider, {})
        if not self.base_url:
            self.base_url = defaults.get("base_url", "")
        if not self.model:
            self.model = defaults.get("model", "")

    @staticmethod
    def custom_urls_allowed() -> bool:
        """Operator opt-in for arbitrary/remote endpoints (SSRF guard)."""
        return os.getenv("LLM_ALLOW_CUSTOM_BASE_URL", "").strip().lower() in (
            "1", "true", "yes", "on",
        )

    @staticmethod
    def _host_allowed(host: str) -> bool:
        allowed = {
            h.strip().lower()
            for h in os.getenv("LLM_ALLOWED_HOSTS", "").split(",")
            if h.strip()
        }
        return not allowed or host in allowed

    @staticmethod
    def _is_loopback(host: str) -> bool:
        return host == "localhost" or host.endswith(".localhost") or host in (
            "127.0.0.1", "::1", "0:0:0:0:0:0:0:1",
        )

    def validate(self) -> "LLMConfig":
        """Reject unsafe or incomplete configurations (raises ValueError).

        The POST /llm/config endpoint previously let any caller point the
        server at an arbitrary URL (SSRF). Now:
          * known providers are pinned to their official base_url unless the
            operator explicitly opted in via LLM_ALLOW_CUSTOM_BASE_URL;
          * provider 'custom' requires base_url and that opt-in (or an
            LLM_ALLOWED_HOSTS allow-list match);
          * http is accepted only for loopback hosts (LAN Ollama needs the
            opt-in flag);
          * credentials embedded in the URL are rejected.
        """
        allow_custom = self.custom_urls_allowed()

        if self.base_url:
            parts = urllib.parse.urlsplit(self.base_url)
            if parts.scheme not in ("http", "https"):
                raise ValueError(
                    f"base_url must use http or https, got '{parts.scheme}'"
                )
            if "@" in (parts.netloc or ""):
                raise ValueError("base_url must not embed credentials")
            host = (parts.hostname or "").lower()
            if not host:
                raise ValueError("base_url has no host")
            if parts.scheme == "http" and not self._is_loopback(host):
                if not allow_custom:
                    raise ValueError(
                        "http is only allowed for localhost; use https or set "
                        "LLM_ALLOW_CUSTOM_BASE_URL=true"
                    )
            if not self._host_allowed(host):
                raise ValueError(
                    f"base_url host '{host}' is not in LLM_ALLOWED_HOSTS"
                )

        if self.provider == LLMProvider.CUSTOM:
            if not self.base_url:
                raise ValueError(
                    "provider 'custom' requires base_url (and the "
                    "LLM_ALLOW_CUSTOM_BASE_URL=true opt-in, or an "
                    "LLM_ALLOWED_HOSTS entry)"
                )
            if not allow_custom and not os.getenv("LLM_ALLOWED_HOSTS", "").strip():
                raise ValueError(
                    "custom base_url is disabled: set "
                    "LLM_ALLOW_CUSTOM_BASE_URL=true (or LLM_ALLOWED_HOSTS) "
                    "on the server to opt in"
                )
        else:
            expected = self.PROVIDER_DEFAULTS[self.provider].get("base_url", "")
            if (
                expected
                and self.base_url
                and self.base_url.rstrip("/") != expected.rstrip("/")
            ):
                if not allow_custom:
                    raise ValueError(
                        f"provider '{self.provider.value}' is pinned to "
                        f"{expected}; use provider 'custom' (with "
                        "LLM_ALLOW_CUSTOM_BASE_URL=true) for other endpoints"
                    )
            if not self.base_url:
                raise ValueError(
                    f"provider '{self.provider.value}' requires a base_url"
                )

        if not self.model:
            raise ValueError("model is required")
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 1 <= int(self.max_tokens) <= 32000:
            raise ValueError("max_tokens must be between 1 and 32000")
        if not 1 <= int(self.timeout) <= 300:
            raise ValueError("timeout must be between 1 and 300 seconds")
        if not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string")
        return self

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider_str = os.getenv("LLM_PROVIDER", "openrouter").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            provider = LLMProvider.OPENROUTER

        config = cls(
            provider=provider,
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", ""),
            base_url=os.getenv("LLM_BASE_URL", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "256")),
            timeout=int(os.getenv("LLM_TIMEOUT", "10")),
        )
        config.apply_defaults()
        config.validate()
        return config

    def to_dict(self) -> dict:
        provider_val = self.provider.value if hasattr(self.provider, 'value') else str(self.provider)
        return {
            "provider": provider_val,
            "model": self.model,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


SYSTEM_PROMPT = """You are a Tunisian Sign Language (TSL) sentence corrector.
You receive a raw sentence produced by a TSL recognition system that may contain:
- Misspelled words
- Missing diacritics
- Incorrect letter substitutions
- Incomplete words
- Mixed French/Arabic transliteration

Your task: Return the corrected, clean sentence in the same language (French or Arabic transliteration).
Rules:
- Fix obvious typos and letter substitutions common in sign language recognition
- Preserve the intended meaning
- Keep the same language/script as the input
- Do NOT add explanations, just return the corrected text
- If the input is already correct, return it unchanged"""


class LLMCorrector:
    """Multi-provider LLM corrector for TSL recognition output."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None
        self._enabled = bool(self.config.api_key) or self.config.provider == LLMProvider.OLLAMA

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(self, config: LLMConfig):
        config.validate()
        self.config = config
        self._enabled = bool(config.api_key) or config.provider == LLMProvider.OLLAMA
        old, self._client = self._client, None  # drop old client
        if old is not None and not old.is_closed:
            try:
                asyncio.get_running_loop().create_task(old.aclose())
            except RuntimeError:
                asyncio.run(old.aclose())  # called from a non-async thread

    def get_config(self) -> dict:
        return self.config.to_dict()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.config.provider == LLMProvider.OPENROUTER:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
                headers["HTTP-Referer"] = "https://tsl-translator.local"
            elif self.config.provider == LLMProvider.OPENAI:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            elif self.config.provider == LLMProvider.ANTHROPIC:
                headers["x-api-key"] = self.config.api_key
                headers["anthropic-version"] = "2023-06-01"
            elif self.config.provider == LLMProvider.OLLAMA:
                pass  # No auth needed for local Ollama

            self._client = httpx.AsyncClient(timeout=self.config.timeout, headers=headers)
        return self._client

    async def correct(self, raw_text: str) -> str:
        """Correct a raw TSL recognition output using the configured LLM."""
        if not self._enabled:
            return raw_text
        if not raw_text or not raw_text.strip():
            return raw_text

        try:
            client = await self._get_client()

            if self.config.provider == LLMProvider.ANTHROPIC:
                return await self._call_anthropic(client, raw_text)
            else:
                return await self._call_openai_compatible(client, raw_text)
        except httpx.TimeoutException:
            logger.warning("LLM correction timed out — returning raw text")
            return raw_text
        except Exception as e:
            logger.warning(f"LLM correction failed: {e} — returning raw text")
            return raw_text

    async def _call_openai_compatible(self, client: httpx.AsyncClient, raw_text: str) -> str:
        """Call OpenAI-compatible API (OpenRouter, OpenAI, Ollama, custom)."""
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Correct this TSL sentence:\n\n{raw_text}"},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    async def _call_anthropic(self, client: httpx.AsyncClient, raw_text: str) -> str:
        """Call Anthropic Messages API."""
        url = f"{self.config.base_url.rstrip('/')}/v1/messages"
        payload = {
            "model": self.config.model,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"Correct this TSL sentence:\n\n{raw_text}"},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
