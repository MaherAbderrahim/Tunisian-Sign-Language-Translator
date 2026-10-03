"""
LLM Corrector — configurable multi-provider LLM for sentence correction.
Supports: OpenRouter, OpenAI, Anthropic, Ollama (local), custom endpoints.
"""
import os
import json
import logging
import httpx
from typing import Optional
from dataclasses import dataclass, field
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
        self.config = config
        self._enabled = bool(config.api_key) or config.provider == LLMProvider.OLLAMA
        if self._client:
            self._client = None  # Reset client on reconfigure

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
