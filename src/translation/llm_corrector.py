"""
LLM Corrector — configurable multi-provider LLM for sentence correction.
Supports: OpenRouter, OpenAI, Anthropic, Ollama (local), custom endpoints.

Config precedence (highest wins):
  1. saved runtime config  (data/llm_config.json, or $TSL_LLM_CONFIG_PATH)
     — written by POST /llm/config so a user selection survives restarts;
  2. environment           (.env / LLM_* variables);
  3. provider defaults     (PROVIDER_DEFAULTS).
"""
import os
import re
import json
import asyncio
import logging
import urllib.parse
import httpx
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "llm_config.json"


def saved_config_path() -> Path:
    """Where the runtime LLM config is persisted (override for tests)."""
    override = (os.getenv("TSL_LLM_CONFIG_PATH") or "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


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
    model: str = ""          # empty -> apply_defaults() fills the provider's default
    base_url: str = ""
    temperature: float = 0.3
    max_tokens: int = 256
    timeout: int = 10

    # Pre-configured provider defaults
    PROVIDER_DEFAULTS = {
        LLMProvider.OPENROUTER: {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openrouter/free",
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

    def base_url_host(self) -> str:
        if not self.base_url:
            return ""
        return (urllib.parse.urlsplit(self.base_url).hostname or "").lower()

    def is_local_endpoint(self) -> bool:
        """True when the LLM is reached on this machine (Ollama, LM Studio,
        llama.cpp, vLLM, …). Local endpoints need no API key and are not an
        SSRF concern."""
        if self.provider == LLMProvider.OLLAMA:
            return True
        host = self.base_url_host()
        return bool(host) and self._is_loopback(host)

    @property
    def usable(self) -> bool:
        """True when a call can actually be attempted: either an API key is
        configured or the endpoint runs locally (Ollama / LM Studio / …)."""
        return bool(self.api_key) or self.is_local_endpoint()

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
            # A loopback endpoint is under the user's own control, so it is
            # allowed without the SSRF opt-in: that is how a local model
            # (LM Studio / llama.cpp / vLLM) gets selected.
            if (
                not self.is_local_endpoint()
                and not allow_custom
                and not os.getenv("LLM_ALLOWED_HOSTS", "").strip()
            ):
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
        provider_str = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        if not provider_str:
            provider_str = LLMProvider.OPENROUTER.value
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            # Never coerce silently: an involuntary OpenRouter fallback is
            # exactly the bug this warning makes visible in the logs.
            logger.warning(
                "Unknown LLM_PROVIDER '%s' — falling back to '%s'. Valid "
                "providers: %s",
                provider_str, LLMProvider.OPENROUTER.value,
                ", ".join(p.value for p in LLMProvider),
            )
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
            "enabled": self.usable,
            "is_local": self.is_local_endpoint(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def load_saved_config() -> Optional[LLMConfig]:
    """Load the runtime config saved by POST /llm/config (if any).

    Returns None when the file is absent/invalid/unsafe so the caller can
    fall back to the environment — never raises.
    """
    path = saved_config_path()
    try:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config file is not a JSON object")
        config = LLMConfig(
            provider=LLMProvider(str(raw.get("provider") or "").strip().lower()),
            api_key=str(raw.get("api_key") or ""),
            model=str(raw.get("model") or ""),
            base_url=str(raw.get("base_url") or ""),
            temperature=float(raw.get("temperature", 0.3)),
            max_tokens=int(raw.get("max_tokens", 256)),
            timeout=int(raw.get("timeout", 10)),
        )
        config.apply_defaults()
        config.validate()
        logger.info(f"Loaded saved LLM config from {path}: "
                    f"{config.provider.value}/{config.model}")
        return config
    except Exception as e:
        logger.warning(f"Ignoring saved LLM config ({path}): {e}")
        return None


def save_config(config: LLMConfig) -> bool:
    """Persist the runtime config so the selected model survives restarts."""
    path = saved_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": config.provider.value
            if hasattr(config.provider, "value") else str(config.provider),
            "api_key": config.api_key,
            "model": config.model,
            "base_url": config.base_url,
            "temperature": float(config.temperature),
            "max_tokens": int(config.max_tokens),
            "timeout": int(config.timeout),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        logger.warning(f"Could not persist LLM config to {path}: {e}")
        return False



# User-message instructions (the system prompt carries the full task).
DEFAULT_INSTRUCTION = "Correct this TSL sentence:"
COORDINATE_INSTRUCTION = (
    "Rebuild this detected sign sequence as ONE natural sentence "
    "(reconstruction, not a summary):"
)

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")

# A one-sentence answer is short and single-paragraph. Free models sometimes
# answer with their reasoning trace instead of the sentence.
MAX_REPLY_CHARS = 600
_THINKING_PREAMBLE = re.compile(
    r"^\s*(here'?s\s+(a|the|my)\b|let'?s?\s+(think|analyze|break|start)|"
    r"let\s+me\s+(think|analyze|start)|thinking\s+process|"
    r"sure[!,.]?\s+here'?s|i'?ll\s+(think|analyze|start)|"
    r"\*\*|step\s*1\b)",
    re.IGNORECASE,
)


def is_plausible_reply(content: str) -> bool:
    """Structural sanity check shared by every LLM call.

    Rejects empty answers, reasoning traces ("Here's a thinking process: ...")
    and anything that cannot be a single corrected sentence.
    """
    text = (content or "").strip()
    if not text or len(text) > MAX_REPLY_CHARS:
        return False
    if "\n\n" in text or "\r\n\r\n" in text:
        return False
    if _THINKING_PREAMBLE.match(text):
        return False
    return True


def _content_words(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) >= 3}


def relates_to_input(content: str, raw_text: str) -> bool:
    """True when the reply is actually about the input.

    ``openrouter/free`` rotates over free models and sometimes routes to
    non-chat endpoints such as ``nvidia/nemotron-3.5-content-safety:free``,
    which always answers ``"User Safety: safe"``: no shared word, not a
    sentence.
    """
    text = (content or "").strip()
    if not text:
        return False
    if _content_words(text) & _content_words(raw_text):
        return True
    return text[-1] in ".!?"


def is_usable_reply(content: str, raw_text: str) -> bool:
    """``is_plausible_reply`` + the reply must relate to the input."""
    return is_plausible_reply(content) and relates_to_input(content, raw_text)


def rate_limit_retry_delay(
    error: "httpx.HTTPStatusError", default: float
) -> Optional[float]:
    """Delay before retrying a 429, or ``None`` when waiting is pointless.

    OpenRouter's free tier exposes ``free-models-per-day`` (50 requests/day):
    when ``x-ratelimit-remaining`` is 0 the quota resets hours later, so
    retrying would only add latency to the user and waste the remaining
    attempts. A burst (remaining > 0) is worth retrying with a short backoff.
    """
    resp = error.response
    if resp is None:
        return default
    remaining = resp.headers.get("x-ratelimit-remaining")
    if remaining is not None:
        try:
            if int(remaining) <= 0:
                return None
        except ValueError:
            pass
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            return default
        return seconds if seconds <= 60 else None
    return default


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

COORDINATE_SYSTEM_PROMPT = """You are a Tunisian Sign Language (TSL) interpreter.
You receive a sequence of words recognized from signing: raw glosses (e.g. "moi aimer tunisie"),
a rough chain stitched with connectors (e.g. "taxi and Tunis"), or a template sentence
(e.g. "They say: bonjour"). The sequence is often badly ordered, incomplete or repetitive.

Your task is RECONSTRUCTION, not summarization: rebuild ONE natural, grammatically correct
sentence in {language} that stays faithful to the detected sequence.

Rules:
- Keep the meaning and the intention of every detected word.
- Fix grammar, agreement, articles and word order so the sentence sounds natural in {language}.
- You may add only the function words {language} itself requires (articles, prepositions,
  pronouns, auxiliaries). Never add new facts, people, places, times or details.
- Do NOT summarize, shorten to an idea, paraphrase away content, explain, comment or
  describe what the person signed. Return ONLY the final sentence.
- Drop scaffolding: connectors repeated between every word ("and", "et", "w"), template
  prefixes ("They say:", "Ils disent :"), and words repeated back-to-back. Keep a
  repetition when it carries meaning (emphasis, intensifier, a real list).
- Keep names, numbers, negations and proper nouns exactly as given.
- Short or incomplete sequences are normal: build the best faithful sentence from the
  words you have. If nothing usable remains, return an empty answer.
- Write the sentence in {language} only, in its usual script.

{examples}
"""

# Few-shot examples per output language. They encode "reconstruct, don't summarize".
_COORD_EXAMPLES = {
    "French": """Examples (input -> output):
moi aimer tunisie -> J'aime la Tunisie.
toi aller où -> Où vas-tu ?
moi vouloir apprendre français -> Je veux apprendre le français.
bonjour bonjour comment ça va -> Bonjour, comment ça va ?""",
    "English": """Examples (input -> output):
I love Tunisia I love Tunisia -> I love Tunisia.
where you go -> Where are you going?
me want learn French -> I want to learn French.
hello hello how are you -> Hello, how are you?""",
    "Arabic": """Example (input -> output):
أنا أحب تونس أحب تونس -> أحب تونس.""",
}

_LANG_NAMES = {
    "en": "English",
    "english": "English",
    "fr": "French",
    "french": "French",
    "ar": "Arabic",
    "arabic": "Arabic",
    "arabic_msa": "Arabic",
    "msa": "Arabic",
    "tunisian": "Tunisian Arabic (Latin script, derja)",
}


def coordinate_prompt(target_lang: str) -> str:
    """Build the coordination system prompt for a target language.

    Unknown/empty languages fall back to English with a warning — the caller
    (API) validates target_lang up-front so this should not happen silently.
    """
    key = (target_lang or "").strip().lower()
    lang_name = _LANG_NAMES.get(key)
    if lang_name is None:
        logger.warning(f"Unknown target language '{target_lang}' — using English")
        lang_name = _LANG_NAMES["en"]
    examples = _COORD_EXAMPLES.get(lang_name, "")
    return COORDINATE_SYSTEM_PROMPT.replace("{language}", lang_name).replace(
        "{examples}", examples
    )



class LLMCorrector:
    """Multi-provider LLM corrector for TSL recognition output."""

    def __init__(self, config: Optional[LLMConfig] = None):
        # Precedence: explicit config > saved runtime config > environment.
        if config is not None:
            config.apply_defaults()   # an explicit config may omit provider defaults
            self.config = config
        else:
            self.config = load_saved_config() or LLMConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None
        self._enabled = self.config.usable

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(self, config: LLMConfig):
        config.validate()
        self.config = config
        self._enabled = config.usable
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
            elif self.config.provider == LLMProvider.CUSTOM and self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

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

    async def coordinate(self, raw_text: str, target_lang: str = "en") -> str:
        """Coordinate a raw gloss chain into ONE natural sentence (LLM).

        Unlike correct() (typo fixing, same language), this turns the noisy
        "word and word and word" chain produced by sentence assembly into a
        grammatically coordinated sentence in ``target_lang``.

        Always degrades gracefully: returns ``raw_text`` when the LLM is
        disabled, times out, or fails (including free-tier 429 rate limits —
        one short retry is attempted before falling back).
        """
        if not self._enabled:
            return raw_text
        if not raw_text or not raw_text.strip():
            return raw_text

        system = coordinate_prompt(target_lang)
        # Free-tier pools throw 429 bursts when several sentences are
        # coordinated in parallel — retry with a growing backoff before
        # falling back to the raw text.
        backoff = (2.0, 5.0, 8.0)
        attempts = len(backoff) + 1
        for attempt in range(1, attempts + 1):
            try:
                client = await self._get_client()
                if self.config.provider == LLMProvider.ANTHROPIC:
                    out = await self._call_anthropic(
                        client, raw_text, system=system,
                        instruction=COORDINATE_INSTRUCTION,
                    )
                else:
                    out = await self._call_openai_compatible(
                        client, raw_text, system=system,
                        instruction=COORDINATE_INSTRUCTION,
                        # a coordination must be a sentence about this input;
                        # reject stub replies from non-chat routed models
                        check_relation=True,
                    )
                if out and out.strip():
                    return out.strip()
                raise ValueError("empty LLM response")
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 429 and attempt < attempts:
                    delay = rate_limit_retry_delay(e, backoff[attempt - 1])
                    if delay is None:
                        logger.warning(
                            "LLM daily rate limit exhausted — returning raw text"
                        )
                        return raw_text
                    await asyncio.sleep(delay)
                    continue
                logger.warning(f"LLM coordination failed: {e} — returning raw text")
                return raw_text
            except httpx.TimeoutException:
                if attempt < attempts:
                    await asyncio.sleep(backoff[attempt - 1])
                    continue
                logger.warning("LLM coordination timed out — returning raw text")
                return raw_text
            except Exception as e:
                logger.warning(f"LLM coordination failed: {e} — returning raw text")
                return raw_text
        return raw_text

    async def _post_variants(
        self,
        client: httpx.AsyncClient,
        url: str,
        variants: list,
        raw_text: str = "",
        check_relation: bool = False,
    ) -> str:
        """POST chat payload variants until one yields usable content.

        Free-tier OpenRouter routing is endpoint-dependent:
        * some providers reject ``reasoning`` with a 400
          ("Reasoning is mandatory ... cannot be disabled"),
        * reasoning models can burn the whole ``max_tokens`` budget on
          thinking and return empty content,
        * some routed endpoints are not chat models at all and answer with a
          stub (``"User Safety: safe"``).

        All three used to abort the call and silently return the raw text (or
        worse, the stub), so the configured LLM looked like it was "never
        used". Walk the variants (400 -> next, empty/unusable -> next) before
        giving up.
        """
        last_error: Optional[Exception] = None
        saw_unusable = False
        for i, payload in enumerate(variants):
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                last_error = e
                if status == 400 and i + 1 < len(variants):
                    logger.info(
                        f"LLM endpoint rejected payload (400) — trying variant "
                        f"{i + 1}/{len(variants)}"
                    )
                    continue
                raise

            try:
                data = resp.json()
                content = (
                    data["choices"][0]["message"].get("content") or ""
                ).strip()
                finish = data["choices"][0].get("finish_reason")
            except (ValueError, KeyError, IndexError, TypeError) as e:
                raise ValueError(f"malformed LLM response: {e}") from e

            usable = is_plausible_reply(content)
            if usable and check_relation:
                usable = relates_to_input(content, raw_text)
            if usable:
                logger.debug(
                    "LLM reply from model=%s provider=%s",
                    data.get("model"), data.get("provider"),
                )
                return content

            reason = "empty" if not content else f"unusable ({content[:40]!r})"
            saw_unusable = True
            if i + 1 < len(variants):
                logger.info(
                    f"LLM returned {reason} content (finish_reason={finish}) — "
                    f"trying variant {i + 1}/{len(variants)}"
                )

        if saw_unusable:
            raise ValueError("empty or unusable content from LLM")
        raise last_error or RuntimeError("unreachable")  # pragma: no cover

    def _openai_variants(self, raw_text: str, system: str,
                         instruction: Optional[str]) -> list:
        """Ordered payload variants for an OpenAI-compatible endpoint."""
        budget = self.config.max_tokens
        base = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"{instruction or DEFAULT_INSTRUCTION}\n\n{raw_text}",
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": budget,
        }
        variants = [base]
        if self.config.provider == LLMProvider.OPENROUTER:
            # Free-tier models default to reasoning/thinking, which burns the
            # whole max_tokens budget and returns empty content. Disable it —
            # but some endpoints forbid disabling, hence the plain fallback.
            variants.insert(0, {**base, "reasoning": {"effort": "none"}})
        # Thinking models may need far more than max_tokens to emit an answer.
        bigger = min(max(budget * 4, 2048), 8192)
        if bigger > budget:
            variants.append({**variants[-1], "max_tokens": bigger})
        return variants

    async def _call_openai_compatible(
        self,
        client: httpx.AsyncClient,
        raw_text: str,
        system: str = SYSTEM_PROMPT,
        instruction: Optional[str] = None,
        check_relation: bool = False,
    ) -> str:
        """Call OpenAI-compatible API (OpenRouter, OpenAI, Ollama, custom)."""
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        return await self._post_variants(
            client,
            url,
            self._openai_variants(raw_text, system, instruction),
            raw_text=raw_text,
            check_relation=check_relation,
        )

    async def _call_anthropic(
        self,
        client: httpx.AsyncClient,
        raw_text: str,
        system: str = SYSTEM_PROMPT,
        instruction: Optional[str] = None,
    ) -> str:
        """Call Anthropic Messages API."""
        url = f"{self.config.base_url.rstrip('/')}/v1/messages"
        payload = {
            "model": self.config.model,
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": f"{instruction or DEFAULT_INSTRUCTION}\n\n{raw_text}",
                },
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
