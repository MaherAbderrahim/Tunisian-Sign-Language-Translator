"""End-to-end tests for the LLM model-selection chain.

Covers: Settings -> selected model -> configuration -> pipeline -> LLM call ->
response -> display.

Regression target: selecting a local model used to leave the system on
``openrouter/free`` because
  * ``LLMConfig.model`` defaulted to ``"openrouter/free"`` for every provider,
  * ``provider 'custom'`` (local servers: LM Studio / llama.cpp / vLLM) was
    rejected without an SSRF opt-in flag, even for 127.0.0.1,
  * a partial ``POST /llm/config`` reset the provider to ``openrouter`` and
    wiped the stored API key,
  * the selection lived in RAM only, so every restart fell back to `.env`.
"""

import sys
import os
import json
import asyncio
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pytest
import httpx

from src.translation.llm_corrector import (
    LLMConfig,
    LLMCorrector,
    LLMProvider,
    coordinate_prompt,
    load_saved_config,
    save_config,
    saved_config_path,
)


# ── helpers ─────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [
                {"message": {"content": self._content}, "finish_reason": "stop"}
            ]
        }


class _RecordingClient:
    """Stands in for httpx.AsyncClient and records every outgoing payload."""

    def __init__(self, content="Reconstructed.", url_error=None):
        self.content = content
        self.url_error = url_error
        self.calls = []

    async def post(self, url, json=None, **kwargs):
        self.calls.append({"url": url, "payload": json})
        if self.url_error is not None:
            raise self.url_error
        return _FakeResponse(self.content)

    @property
    def last_payload(self):
        return self.calls[-1]["payload"] if self.calls else None


def _use_client(corrector, client_obj):
    async def _get():
        return client_obj

    corrector._get_client = _get


def _local_config(model="qwen2.5:3b", provider=LLMProvider.OLLAMA):
    config = LLMConfig(provider=provider, api_key="", model=model, base_url="")
    config.apply_defaults()
    return config.validate()


# ── 1. config defaults: no cross-provider leakage ───────────────────────

class TestConfigDefaults:
    def test_dataclass_model_default_is_empty(self):
        assert LLMConfig().model == ""

    def test_openrouter_default_is_documented_value(self):
        config = LLMConfig(provider=LLMProvider.OPENROUTER)
        config.apply_defaults()
        assert config.model == "openrouter/free"

    def test_local_provider_never_inherits_openrouter_model(self):
        """The historical bug: LLMConfig(provider=...) kept "openrouter/free"."""
        for provider, expected in (
            (LLMProvider.OLLAMA, "llama3.2:3b"),
            (LLMProvider.OPENAI, "gpt-4o-mini"),
            (LLMProvider.ANTHROPIC, "claude-3-haiku-20240307"),
        ):
            config = LLMConfig(provider=provider)
            config.apply_defaults()
            assert config.model == expected, provider
            assert config.model != "openrouter/free"

    def test_explicit_model_is_never_overwritten(self):
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="qwen2.5:3b")
        config.apply_defaults()
        assert config.model == "qwen2.5:3b"

    def test_from_env_unknown_provider_warns(self, caplog, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
        with caplog.at_level(logging.WARNING, logger="src.translation.llm_corrector"):
            config = LLMConfig.from_env()
        assert config.provider == LLMProvider.OPENROUTER
        messages = [r.getMessage() for r in caplog.records]
        assert any("LLM_PROVIDER" in m for m in messages)


# ── 2. local endpoints are selectable ───────────────────────────────────

class TestLocalEndpointSelection:
    def test_local_custom_endpoint_accepted_without_optin(self):
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            api_key="",
            model="qwen2.5-7b-instruct",
            base_url="http://localhost:1234/v1",
        )
        config.validate()
        assert config.is_local_endpoint()
        assert config.usable  # local endpoints need no API key

    def test_local_custom_endpoint_with_127_0_0_1(self):
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="m",
            base_url="http://127.0.0.1:8080/v1",
        )
        config.validate()
        assert config.is_local_endpoint()

    def test_remote_custom_endpoint_still_rejected(self):
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="m",
            base_url="https://evil.example/v1",
        )
        with pytest.raises(ValueError, match="LLM_ALLOW_CUSTOM_BASE_URL"):
            config.validate()

    def test_ollama_is_usable_without_key(self):
        config = _local_config()
        assert LLMCorrector(config).enabled is True

    def test_cloud_provider_without_key_is_disabled(self):
        config = LLMConfig(provider=LLMProvider.OPENROUTER, api_key="")
        config.apply_defaults()
        config.validate()
        assert LLMCorrector(config).enabled is False


# ── 3. the selection survives a restart ─────────────────────────────────

class TestPersistence:
    @pytest.fixture(autouse=True)
    def _isolated_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "TSL_LLM_CONFIG_PATH", str(tmp_path / "llm_config.json")
        )

    def test_saved_config_is_reloaded(self):
        config = _local_config(model="qwen2.5:7b")
        assert save_config(config) is True
        assert saved_config_path().is_file()

        reloaded = load_saved_config()
        assert reloaded is not None
        assert reloaded.provider == LLMProvider.OLLAMA
        assert reloaded.model == "qwen2.5:7b"

    def test_saved_config_beats_environment(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("LLM_MODEL", "openrouter/free")
        monkeypatch.setenv("LLM_API_KEY", "sk-env")
        save_config(_local_config(model="qwen2.5:7b"))

        corrector = LLMCorrector()
        assert corrector.config.provider == LLMProvider.OLLAMA
        assert corrector.config.model == "qwen2.5:7b"
        assert corrector.enabled is True

    def test_no_saved_config_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "llama3.2:3b")
        assert load_saved_config() is None
        corrector = LLMCorrector()
        assert corrector.config.provider == LLMProvider.OLLAMA
        assert corrector.config.model == "llama3.2:3b"

    def test_corrupt_saved_config_is_ignored(self):
        saved_config_path().parent.mkdir(parents=True, exist_ok=True)
        saved_config_path().write_text("{not json", encoding="utf-8")
        assert load_saved_config() is None

    def test_unsafe_saved_config_is_ignored(self):
        saved_config_path().write_text(
            json.dumps(
                {
                    "provider": "custom",
                    "model": "m",
                    "base_url": "https://evil.example/v1",
                }
            ),
            encoding="utf-8",
        )
        assert load_saved_config() is None


# ── 4. the model in Settings is the model actually called ───────────────

class TestSelectedModelIsUsed:
    def test_local_model_is_the_one_sent(self):
        corrector = LLMCorrector(_local_config(model="qwen2.5:3b"))
        client = _RecordingClient(content="Je vais en Tunisie.")
        _use_client(corrector, client)

        out = asyncio.run(corrector.coordinate("moi aller tunisie", "fr"))
        assert out == "Je vais en Tunisie."
        payload = client.last_payload
        assert payload["model"] == "qwen2.5:3b"
        assert payload["model"] != "openrouter/free"
        assert client.calls[0]["url"].startswith("http://localhost:11434/v1/")

    def test_custom_local_model_is_the_one_sent(self):
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            api_key="",
            model="local-test-model",
            base_url="http://localhost:1234/v1",
        )
        config.apply_defaults()
        corrector = LLMCorrector(config.validate())
        client = _RecordingClient()
        _use_client(corrector, client)

        asyncio.run(corrector.coordinate("hello", "en"))
        assert client.last_payload["model"] == "local-test-model"
        assert client.calls[0]["url"].startswith("http://localhost:1234/v1/")

    def test_openrouter_model_is_sent_when_openrouter_selected(self):
        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="sk-test", model="")
        )
        corrector.config.apply_defaults()
        client = _RecordingClient()
        _use_client(corrector, client)

        asyncio.run(corrector.coordinate("hello", "en"))
        assert client.last_payload["model"] == "openrouter/free"
        assert client.calls[0]["url"].startswith("https://openrouter.ai/api/v1/")


# ── 5. graceful behaviour when the local model is unavailable ───────────

class TestLocalModelUnavailable:
    def test_unreachable_local_endpoint_falls_back_to_raw(self):
        import httpx

        corrector = LLMCorrector(_local_config(model="qwen2.5:3b"))
        client = _RecordingClient(
            url_error=httpx.ConnectError("connection refused")
        )
        _use_client(corrector, client)

        out = asyncio.run(corrector.coordinate("moi aimer tunisie", "fr"))
        assert out == "moi aimer tunisie"  # raw text, no crash

    def test_disabled_llm_never_calls_the_network(self):
        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="")
        )
        client = _RecordingClient()
        _use_client(corrector, client)
        out = asyncio.run(corrector.coordinate("moi aimer tunisie", "fr"))
        assert out == "moi aimer tunisie"
        assert client.calls == []


# ── 6. reconstruction prompt (not a summary prompt) ─────────────────────

class TestReconstructionPrompt:
    def test_prompt_is_reconstruction_not_summary(self):
        prompt = coordinate_prompt("fr")
        assert "RECONSTRUCTION" in prompt
        assert "not summarization" in prompt
        assert "Do NOT summarize" in prompt
        assert "Return ONLY the final sentence" in prompt
        assert "Never add new facts" in prompt

    def test_prompt_keeps_meaning_and_fixes_grammar(self):
        prompt = coordinate_prompt("fr")
        assert "Keep the meaning and the intention" in prompt
        assert "Fix grammar" in prompt
        assert "word order" in prompt

    def test_prompt_handles_short_or_imperfect_sequences(self):
        prompt = coordinate_prompt("fr")
        assert "Short or incomplete sequences are normal" in prompt

    def test_prompt_repetition_rule(self):
        prompt = coordinate_prompt("fr")
        assert "repeated back-to-back" in prompt
        assert "carries meaning" in prompt

    def test_prompt_language_is_selected(self):
        assert "in French" in coordinate_prompt("fr")
        assert "in English" in coordinate_prompt("en")
        assert "in Arabic" in coordinate_prompt("ar")
        assert "Tunisian Arabic" in coordinate_prompt("tunisian")

    def test_prompt_has_no_unsubstituted_placeholder(self):
        for lang in ("en", "fr", "ar", "tunisian"):
            prompt = coordinate_prompt(lang)
            assert "{language}" not in prompt
            assert "{examples}" not in prompt

    def test_prompt_contains_examples_for_fr_and_en(self):
        fr = coordinate_prompt("fr")
        assert "J'aime la Tunisie." in fr
        assert "Où vas-tu ?" in fr
        assert "Je veux apprendre le français." in fr
        assert "Bonjour, comment ça va ?" in fr

        en = coordinate_prompt("en")
        assert "Where are you going?" in en
        assert "I want to learn French." in en

    def test_unknown_language_falls_back_to_english_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.translation.llm_corrector"):
            prompt = coordinate_prompt("klingon")
        assert "in English" in prompt
        assert any("klingon" in r.getMessage() for r in caplog.records)

    def test_coordinate_uses_target_language_prompt(self, monkeypatch):
        corrector = LLMCorrector(_local_config())
        seen = {}
        client = _RecordingClient()

        async def fake_post(url, json=None, **kw):
            seen.update(system=json["messages"][0]["content"])
            return _FakeResponse("Phrase reconstruite.")

        client.post = fake_post
        _use_client(corrector, client)

        out = asyncio.run(corrector.coordinate("moi aimer tunisie", "fr"))
        assert out == "Phrase reconstruite."
        assert "in French" in seen["system"]
        assert "J'aime la Tunisie." in seen["system"]


# ── 7. OpenRouter reasoning-parameter fallback ──────────────────────────

class TestReasoningFallback:
    """OpenRouter free-tier endpoints disagree about `reasoning`: some 400
    when it is disabled. The call must adapt instead of returning raw text."""

    @staticmethod
    def _client(status_for_variant, content="Sortie LLM."):
        import httpx

        attempts = []

        class Resp:
            def __init__(self, status):
                self.status_code = status

            def raise_for_status(self):
                if self.status_code >= 400:
                    req = httpx.Request(
                        "POST", "https://openrouter.ai/api/v1/chat/completions"
                    )
                    raise httpx.HTTPStatusError(
                        str(self.status_code), request=req,
                        response=httpx.Response(self.status_code, request=req),
                    )

            def json(self):
                return {
                    "choices": [
                        {"message": {"content": content}, "finish_reason": "stop"}
                    ]
                }

        class Client:
            async def post(self, url, json=None, **kw):
                attempts.append(json)
                return Resp(status_for_variant(json))

        return Client(), attempts

    def test_400_on_reasoning_retries_without_it(self):
        client, attempts = self._client(
            lambda p: 400 if "reasoning" in p else 200
        )
        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="k", model="m")
        )
        _use_client(corrector, client)

        out = asyncio.run(corrector.coordinate("moi aimer tunisie", "fr"))
        assert out == "Sortie LLM."
        assert len(attempts) == 2
        assert "reasoning" in attempts[0]
        assert "reasoning" not in attempts[1]

    def test_persistent_400_falls_back_to_raw(self):
        client, attempts = self._client(lambda p: 400)
        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="k", model="m")
        )
        _use_client(corrector, client)

        out = asyncio.run(corrector.coordinate("moi aimer tunisie", "fr"))
        assert out == "moi aimer tunisie"
        assert len(attempts) == 3  # all variants tried before giving up

    def test_no_variant_when_reasoning_absent(self):
        client, attempts = self._client(lambda p: 200)
        corrector = LLMCorrector(_local_config())
        _use_client(corrector, client)

        asyncio.run(corrector.coordinate("hello", "en"))
        assert len(attempts) == 1
        assert "reasoning" not in attempts[0]

    def test_empty_content_retries_with_bigger_budget(self):
        attempts = []

        class Resp:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                # the model only answers when it is given a real budget
                answer = (
                    "Comment ça va ?"
                    if self.payload["max_tokens"] >= 4096
                    else ""
                )
                return {
                    "choices": [
                        {"message": {"content": answer},
                         "finish_reason": "length"}
                    ]
                }

        class Client:
            async def post(self, url, json=None, **kw):
                attempts.append(json)
                return Resp(json)

        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="k",
                      model="m", max_tokens=1024)
        )
        _use_client(corrector, Client())

        out = asyncio.run(corrector.coordinate("bonjour", "fr"))
        assert out == "Comment ça va ?"
        assert len(attempts) == 3
        assert attempts[-1]["max_tokens"] == 4096
        assert attempts[-1].get("reasoning") is None

    def test_transport_error_is_not_swallowed_as_empty(self):
        corrector = LLMCorrector(_local_config())
        with pytest.raises(httpx.ConnectError):
            asyncio.run(corrector._post_variants(
                _ThrowingClient(),
                "http://localhost:1234/v1/chat/completions",
                [{}],
            ))


class _ThrowingClient:
    async def post(self, url, json=None, **kw):
        req = httpx.Request("POST", url)
        raise httpx.ConnectError("boom", request=req)


# ── 8. stub replies from non-chat routed models ─────────────────────────

class TestStubReplyGuard:
    """`openrouter/free` can route to `nvidia/nemotron-3.5-content-safety`,
    which answers 'User Safety: safe' to every prompt."""

    @staticmethod
    def _client(contents):
        attempts = []

        class Client:
            async def post(self, url, json=None, **kw):
                idx = min(len(attempts), len(contents) - 1)
                attempts.append(json)
                return _FakeResponse(contents[idx])

        return Client(), attempts

    def test_is_usable_reply_rules(self):
        from src.translation.llm_corrector import is_usable_reply

        # stub: no shared word, no terminal punctuation
        assert not is_usable_reply("User Safety: safe", "taxi et Tunis")
        assert not is_usable_reply("User Safety: safe", "where you go")
        # empty
        assert not is_usable_reply("", "hello")
        # shares a word with the input
        assert is_usable_reply("J'aime la Tunisie.", "moi aimer tunisie")
        assert is_usable_reply("Où vas-tu ?", "toi aller où")
        # well-formed short sentence, translated (no shared word)
        assert is_usable_reply("Where are you going?", "where you go")
        assert is_usable_reply("Hi!", "hello")

    def test_stub_reply_is_retried(self):
        client, attempts = self._client(
            ["User Safety: safe", "Where are you going?"]
        )
        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="k", model="m")
        )
        _use_client(corrector, client)

        out = asyncio.run(corrector.coordinate("where you go", "en"))
        assert out == "Where are you going?"
        assert len(attempts) == 2

    def test_stub_only_falls_back_to_raw(self):
        client, attempts = self._client(["User Safety: safe"] * 5)
        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="k", model="m")
        )
        _use_client(corrector, client)

        out = asyncio.run(corrector.coordinate("taxi et Tunis", "fr"))
        assert out == "taxi et Tunis"
        assert len(attempts) == 3  # every variant rejected, then give up

    def test_correction_path_has_no_relation_guard(self):
        client, attempts = self._client(["test"])
        corrector = LLMCorrector(_local_config())
        _use_client(corrector, client)

        out = asyncio.run(corrector.correct("tset"))
        assert out == "test"
        assert len(attempts) == 1

    def test_plausible_reply_rules(self):
        from src.translation.llm_corrector import is_plausible_reply

        assert not is_plausible_reply("")
        assert not is_plausible_reply("   ")
        # reasoning trace leaked into the answer
        assert not is_plausible_reply(
            "Here's a thinking process:\n\n1. Analyze the input"
        )
        assert not is_plausible_reply("Let's think about it")
        assert not is_plausible_reply("**Step 1**: parse the signs")
        # multi-paragraph / far too long for one sentence
        assert not is_plausible_reply("Bonjour.\n\nAutre phrase.")
        assert not is_plausible_reply("x" * 700)
        assert is_plausible_reply("Bonjour, comment ça va ?")

    def test_thinking_trace_is_retried_then_accepted(self):
        client, attempts = self._client(
            ["Let's think about it", "Le frère et vendredi."]
        )
        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="k", model="m")
        )
        _use_client(corrector, client)

        out = asyncio.run(corrector.coordinate("frère et vendredi", "fr"))
        assert out == "Le frère et vendredi."
        assert len(attempts) == 2

    def test_thinking_trace_is_never_returned_by_correction(self):
        client, attempts = self._client(["Let's think about it"])
        corrector = LLMCorrector(_local_config())
        _use_client(corrector, client)

        out = asyncio.run(corrector.correct("tset"))
        assert out == "tset"  # raw fallback, not the trace
        assert len(attempts) == 2  # bigger-budget variant also rejected


# ── 9. 429 rate limits (OpenRouter free tier = 50 requests/day) ─────────

class TestRateLimitHandling:
    @staticmethod
    def _error(headers):
        req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        return httpx.HTTPStatusError(
            "429", request=req,
            response=httpx.Response(429, headers=headers, request=req),
        )

    def test_delay_rules(self):
        from src.translation.llm_corrector import rate_limit_retry_delay

        # daily quota exhausted: waiting cannot help inside a request
        assert rate_limit_retry_delay(
            self._error({"x-ratelimit-remaining": "0"}), 2.0
        ) is None
        assert rate_limit_retry_delay(
            self._error({"retry-after": "3600"}), 2.0
        ) is None
        # short server-side delay is honoured
        assert rate_limit_retry_delay(
            self._error({"retry-after": "10"}), 2.0
        ) == 10.0
        # burst / no metadata: keep the local backoff
        assert rate_limit_retry_delay(
            self._error({"x-ratelimit-remaining": "5"}), 2.0
        ) == 2.0
        assert rate_limit_retry_delay(self._error({}), 2.0) == 2.0

    def test_daily_quota_stops_after_one_attempt(self):
        calls = {"n": 0}

        class Client:
            async def post(self, url, json=None, **kw):
                calls["n"] += 1
                raise TestRateLimitHandling._error(
                    {"x-ratelimit-remaining": "0"}
                )

        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="k", model="m")
        )
        _use_client(corrector, Client())

        out = asyncio.run(corrector.coordinate("bonjour tout le monde", "fr"))
        assert out == "bonjour tout le monde"
        assert calls["n"] == 1  # no futile retries against a daily quota

    def test_burst_rate_limit_still_retries(self, monkeypatch):
        calls = {"n": 0}

        class Client:
            async def post(self, url, json=None, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise TestRateLimitHandling._error(
                        {"x-ratelimit-remaining": "5"}
                    )
                return _FakeResponse("Bonjour tout le monde.")

        corrector = LLMCorrector(
            LLMConfig(provider=LLMProvider.OPENROUTER, api_key="k", model="m")
        )
        _use_client(corrector, Client())

        async def no_sleep(_):
            return None

        monkeypatch.setattr(
            "src.translation.llm_corrector.asyncio.sleep", no_sleep
        )

        out = asyncio.run(corrector.coordinate("bonjour tout le monde", "fr"))
        assert out == "Bonjour tout le monde."
        assert calls["n"] == 2


# ── 10. user instruction sent with the reconstruction prompt ────────────

class TestCoordinateInstruction:
    def test_coordinate_user_message_asks_for_reconstruction(self):
        corrector = LLMCorrector(_local_config())
        client = _RecordingClient()
        _use_client(corrector, client)

        asyncio.run(corrector.coordinate("moi aimer tunisie", "fr"))
        user_msg = client.last_payload["messages"][1]["content"]
        assert user_msg.endswith("moi aimer tunisie")
        assert "Rebuild this detected sign sequence" in user_msg
        assert "reconstruction, not a summary" in user_msg

    def test_correct_user_message_is_unchanged(self):
        corrector = LLMCorrector(_local_config())
        client = _RecordingClient()
        _use_client(corrector, client)

        asyncio.run(corrector.correct("tset"))
        user_msg = client.last_payload["messages"][1]["content"]
        assert user_msg.startswith("Correct this TSL sentence:")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
