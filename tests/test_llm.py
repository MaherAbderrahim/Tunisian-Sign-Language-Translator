"""Tests for LLM corrector."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import asyncio
import pytest
from src.translation.llm_corrector import LLMCorrector, LLMConfig, SYSTEM_PROMPT


class TestLLMCorrector:
    def test_disabled_without_key(self):
        config = LLMConfig(api_key="")
        corrector = LLMCorrector(config)
        assert not corrector.enabled

    def test_fallback_returns_raw(self):
        config = LLMConfig(api_key="")
        corrector = LLMCorrector(config)
        result = asyncio.run(corrector.correct("hello"))
        assert result == "hello"

    def test_empty_input(self):
        config = LLMConfig(api_key="")
        corrector = LLMCorrector(config)
        result = asyncio.run(corrector.correct(""))
        assert result == ""

    def test_wrong_api_key_fallback(self):
        if os.environ.get("RUN_NETWORK_TESTS") != "1":
            pytest.skip("network tests disabled")
        config = LLMConfig(provider="openrouter", api_key="sk-invalid-key", model="test")
        corrector = LLMCorrector(config)
        result = asyncio.run(corrector.correct("hello"))
        assert result == "hello"  # Falls back to raw text

    def test_config_to_dict(self):
        config = LLMConfig(provider="openrouter", api_key="test-key", model="test-model")
        d = config.to_dict()
        assert d["provider"] == "openrouter"
        assert d["model"] == "test-model"
        assert d["has_api_key"] is True

    def test_provider_defaults(self):
        config = LLMConfig(provider="openrouter")
        config.apply_defaults()
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.model == "openrouter/free"

    def test_coordinate_disabled_returns_raw(self):
        corrector = LLMCorrector(LLMConfig(api_key=""))
        out = asyncio.run(corrector.coordinate("cv and 3ayla and 5ou", "fr"))
        assert out == "cv and 3ayla and 5ou"

    def test_coordinate_empty_returns_empty(self):
        corrector = LLMCorrector(LLMConfig(api_key=""))
        assert asyncio.run(corrector.coordinate("", "en")) == ""

    def test_coordinate_uses_coordination_prompt_in_target_lang(self, monkeypatch):
        corrector = LLMCorrector(LLMConfig(api_key="k", model="m"))
        seen = {}

        async def fake_call(client, raw_text, system=SYSTEM_PROMPT, **kwargs):
            seen["system"] = system
            seen["raw"] = raw_text
            return "Une phrase naturelle."

        monkeypatch.setattr(corrector, "_call_openai_compatible", fake_call)
        out = asyncio.run(corrector.coordinate("cv and 3ayla and 5ou", "fr"))
        assert out == "Une phrase naturelle."
        assert seen["raw"] == "cv and 3ayla and 5ou"
        assert "French" in seen["system"]
        # coordination path uses the reconstruction prompt, never the corrector one
        assert "RECONSTRUCTION" in seen["system"]
        assert "sentence corrector" not in seen["system"]

    def test_coordinate_timeout_falls_back_to_raw(self, monkeypatch):
        import httpx
        corrector = LLMCorrector(LLMConfig(api_key="k", model="m"))

        async def timeout(client, raw_text, system=SYSTEM_PROMPT, **kwargs):
            raise httpx.TimeoutException("boom")

        monkeypatch.setattr(corrector, "_call_openai_compatible", timeout)
        assert asyncio.run(corrector.coordinate("raw chain", "en")) == "raw chain"

    def test_coordinate_retries_once_on_429(self, monkeypatch):
        import httpx
        corrector = LLMCorrector(LLMConfig(api_key="k", model="m"))
        calls = {"n": 0}

        async def flaky(client, raw_text, system=SYSTEM_PROMPT, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
                raise httpx.HTTPStatusError(
                    "429", request=req, response=httpx.Response(429, request=req)
                )
            return "Coordonné."

        async def no_sleep(_):
            return None

        monkeypatch.setattr(corrector, "_call_openai_compatible", flaky)
        monkeypatch.setattr("src.translation.llm_corrector.asyncio.sleep", no_sleep)
        out = asyncio.run(corrector.coordinate("a and b", "en"))
        assert out == "Coordonné."
        assert calls["n"] == 2

    def test_correct_still_uses_correction_prompt(self, monkeypatch):
        corrector = LLMCorrector(LLMConfig(api_key="k", model="m"))
        seen = {}

        async def fake_call(client, raw_text, system=SYSTEM_PROMPT, **kwargs):
            seen["system"] = system
            return "corrigé"

        monkeypatch.setattr(corrector, "_call_openai_compatible", fake_call)
        out = asyncio.run(corrector.correct("tset"))
        assert out == "corrigé"
        assert "corrector" in seen["system"]
