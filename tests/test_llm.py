"""Tests for LLM corrector."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import asyncio
import pytest
from src.translation.llm_corrector import LLMCorrector, LLMConfig


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
        assert config.model == "meta-llama/llama-3.1-8b-instruct:free"
