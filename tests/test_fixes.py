"""Regression tests for the audit fixes.

Covers: per-client sessions (H1), mode-control signs blocked by the
UNSUPPORTED filter (H2), SSRF guard on the LLM config (C1), streaming
gap dedup / single sentence boundary (L6), API session isolation and
inference settings endpoints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(scope="module")
def translator():
    from src.translation.translator import TSLTranslator
    return TSLTranslator()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestSessionIsolation:
    """H1: concurrent clients must not share history / mode / smoothing."""

    def test_mode_isolated_per_session(self, translator):
        translator.clear_history("sessA")
        translator.clear_history("sessB")
        assert translator.get_mode_state("sessA")["mode"] == "sign"
        translator.toggle_mode("sessA")
        assert translator.get_mode_state("sessA")["mode"] == "spell"
        assert translator.get_mode_state("sessB")["mode"] == "sign"
        translator.toggle_mode("sessA")  # restore

    def test_history_isolated_per_session(self, translator):
        translator.clear_history("histA")
        translator.clear_history("histB")
        translator._append_history({"tsl": "x"}, translator.get_session("histA"))
        assert len(translator.get_history("histA")) == 1
        assert translator.get_history("histB") == []

    def test_get_history_returns_copy(self, translator):
        translator.clear_history("copyS")
        translator._append_history({"tsl": "x"}, translator.get_session("copyS"))
        snapshot = translator.get_history("copyS")
        snapshot.append({"tsl": "injected"})
        assert translator.get_history("copyS") == [{"tsl": "x"}]

    def test_delete_session_resets_state(self, translator):
        sid = "delSession"
        translator._append_history({"tsl": "y"}, translator.get_session(sid))
        translator.delete_session(sid)
        assert translator.get_history(sid) == []

    def test_inference_settings_validated(self, translator):
        original = translator.get_inference_settings()
        try:
            with pytest.raises(ValueError):
                translator.set_inference_settings(confidence_threshold=1.5)
            with pytest.raises(ValueError):
                translator.set_inference_settings(smoothing_window=999)
            out = translator.set_inference_settings(
                confidence_threshold=0.6, smoothing_window=4
            )
            assert out["confidence_threshold"] == 0.6
            assert out["smoothing_window"] == 4
        finally:
            translator.set_inference_settings(**original)

    def test_smoothing_resize_keeps_window_bounded(self, translator):
        sid = "resizeSession"
        state = translator.get_session(sid)
        state.recent_predictions.extend([1, 2, 3])
        state.set_smoothing_window(2)
        assert state.recent_predictions.maxlen == 2
        assert len(state.recent_predictions) == 2


class TestModeControlSigns:
    """H2: control signs blocked by UNSUPPORTED_SIGN_CLASSES must still
    reach the mode controller (via the raw mode_label)."""

    def test_ta3raf_is_unsupported_but_consumed_by_mode_controller(self):
        from src.config.settings import UNSUPPORTED_SIGN_CLASSES
        from src.translation.mode_controller import ModeController

        assert "ta3raf" in UNSUPPORTED_SIGN_CLASSES
        mc = ModeController()
        ev = mc.update("ta3raf")
        assert ev["consumed"] is True
        assert ev["mode"] == "spell"
        ev = mc.update("labes")
        assert ev["consumed"] is True
        assert ev["mode"] == "sign"

    def test_ordinary_sign_does_not_switch_mode(self):
        from src.translation.mode_controller import ModeController
        mc = ModeController()
        ev = mc.update("bou")
        assert ev["consumed"] is False
        assert mc.mode == "sign"


class TestStreamingGapFixes:
    """L6: no duplicate emission on gap finalize; boundary fires once."""

    def test_no_duplicate_emit_on_gap_finalize(self):
        from src.translation.streaming import StreamingRecognizer
        rec = StreamingRecognizer(stable_frames=3, gap_frames=5)
        last = None
        for _ in range(3):
            ev = rec.feed({"tsl": "labes", "rejected": False})
            if ev["sign"]:
                last = ev["sign"]
        assert last == "labes"
        ev = rec.feed({"error": "no hands"})
        assert ev["sign"] is None, "already-emitted sign re-emitted on gap"

    def test_sentence_boundary_fires_exactly_once(self):
        from src.translation.streaming import StreamingRecognizer
        rec = StreamingRecognizer(stable_frames=2, gap_frames=3)
        for _ in range(3):
            rec.feed({"tsl": "labes", "rejected": False})
        boundaries = 0
        for _ in range(10):
            ev = rec.feed({"error": "no hands"})
            if ev["sentence_boundary"]:
                boundaries += 1
        assert boundaries == 1

    def test_run_can_reemit_after_real_sign_change(self):
        from src.translation.streaming import StreamingRecognizer
        rec = StreamingRecognizer(stable_frames=2, gap_frames=3)
        emitted = []
        for label in ["a", "a", "b", "b"]:
            ev = rec.feed({"tsl": label, "rejected": False})
            if ev["sign"]:
                emitted.append(ev["sign"])
        assert emitted == ["a", "b"]


class TestLLMConfigSecurity:
    """C1: SSRF guard on LLMConfig.validate()."""

    @staticmethod
    def _cfg(**kw):
        from src.translation.llm_corrector import LLMConfig
        defaults = dict(api_key="k", model="m")
        defaults.update(kw)
        cfg = LLMConfig(**defaults)
        cfg.apply_defaults()
        return cfg

    def test_known_provider_default_is_valid(self):
        from src.translation.llm_corrector import LLMProvider
        self._cfg(provider=LLMProvider.OPENROUTER, base_url="").validate()

    def test_pinned_provider_rejects_arbitrary_base_url(self, monkeypatch):
        from src.translation.llm_corrector import LLMProvider
        monkeypatch.delenv("LLM_ALLOW_CUSTOM_BASE_URL", raising=False)
        cfg = self._cfg(
            provider=LLMProvider.OPENROUTER,
            base_url="https://evil.example/v1",
        )
        with pytest.raises(ValueError):
            cfg.validate()

    def test_custom_requires_optin(self, monkeypatch):
        from src.translation.llm_corrector import LLMProvider
        monkeypatch.delenv("LLM_ALLOW_CUSTOM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_ALLOWED_HOSTS", raising=False)
        cfg = self._cfg(
            provider=LLMProvider.CUSTOM,
            base_url="https://api.example.com/v1",
        )
        with pytest.raises(ValueError):
            cfg.validate()

    def test_custom_allowed_with_optin(self, monkeypatch):
        from src.translation.llm_corrector import LLMProvider
        monkeypatch.setenv("LLM_ALLOW_CUSTOM_BASE_URL", "true")
        cfg = self._cfg(
            provider=LLMProvider.CUSTOM,
            base_url="https://api.example.com/v1",
        )
        cfg.validate()

    def test_http_rejected_for_non_loopback_without_optin(self, monkeypatch):
        from src.translation.llm_corrector import LLMProvider
        monkeypatch.delenv("LLM_ALLOW_CUSTOM_BASE_URL", raising=False)
        cfg = self._cfg(
            provider=LLMProvider.OLLAMA,
            base_url="http://169.254.169.254/v1",
        )
        with pytest.raises(ValueError):
            cfg.validate()

    def test_http_loopback_allowed(self, monkeypatch):
        from src.translation.llm_corrector import LLMProvider
        monkeypatch.delenv("LLM_ALLOW_CUSTOM_BASE_URL", raising=False)
        cfg = self._cfg(
            provider=LLMProvider.OLLAMA,
            base_url="http://localhost:11434/v1",
        )
        cfg.validate()

    def test_credentials_in_url_rejected(self, monkeypatch):
        from src.translation.llm_corrector import LLMProvider
        monkeypatch.setenv("LLM_ALLOW_CUSTOM_BASE_URL", "true")
        cfg = self._cfg(
            provider=LLMProvider.CUSTOM,
            base_url="https://user:pass@api.example.com/v1",
        )
        with pytest.raises(ValueError):
            cfg.validate()

    def test_non_http_scheme_rejected(self, monkeypatch):
        from src.translation.llm_corrector import LLMProvider
        monkeypatch.setenv("LLM_ALLOW_CUSTOM_BASE_URL", "true")
        cfg = self._cfg(
            provider=LLMProvider.CUSTOM,
            base_url="ftp://api.example.com/v1",
        )
        with pytest.raises(ValueError):
            cfg.validate()

    def test_bounds_validated(self, monkeypatch):
        monkeypatch.setenv("LLM_ALLOW_CUSTOM_BASE_URL", "true")
        with pytest.raises(ValueError):
            self._cfg(temperature=9.0).validate()
        with pytest.raises(ValueError):
            self._cfg(timeout=0).validate()
        with pytest.raises(ValueError):
            self._cfg(max_tokens=0).validate()

    def test_host_allowlist(self, monkeypatch):
        from src.translation.llm_corrector import LLMProvider
        monkeypatch.delenv("LLM_ALLOW_CUSTOM_BASE_URL", raising=False)
        monkeypatch.setenv("LLM_ALLOWED_HOSTS", "api.example.com")
        ok = self._cfg(
            provider=LLMProvider.CUSTOM,
            base_url="https://api.example.com/v1",
        )
        ok.validate()
        bad = self._cfg(
            provider=LLMProvider.CUSTOM,
            base_url="https://other.example.com/v1",
        )
        with pytest.raises(ValueError):
            bad.validate()


class TestAPISessionsAndSettings:
    """Session header isolation + /settings/inference + LLM config guard."""

    def test_root_version_matches_app(self, client):
        from api.main import app
        assert client.get("/").json()["version"] == app.version

    def test_settings_get(self, client):
        r = client.get("/settings/inference")
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["confidence_threshold"] <= 1
        assert body["smoothing_window"] >= 1

    def test_settings_rejects_invalid(self, client):
        assert client.post(
            "/settings/inference", json={"confidence_threshold": 1.5}
        ).status_code == 400
        assert client.post(
            "/settings/inference", json={"smoothing_window": 999}
        ).status_code == 400

    def test_mode_isolated_by_session_header(self, client):
        a = {"X-Session-Id": "pytestSessA"}
        b = {"X-Session-Id": "pytestSessB"}
        client.post("/mode/toggle", headers=a)
        ma = client.get("/mode", headers=a).json()["mode"]
        mb = client.get("/mode", headers=b).json()["mode"]
        assert ma == "spell"
        assert mb == "sign"
        client.post("/mode/toggle", headers=a)  # restore

    def test_history_isolated_by_session_header(self, client):
        a = {"X-Session-Id": "pytestHistA"}
        b = {"X-Session-Id": "pytestHistB"}
        client.delete("/history", headers=a)
        client.delete("/history", headers=b)
        ha = client.get("/history", headers=a).json()
        hb = client.get("/history", headers=b).json()
        assert ha["count"] == hb["count"] == 0

    def test_llm_config_rejects_ssrf(self, client):
        r = client.post("/llm/config", json={
            "provider": "custom",
            "base_url": "https://evil.example/v1",
            "model": "m",
            "api_key": "k",
        })
        assert r.status_code == 400
        assert "LLM_ALLOW_CUSTOM_BASE_URL" in r.json()["detail"] or \
               "opt" in r.json()["detail"].lower()

    def test_llm_config_rejects_unknown_provider(self, client):
        r = client.post("/llm/config", json={
            "provider": "not-a-provider",
            "api_key": "k",
            "model": "m",
        })
        assert r.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
