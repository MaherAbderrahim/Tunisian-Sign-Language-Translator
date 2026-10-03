"""Tests for REST API endpoints."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestRootEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "version" in r.json()

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["ready"] is True


class TestDictionary:
    def test_labels(self, client):
        r = client.get("/dictionary/labels")
        assert r.status_code == 200
        assert r.json()["count"] == 57

    def test_words(self, client):
        r = client.get("/dictionary/words")
        assert r.status_code == 200

    def test_categories(self, client):
        r = client.get("/dictionary/categories")
        assert r.status_code == 200

    def test_not_found(self, client):
        r = client.get("/dictionary/nonexistent_xyz")
        assert r.status_code == 404


class TestAlphabet:
    def test_labels(self, client):
        r = client.get("/alphabet/labels")
        assert r.status_code == 200
        assert r.json()["count"] == 31

    def test_label_found(self, client):
        r = client.get("/alphabet/alef")
        assert r.status_code == 200
        assert r.json().get("arabic") == "أ"

    def test_not_found(self, client):
        r = client.get("/alphabet/nonexistent_xyz")
        assert r.status_code == 404


class TestTranslation:
    def test_translate_text(self, client):
        r = client.post("/translate/text", json={"text": "bonjour"})
        assert r.status_code == 200
        assert r.json()["input"] == "bonjour"

    def test_translate_text_empty(self, client):
        r = client.post("/translate/text", json={"text": ""})
        assert r.status_code == 200

    def test_translate_multi(self, client):
        r = client.post("/translate/multi", json={"text": "bonjour", "target_langs": ["english"]})
        assert r.status_code == 200

    def test_translate_frame_invalid(self, client):
        r = client.post("/translate/frame", files={"file": ("t.txt", b"bad", "text/plain")})
        assert r.status_code == 400


class TestModeController:
    def test_get_mode(self, client):
        r = client.get("/mode")
        assert r.status_code == 200
        assert "mode" in r.json()

    def test_toggle(self, client):
        r = client.post("/mode/toggle")
        assert r.status_code == 200
        assert r.json()["mode"] in ("sign", "spell")
        # Toggle back
        client.post("/mode/toggle")


class TestSpell:
    def test_flush(self, client):
        r = client.post("/spell/flush")
        assert r.status_code == 200
        assert "sentence" in r.json()


class TestLLM:
    def test_config_get(self, client):
        r = client.get("/llm/config")
        assert r.status_code == 200
        assert "provider" in r.json()

    def test_config_set(self, client):
        r = client.post("/llm/config", json={
            "provider": "openrouter",
            "api_key": "test-key",
            "model": "test-model",
        })
        assert r.status_code == 200

    def test_correct(self, client):
        if os.environ.get("RUN_NETWORK_TESTS") != "1":
            pytest.skip("network tests disabled")
        r = client.post("/llm/correct", json={"text": "hello"})
        assert r.status_code == 200
        assert "corrected" in r.json()


class TestHistory:
    def test_get(self, client):
        r = client.get("/history")
        assert r.status_code == 200

    def test_clear(self, client):
        r = client.delete("/history")
        assert r.status_code == 200


class TestLanguages:
    def test_list(self, client):
        r = client.get("/translate/languages")
        assert r.status_code == 200
        assert "languages" in r.json()
