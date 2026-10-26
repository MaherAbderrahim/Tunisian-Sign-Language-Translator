"""API-level tests for the LLM model-selection chain.

Settings -> POST /llm/config -> pipeline -> LLM call payload -> response.

These run against the real FastAPI app (TestClient + lifespan) and the real
video pipeline; only the outbound HTTP call to the LLM is faked so the test
can assert *which* model was requested.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def restore_llm_config():
    """Snapshot/restore the process-wide LLM configuration around a test."""
    from api.main import translator as tr
    cfg = tr.llm_corrector.config
    snapshot = dict(
        provider=cfg.provider.value,
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
    )
    yield
    tr.configure_llm(**snapshot)


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
    def __init__(self, content="Reconstructed."):
        self.content = content
        self.calls = []

    async def post(self, url, json=None, **kwargs):
        self.calls.append({"url": url, "payload": json})
        return _FakeResponse(self.content)

    @property
    def last_payload(self):
        return self.calls[-1]["payload"] if self.calls else None


def _tiny_video(tmp_path):
    import cv2
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "labes.jpg"
    img = cv2.imread(str(img_path))
    assert img is not None, f"missing fixture {img_path}"
    h, w = img.shape[:2]
    vw, vh = min(w, 320), int(min(h, 320) * h / w)
    if vh % 2:
        vh += 1
    path = str(tmp_path / "tiny.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (vw, vh))
    for _ in range(45):
        out.write(cv2.resize(img, (vw, vh)))
    out.release()
    return path


def _upload(client, path, query):
    with open(path, "rb") as f:
        return client.post(
            f"/translate/video?{query}",
            files={"file": ("tiny.mp4", f, "video/mp4")},
        )


class TestLocalModelSelection:
    def test_partial_update_keeps_provider_and_key(self, client, restore_llm_config):
        """A payload with only `model` must not reset the provider or wipe the key."""
        r = client.post("/llm/config", json={
            "provider": "openrouter",
            "api_key": "test-key",
            "model": "some/model",
        })
        assert r.status_code == 200
        first = r.json()["config"]
        assert first["has_api_key"] is True

        r = client.post("/llm/config", json={"model": "another/model"})
        assert r.status_code == 200
        cfg = r.json()["config"]
        assert cfg["provider"] == "openrouter"          # not silently reset
        assert cfg["model"] == "another/model"
        assert cfg["has_api_key"] is True               # key not wiped
        assert cfg["temperature"] == first["temperature"]
        assert cfg["max_tokens"] == first["max_tokens"]  # generation params kept

    def test_local_model_is_selected_and_actually_called(
        self, client, tmp_path, monkeypatch, restore_llm_config
    ):
        from api.main import translator as tr

        # 1. Settings: select a local (OpenAI-compatible) model
        r = client.post("/llm/config", json={
            "provider": "custom",
            "api_key": "",
            "model": "local-test-model",
            "base_url": "http://localhost:1234/v1",
        })
        assert r.status_code == 200, r.text
        cfg = r.json()["config"]
        assert cfg["provider"] == "custom"
        assert cfg["model"] == "local-test-model"
        assert cfg["is_local"] is True
        assert cfg["enabled"] is True
        assert cfg["model"] != "openrouter/free"

        # 2. The server really keeps that selection on read-back
        cfg = client.get("/llm/config").json()
        assert cfg["model"] == "local-test-model"
        assert cfg["provider"] == "custom"

        # 3. Pipeline: video -> signs -> LLM coordination
        recorder = _RecordingClient(content="J'aime la Tunisie.")

        async def _get():
            return recorder

        monkeypatch.setattr(tr.llm_corrector, "_get_client", _get)

        video = _tiny_video(tmp_path)
        r = _upload(client, video, "llm=1&target_lang=fr")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["target_lang"] == "fr"
        assert data["count"] >= 1

        # 4. The outbound call went to the LOCAL endpoint with the SELECTED model
        assert recorder.calls, "the LLM was never called"
        assert recorder.calls[0]["url"].startswith("http://localhost:1234/v1/")
        assert recorder.last_payload["model"] == "local-test-model"
        assert recorder.last_payload["model"] != "openrouter/free"

        # 5. The French reconstruction prompt was used and the sentence returned
        system = recorder.last_payload["messages"][0]["content"]
        assert "in French" in system
        assert "RECONSTRUCTION" in system
        for s in data["sentences"]:
            assert s["coordinated"] == "J'aime la Tunisie."
            assert s["coordinated"] != s["sentence"]

        # 6. Selection is persisted so a restart cannot revert it
        from src.translation.llm_corrector import load_saved_config
        saved = load_saved_config()
        assert saved is not None
        assert saved.provider.value == "custom"
        assert saved.model == "local-test-model"


class TestTargetLanguage:
    def test_unsupported_target_lang_is_rejected(self, client, tmp_path):
        video = _tiny_video(tmp_path)
        r = _upload(client, video, "llm=0&target_lang=zz")
        assert r.status_code == 400
        assert "target_lang" in r.json()["detail"]

    def test_target_lang_alias_is_normalized(self, client, tmp_path):
        video = _tiny_video(tmp_path)
        r = _upload(client, video, "llm=0&target_lang=french")
        assert r.status_code == 200
        assert r.json()["target_lang"] == "fr"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
