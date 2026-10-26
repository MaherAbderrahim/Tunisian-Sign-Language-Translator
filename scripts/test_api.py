"""
Test all REST API endpoints using FastAPI TestClient.
No server needed — runs in-process.
"""
import sys, os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# This script POSTs a test LLM config: keep it away from the developer's real
# data/llm_config.json (which may contain a live API key).
os.environ.setdefault(
    "TSL_LLM_CONFIG_PATH",
    str(Path(tempfile.mkdtemp(prefix="tsl_api_test_")) / "llm_config.json"),
)

from fastapi.testclient import TestClient
from api.main import app

results = {"pass": 0, "fail": 0}

def test(name, fn):
    try:
        fn()
        results["pass"] += 1
        print(f"  [PASS] {name}")
    except AssertionError as e:
        results["fail"] += 1
        print(f"  [FAIL] {name}: {e}")
    except Exception as e:
        results["fail"] += 1
        print(f"  [ERROR] {name}: {type(e).__name__}: {e}")

# Use context manager to trigger lifespan (loads translator)
with TestClient(app, raise_server_exceptions=False) as client:

    # ── Health & Root ──

    def test_root():
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "version" in data
    test("GET /", test_root)

    def test_health():
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ready"] is True
    test("GET /health", test_health)

    # ── Dictionary ──

    def test_labels():
        r = client.get("/dictionary/labels")
        assert r.status_code == 200
        assert r.json()["count"] == 57
    test("GET /dictionary/labels", test_labels)

    def test_words():
        r = client.get("/dictionary/words")
        assert r.status_code == 200
    test("GET /dictionary/words", test_words)

    def test_categories():
        r = client.get("/dictionary/categories")
        assert r.status_code == 200
    test("GET /dictionary/categories", test_categories)

    def test_label_not_found():
        r = client.get("/dictionary/nonexistent_xyz")
        assert r.status_code == 404
    test("GET /dictionary/nonexistent (404)", test_label_not_found)

    # ── Alphabet ──

    def test_alphabet_labels():
        r = client.get("/alphabet/labels")
        assert r.status_code == 200
        assert r.json()["count"] == 31
    test("GET /alphabet/labels", test_alphabet_labels)

    def test_alphabet_label_found():
        r = client.get("/alphabet/alef")
        assert r.status_code == 200
        data = r.json()
        assert data.get("arabic") == "أ"
    test("GET /alphabet/alef", test_alphabet_label_found)

    def test_alphabet_label_not_found():
        r = client.get("/alphabet/nonexistent_xyz")
        assert r.status_code == 404
    test("GET /alphabet/nonexistent (404)", test_alphabet_label_not_found)

    # ── Text Translation ──

    def test_translate_text():
        r = client.post("/translate/text", json={"text": "bonjour", "target_lang": "french"})
        assert r.status_code == 200
        data = r.json()
        assert data["input"] == "bonjour"
    test("POST /translate/text", test_translate_text)

    def test_translate_text_empty():
        r = client.post("/translate/text", json={"text": "", "target_lang": "french"})
        assert r.status_code == 200
    test("POST /translate/text (empty)", test_translate_text_empty)

    # ── Multi Translation ──

    def test_translate_multi():
        r = client.post("/translate/multi", json={"text": "bonjour", "source_lang": "french", "target_langs": ["english"]})
        assert r.status_code == 200
        data = r.json()
        assert "translations" in data
    test("POST /translate/multi", test_translate_multi)

    # ── Frame Translation ──

    def test_translate_frame_invalid():
        r = client.post("/translate/frame", files={"file": ("test.txt", b"not an image", "text/plain")})
        assert r.status_code == 400
    test("POST /translate/frame (invalid)", test_translate_frame_invalid)

    # ── Mode Controller ──

    def test_mode_get():
        r = client.get("/mode")
        assert r.status_code == 200
        data = r.json()
        assert "mode" in data
    test("GET /mode", test_mode_get)

    def test_mode_toggle():
        r = client.post("/mode/toggle")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] in ("sign", "spell")
        r2 = client.post("/mode/toggle")
        assert r2.status_code == 200
    test("POST /mode/toggle", test_mode_toggle)

    # ── Spell Flush ──

    def test_spell_flush():
        r = client.post("/spell/flush")
        assert r.status_code == 200
        assert "sentence" in r.json()
    test("POST /spell/flush", test_spell_flush)

    # ── LLM Config ──

    def test_llm_config_get():
        r = client.get("/llm/config")
        assert r.status_code == 200
        data = r.json()
        assert "provider" in data
    test("GET /llm/config", test_llm_config_get)

    def test_llm_config_set():
        r = client.post("/llm/config", json={
            "provider": "openrouter",
            "api_key": "test-key-not-real",
            "model": "meta-llama/llama-3.1-8b-instruct:free",
        })
        assert r.status_code == 200
    test("POST /llm/config", test_llm_config_set)

    # ── LLM Correct ──

    def test_llm_correct():
        r = client.post("/llm/correct", json={"text": "bonjur"})
        assert r.status_code == 200
        data = r.json()
        assert "raw" in data
        assert "corrected" in data
    test("POST /llm/correct", test_llm_correct)

    def test_llm_correct_empty():
        r = client.post("/llm/correct", json={"text": ""})
        assert r.status_code == 200
    test("POST /llm/correct (empty)", test_llm_correct_empty)

    # ── History ──

    def test_history_get():
        r = client.get("/history")
        assert r.status_code == 200
        assert "history" in r.json()
    test("GET /history", test_history_get)

    def test_history_clear():
        r = client.delete("/history")
        assert r.status_code == 200
    test("DELETE /history", test_history_clear)

    # ── Languages ──

    def test_languages():
        r = client.get("/translate/languages")
        assert r.status_code == 200
        assert "languages" in r.json()
    test("GET /translate/languages", test_languages)

# ── Summary ──
print(f"\n{'='*60}")
print(f"API TEST RESULTS")
print(f"{'='*60}")
print(f"  Passed: {results['pass']}")
print(f"  Failed: {results['fail']}")
