"""
Test LLM corrector with real API calls.
Requires LLM_API_KEY in .env.
"""
import sys, os, asyncio
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.translation.llm_corrector import LLMCorrector, LLMConfig

results = {"pass": 0, "fail": 0, "skip": 0}

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
        print(f"  [ERROR] {name}: {e}")

# Load config from env
config = LLMConfig.from_env()
corrector = LLMCorrector(config)

if not corrector.enabled:
    print("  LLM corrector disabled (no API key). Skipping real LLM tests.")
    print("  Set LLM_API_KEY in .env to enable.")
    sys.exit(0)

print(f"  Provider: {config.provider}")
print(f"  Model: {config.model}")

# ── Test Cases ──

async def test_correct_correct_phrase():
    """Correct phrase (should remain unchanged or minimally changed)."""
    result = await corrector.correct("bonjour")
    assert isinstance(result, str), f"Expected string, got {type(result)}"
    assert len(result) > 0, "Empty result"

test("Cas 1: Correct phrase", lambda: asyncio.run(test_correct_correct_phrase()))

async def test_correct_typo():
    """Phrase with typos."""
    result = await corrector.correct("bonjur ca va")
    assert isinstance(result, str)
    # The LLM should either correct "bonjur" to "bonjour" or leave it
    assert len(result) > 0

test("Cas 2: Phrase with typos", lambda: asyncio.run(test_correct_typo()))

async def test_correct_letters():
    """Assembled letters (raw finger spelling)."""
    result = await corrector.correct("ب و ن ج و ر")
    assert isinstance(result, str)
    assert len(result) > 0

test("Cas 3: Assembled letters", lambda: asyncio.run(test_correct_letters()))

async def test_correct_mixed():
    """Mixed sign + spelling."""
    result = await corrector.correct("bonjour م ر ح ب ا")
    assert isinstance(result, str)
    assert len(result) > 0

test("Cas 4: Mixed sign + spelling", lambda: asyncio.run(test_correct_mixed()))

async def test_correct_empty():
    """Empty input."""
    result = await corrector.correct("")
    assert result == "", f"Expected empty, got '{result}'"

test("Cas 5: Empty input", lambda: asyncio.run(test_correct_empty()))

async def test_correct_whitespace():
    """Whitespace only."""
    result = await corrector.correct("   ")
    assert isinstance(result, str)

test("Cas 6: Whitespace only", lambda: asyncio.run(test_correct_whitespace()))

# ── Error Handling ──

async def test_wrong_api_key():
    """Wrong API key — should fall back gracefully."""
    bad_config = LLMConfig(provider="openrouter", api_key="sk-invalid-key-12345", model="meta-llama/llama-3.1-8b-instruct:free")
    bad_corrector = LLMCorrector(bad_config)
    result = await bad_corrector.correct("bonjour")
    # Should return raw text, not crash
    assert result == "bonjour", f"Expected 'bonjour', got '{result}'"

test("Cas 7: Wrong API key (fallback)", lambda: asyncio.run(test_wrong_api_key()))

# ── Summary ──
print(f"\n{'='*60}")
print(f"LLM TEST RESULTS")
print(f"{'='*60}")
print(f"  Passed: {results['pass']}")
print(f"  Failed: {results['fail']}")
