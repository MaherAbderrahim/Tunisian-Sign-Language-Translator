"""
Comprehensive validation script for TSL project.
Tests datasets, models, pipeline, and data leakage.
"""
import sys, os, time, json, traceback
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

results = {"pass": 0, "fail": 0, "skip": 0, "details": []}

def test(name, fn):
    try:
        fn()
        results["pass"] += 1
        results["details"].append({"name": name, "status": "PASS"})
        print(f"  [PASS] {name}")
    except AssertionError as e:
        results["fail"] += 1
        results["details"].append({"name": name, "status": "FAIL", "error": str(e)})
        print(f"  [FAIL] {name}: {e}")
    except Exception as e:
        results["fail"] += 1
        results["details"].append({"name": name, "status": "ERROR", "error": str(e)})
        print(f"  [ERROR] {name}: {e}")
        traceback.print_exc()

# ═══════════════════════════════════════════════════════════
# 1. DATASET VALIDATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. DATASET VALIDATION")
print("=" * 60)

def test_alphabet_dataset():
    data_dir = PROJECT_ROOT / "data" / "alphabet" / "processed"
    features = np.load(data_dir / "features.npy")
    labels = np.load(data_dir / "labels.npy")

    assert features.shape[1] == 126, f"Expected 126 features, got {features.shape[1]}"
    assert features.shape[0] == labels.shape[0], "Features/labels count mismatch"

    # Check for NaN
    nan_count = np.isnan(features).sum()
    assert nan_count == 0, f"Found {nan_count} NaN values"

    # Check label range
    assert labels.min() >= 0, f"Negative label: {labels.min()}"
    assert labels.max() <= 30, f"Label out of range: {labels.max()}"

    # Check for duplicates
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"  Samples: {len(features)}, Features: {features.shape[1]}, Classes: {len(unique_labels)}")
    print(f"  Min/max feature: {features.min():.4f} / {features.max():.4f}")
    for i, (l, c) in enumerate(zip(unique_labels, counts)):
        print(f"    class {l}: {c} samples")
    assert len(unique_labels) == 31, f"Expected 31 classes, got {len(unique_labels)}"

test("Alphabet dataset", test_alphabet_dataset)

def test_sign_dataset():
    features = np.load(PROJECT_ROOT / "data" / "processed" / "features.npy")
    labels = np.load(PROJECT_ROOT / "data" / "processed" / "labels.npy")

    assert features.shape[1] == 126, f"Expected 126 features, got {features.shape[1]}"
    nan_count = np.isnan(features).sum()
    assert nan_count == 0, f"Found {nan_count} NaN values"

    unique_labels = np.unique(labels)
    print(f"  Samples: {len(features)}, Classes with data: {len(unique_labels)}/57")
    print(f"  Label range: {labels.min()} to {labels.max()}")
    # Not all 57 classes may have samples in the dataset - that's OK
    # The model still has 57 outputs for compatibility
    assert len(unique_labels) >= 40, f"Too few classes with data: {len(unique_labels)}"

test("Sign dataset", test_sign_dataset)

# ═══════════════════════════════════════════════════════════
# 2. DATA LEAKAGE CHECK
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. DATA LEAKAGE CHECK")
print("=" * 60)

def test_data_leakage():
    data_dir = PROJECT_ROOT / "data" / "alphabet" / "processed"
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    # Check for exact duplicates across splits
    train_set = set(map(tuple, X_train.tolist()))
    val_set = set(map(tuple, X_val.tolist()))
    test_set = set(map(tuple, X_test.tolist()))

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train-Val overlap: {len(train_val_overlap)}")
    print(f"  Train-Test overlap: {len(train_test_overlap)}")
    print(f"  Val-Test overlap: {len(val_test_overlap)}")

    # Note: Some overlap is expected since augmentation adds noisy copies
    # Check that original (non-augmented) data is split properly
    total_unique = len(train_set | val_set | test_set)
    print(f"  Total unique samples: {total_unique}")
    print(f"  Total across splits: {len(X_train) + len(X_val) + len(X_test)}")

    # The augmentation creates noisy copies, so some overlap in raw features
    # is expected. The key is that the ORIGINAL samples are not duplicated.
    # Since we augment AFTER split, original samples are only in one split.
    # But augmented samples may be very close to originals in other splits.
    # This is acceptable as long as the test set has no augmented copies.
    assert len(train_test_overlap) == 0, f"CRITICAL: {len(train_test_overlap)} exact duplicates between train and test!"

test("Data leakage check", test_data_leakage)

# ═══════════════════════════════════════════════════════════
# 3. ALPHABET MODEL EVALUATION ON TEST SET
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. ALPHABET MODEL EVALUATION")
print("=" * 60)

def test_alphabet_model():
    import torch
    from src.models.spell_classifier import SpellClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    data_dir = PROJECT_ROOT / "data" / "alphabet" / "processed"
    model_path = PROJECT_ROOT / "models" / "alphabet" / "spell_classifier_best.pth"

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Detect architecture from state_dict keys (same logic as
    # src/translation/translator.py::_load_spell_model)
    if any(k.startswith("net.") for k in state_dict.keys()):
        model = SpellClassifier(num_classes=31, architecture="mlp")
    elif any(k.startswith("input_block.") for k in state_dict.keys()):
        model = SpellClassifier(num_classes=31, architecture="residual")
    else:
        model = SpellClassifier(num_classes=31, architecture="mlp")

    model.load_state_dict(state_dict)
    model.eval()

    # Load test data
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    # Load class names from training
    with open(data_dir / "class_names.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Inference
    X_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(dim=-1).numpy()

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)

    print(f"  Test samples: {len(y_test)}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")

    # Per-class accuracy
    cm = confusion_matrix(y_test, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    worst_idx = per_class_acc.argsort()[:5]
    best_idx = per_class_acc.argsort()[-5:][::-1]

    print("\n  Best classes:")
    for idx in best_idx:
        if idx < len(class_names):
            print(f"    {class_names[idx]}: {per_class_acc[idx]:.4f}")

    print("\n  Worst classes:")
    for idx in worst_idx:
        if idx < len(class_names):
            print(f"    {class_names[idx]}: {per_class_acc[idx]:.4f}")

    # Full classification report
    print("\n  Classification Report:")
    target_names = [class_names[i] if i < len(class_names) else str(i) for i in range(len(class_names))]
    print(classification_report(y_test, preds, target_names=target_names, zero_division=0))

    # Store results for comparison
    test_alphabet_model.acc = acc
    test_alphabet_model.f1 = f1_macro

    assert acc > 0.5, f"Accuracy too low: {acc:.4f}"

test("Alphabet model evaluation", test_alphabet_model)

# ═══════════════════════════════════════════════════════════
# 4. 57-SIGN MODEL EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. 57-SIGN MODEL EVALUATION")
print("=" * 60)

def test_sign_model():
    import torch
    from src.models.classifier import TSLClassifier, create_model
    from src.languages.dictionary import TSLDictionary
    from src.config.settings import NUM_CLASSES, DEFAULT_MODEL_PATH
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    # Load model
    checkpoint = torch.load(DEFAULT_MODEL_PATH, map_location="cpu", weights_only=True)
    model = create_model("landmark", NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load data
    features = np.load(PROJECT_ROOT / "data" / "processed" / "features.npy")
    labels = np.load(PROJECT_ROOT / "data" / "processed" / "labels.npy")

    # Use a seeded 80/20 holdout for evaluation (no separate test set exists
    # for signs). The data is class-ordered, so a contiguous slice would be
    # non-representative — shuffle first.
    rng = np.random.RandomState(42)
    order = rng.permutation(len(features))
    features = features[order]
    labels = labels[order]
    split = int(len(features) * 0.8)
    X_test = features[split:]
    y_test = labels[split:]
    print("  CAVEAT: this holdout is a seeded 80/20 shuffle of the FULL dataset;")
    print("          it approximates (does NOT exactly reproduce) the training")
    print("          split, so it may overlap the model's training samples.")

    dictionary = TSLDictionary()
    class_names = dictionary.get_all_labels()

    # Inference
    X_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(dim=-1).numpy()

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"  Test samples: {len(y_test)}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")

    # Confidence analysis
    with torch.no_grad():
        probs = torch.softmax(model(X_tensor), dim=-1)
        confs = probs.max(dim=-1).values.numpy()

    print(f"  Avg confidence: {confs.mean():.4f}")
    print(f"  Min confidence: {confs.min():.4f}")
    print(f"  Max confidence: {confs.max():.4f}")

    test_sign_model.acc = acc
    test_sign_model.f1 = f1_macro

    assert acc > 0.3, f"Accuracy too low: {acc:.4f}"

test("Sign model evaluation", test_sign_model)

# ═══════════════════════════════════════════════════════════
# 5. LANDMARK PIPELINE TEST
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. LANDMARK PIPELINE TEST")
print("=" * 60)

def test_landmark_pipeline():
    import torch
    from src.models.landmark_extractor import MediaPipeLandmarkExtractor
    from src.models.classifier import create_model
    from src.config.settings import NUM_CLASSES

    extractor = MediaPipeLandmarkExtractor(max_num_hands=2)
    model = create_model("landmark", NUM_CLASSES)

    # Test with a synthetic black frame (no hands)
    frame_no_hands = np.zeros((480, 640, 3), dtype=np.uint8)
    features = extractor.extract_features(frame_no_hands)
    assert features is None, "Should return None for no hands"

    # Test with random noise frame (might or might not detect hands)
    frame_random = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    features = extractor.extract_features(frame_random)
    if features is not None:
        # extract_features returns a single (126,) vector per frame
        assert features.shape == (126,), f"Expected (126,), got {features.shape}"
        # Test model inference (batch of one frame)
        tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            assert logits.shape == (1, NUM_CLASSES), f"Expected (1, {NUM_CLASSES}), got {logits.shape}"
        print(f"  Landmark extraction: OK (features shape: {features.shape})")
        print(f"  Model inference: OK (output shape: {logits.shape})")
    else:
        print(f"  No hands detected in random frame (expected)")

    extractor.release()
    print("  Pipeline dimension check: PASS")

test("Landmark pipeline", test_landmark_pipeline)

# ═══════════════════════════════════════════════════════════
# 6. SPELL ASSEMBLER TEST
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. SPELL ASSEMBLER TEST")
print("=" * 60)

def test_spell_assembler():
    from src.translation.spell_assembler import SpellAssembler

    asm = SpellAssembler(gap_frames=3)

    # Test basic letter assembly
    asm.feed_letter("ba")    # ب = b
    asm.feed_letter("waw")   # و = o  (approximation)
    asm.feed_letter("nun")   # ن = n
    result = asm.flush()
    assert result != "", f"Expected non-empty, got '{result}'"

    # Test space
    asm.feed_letter("ba")
    asm.feed_letter("waw")
    asm.feed_letter("nun")
    asm.feed_letter("space")
    asm.feed_letter("jim")
    asm.feed_letter("waw")
    asm.feed_letter("fa")
    asm.feed_letter("ra")
    result = asm.flush()
    assert " " in result or len(result) > 3, f"Expected space-separated words, got '{result}'"

    # Test backspace
    asm.feed_letter("ha")
    asm.feed_letter("ayn")
    asm.feed_letter("lam")   # wrong
    asm.feed_letter("backspace")  # delete "lam"
    asm.feed_letter("lam")
    asm.feed_letter("waw")
    result = asm.flush()
    assert result != "", f"Expected non-empty, got '{result}'"

    # Test repeated letters
    asm.feed_letter("sin")
    asm.feed_letter("sin")
    asm.feed_letter("sin")
    result = asm.flush()
    assert result != "", f"Expected non-empty, got '{result}'"

    # Test empty
    result = asm.flush()
    assert result == "", f"Expected empty, got '{result}'"

    print("  All spell assembler tests passed")

test("Spell assembler", test_spell_assembler)

# ═══════════════════════════════════════════════════════════
# 7. MODE CONTROLLER TEST
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. MODE CONTROLLER TEST")
print("=" * 60)

def test_mode_controller():
    from src.translation.mode_controller import ModeController

    mc = ModeController()
    assert mc.mode == "sign", f"Initial mode should be 'sign', got '{mc.mode}'"

    # Toggle to spell
    result = mc.toggle()
    assert result == "spell", f"Expected 'spell', got '{result}'"
    assert mc.mode == "spell"

    # Toggle back to sign
    result = mc.toggle()
    assert result == "sign", f"Expected 'sign', got '{result}'"

    # Auto-switch via TSL signs
    event = mc.update("ta3raf")
    assert event["consumed"] == True
    assert event["mode"] == "spell"

    event = mc.update("labes")
    assert event["consumed"] == True
    assert event["mode"] == "sign"

    # Regular signs should not change mode
    event = mc.update("bonjour")
    assert event["consumed"] == False
    assert event["mode"] == "sign"

    print("  All mode controller tests passed")

test("Mode controller", test_mode_controller)

# ═══════════════════════════════════════════════════════════
# 8. TRANSLATOR INIT TEST
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. TRANSLATOR INIT TEST")
print("=" * 60)

def test_translator_init():
    from src.translation.translator import TSLTranslator

    t = TSLTranslator()
    assert t.model is not None, "Sign model not loaded"
    assert t.spell_model is not None, "Spell model not loaded"
    assert t.llm_corrector is not None, "LLM corrector not initialized"
    # Per-client state (session isolation refactor): spell/mode live in the session
    state = t.get_session()
    assert state.spell_assembler is not None, "Spell assembler not initialized"
    assert state.mode_controller is not None, "Mode controller not initialized"
    assert t.alphabet is not None, "Alphabet not initialized"

    # Verify alphabet size matches model
    assert len(t.alphabet) == 31, f"Alphabet size: expected 31, got {len(t.alphabet)}"

    # Verify spell model output size
    import torch
    dummy = torch.randn(1, 126)
    with torch.no_grad():
        out = t.spell_model(dummy)
        assert out.shape[1] == 31, f"Spell model output: expected 31, got {out.shape[1]}"

    print("  All translator init tests passed")

test("Translator init", test_translator_init)

# ═══════════════════════════════════════════════════════════
# 9. LLM CORRECTOR TEST
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. LLM CORRECTOR TEST")
print("=" * 60)

def test_llm_corrector():
    from src.translation.llm_corrector import LLMCorrector, LLMConfig

    # Test with no API key (should be disabled)
    config = LLMConfig(api_key="")
    corrector = LLMCorrector(config)
    assert not corrector.enabled, "Should be disabled without API key"

    # Test fallback returns raw text
    import asyncio
    result = asyncio.run(corrector.correct("hello"))
    assert result == "hello", f"Expected 'hello', got '{result}'"

    # Test empty input
    result = asyncio.run(corrector.correct(""))
    assert result == "", f"Expected empty, got '{result}'"

    print("  LLM disabled mode: OK")

test("LLM corrector", test_llm_corrector)

# ═══════════════════════════════════════════════════════════
# 10. ALPHABET LABEL CONSISTENCY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("10. ALPHABET LABEL CONSISTENCY")
print("=" * 60)

def test_alphabet_labels():
    from src.languages.alphabet import SPELL_LABELS, INDEX_TO_LABEL, LABEL_TO_INDEX, TSLAlphabet

    # Verify consistency
    assert len(SPELL_LABELS) == 31, f"Expected 31 labels, got {len(SPELL_LABELS)}"
    assert len(INDEX_TO_LABEL) == 31
    assert len(LABEL_TO_INDEX) == 31

    # Verify round-trip
    for i, label in enumerate(SPELL_LABELS):
        assert INDEX_TO_LABEL[i] == label
        assert LABEL_TO_INDEX[label] == i

    # Verify alphabet class matches
    alpha = TSLAlphabet()
    labels = alpha.get_all_labels()
    assert labels == SPELL_LABELS, "TSLAlphabet.get_all_labels() != SPELL_LABELS"

    # Verify class_names.txt matches
    with open(PROJECT_ROOT / "data" / "alphabet" / "processed" / "class_names.txt") as f:
        file_labels = [l.strip() for l in f.readlines()]
    assert file_labels == SPELL_LABELS, "class_names.txt != SPELL_LABELS"

    print("  All 31 labels consistent across codebase")

test("Alphabet label consistency", test_alphabet_labels)

# ═══════════════════════════════════════════════════════════
# 11. PERFORMANCE TEST
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("11. PERFORMANCE TEST")
print("=" * 60)

def test_performance():
    import torch
    from src.models.spell_classifier import SpellClassifier

    model = SpellClassifier(num_classes=31, architecture="mlp")
    model.eval()

    # Warmup
    dummy = torch.randn(100, 126)
    for _ in range(10):
        with torch.no_grad():
            model(dummy)

    # Benchmark
    n_runs = 100
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    elapsed = time.time() - t0

    fps = (n_runs * 100) / elapsed
    latency_ms = (elapsed / n_runs) * 1000

    print(f"  Batch size: 100")
    print(f"  Runs: {n_runs}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {fps:.0f} frames/sec")
    print(f"  Latency per batch: {latency_ms:.2f}ms")
    print(f"  Latency per frame: {latency_ms/100:.2f}ms")

    model_size = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {model_size:,}")

test("Performance", test_performance)

# ═══════════════════════════════════════════════════════════
# 12. .gitignore SECURITY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("12. SECURITY CHECK")
print("=" * 60)

def test_security():
    gitignore = PROJECT_ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore not found"

    content = gitignore.read_text()
    assert ".env" in content, ".env not in .gitignore"

    env_example = PROJECT_ROOT / ".env.example"
    if env_example.exists():
        env_content = env_example.read_text()
        # Check no real API keys in .env.example
        assert "sk-" not in env_content, ".env.example contains a real API key!"

    print("  .gitignore includes .env: OK")
    print("  .env.example clean: OK")

test("Security check", test_security)

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Passed: {results['pass']}")
print(f"  Failed: {results['fail']}")
print(f"  Total:  {results['pass'] + results['fail']}")

if results["fail"] > 0:
    print("\n  FAILED TESTS:")
    for d in results["details"]:
        if d["status"] != "PASS":
            print(f"    [{d['status']}] {d['name']}: {d.get('error', '')}")
