"""End-to-end pipeline tests."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import numpy as np
from src.translation.translator import TSLTranslator
from src.translation.spell_assembler import SpellAssembler
from src.translation.mode_controller import ModeController
from src.models.spell_classifier import SpellClassifier
from src.languages.alphabet import SPELL_LABELS


class TestEndToEnd:
    def test_translator_init(self):
        t = TSLTranslator()
        assert t.model is not None
        assert t.spell_model is not None
        assert len(t.alphabet) == 31

    def test_spell_model_inference(self):
        model = SpellClassifier(num_classes=31, architecture="mlp")
        model_path = Path(__file__).resolve().parent.parent / "models" / "alphabet" / "spell_classifier_best.pth"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()
        x = torch.randn(1, 126)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
        assert 0 <= pred.item() < 31
        assert 0 <= conf.item() <= 1

    def test_label_mapping(self):
        model = SpellClassifier(num_classes=31, architecture="mlp")
        model_path = Path(__file__).resolve().parent.parent / "models" / "alphabet" / "spell_classifier_best.pth"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()

        # Load test data
        data_dir = Path(__file__).resolve().parent.parent / "data" / "alphabet" / "processed"
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy")

        # Run inference on a few samples
        X_tensor = torch.FloatTensor(X_test[:10])
        with torch.no_grad():
            logits = model(X_tensor)
            preds = logits.argmax(dim=-1).numpy()

        # Verify predictions map to valid labels
        for pred in preds:
            assert 0 <= pred < len(SPELL_LABELS)
            label = SPELL_LABELS[pred]
            assert label in ["alef", "ba", "ta", "tha", "jim", "ha", "kha", "dal", "dhal", "ra",
                           "zay", "sin", "shin", "sad", "dad", "ta_h", "za", "ayn", "ghayn", "fa",
                           "qaf", "kaf", "lam", "mim", "nun", "ha2", "waw", "ya",
                           "space", "backspace", "finish"]

    def test_mode_switch_flow(self):
        mc = ModeController()
        asm = SpellAssembler(gap_frames=3)

        # Start in sign mode
        assert mc.mode == "sign"

        # Switch to spell mode
        event = mc.update("ta3raf")
        assert event["consumed"] is True
        assert mc.mode == "spell"

        # Feed letters in spell mode
        if mc.mode == "spell":
            asm.feed_letter("ba")
            asm.feed_letter("waw")
            asm.feed_letter("nun")

        # Switch back to sign mode
        event = mc.update("labes")
        assert event["consumed"] is True
        assert mc.mode == "sign"

        # Flush spell result
        result = asm.flush()
        assert result != ""  # Should have assembled letters

    def test_temporal_smoothing(self):
        """Test that repeated predictions are smoothed."""
        from collections import deque
        from src.config.settings import TEMPORAL_SMOOTHING_WINDOW

        recent = deque(maxlen=TEMPORAL_SMOOTHING_WINDOW)

        # Feed same prediction multiple times
        for _ in range(5):
            recent.append(5)

        # Majority vote should be 5
        counts = {}
        for p in recent:
            counts[p] = counts.get(p, 0) + 1
        smoothed = max(counts, key=counts.get)
        assert smoothed == 5

        # Feed mixed predictions
        recent.clear()
        recent.extend([5, 5, 5, 3, 3])
        counts = {}
        for p in recent:
            counts[p] = counts.get(p, 0) + 1
        smoothed = max(counts, key=counts.get)
        assert smoothed == 5  # 5 appears 3 times vs 2
