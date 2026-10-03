"""Tests for SpellClassifier model."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import numpy as np
import pytest
from src.models.spell_classifier import SpellClassifier


class TestSpellClassifier:
    def test_mlp_forward(self):
        model = SpellClassifier(num_classes=31, architecture="mlp")
        model.eval()
        x = torch.randn(1, 126)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 31)

    def test_mlp_batch(self):
        model = SpellClassifier(num_classes=31, architecture="mlp")
        model.eval()
        x = torch.randn(32, 126)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32, 31)

    def test_prediction(self):
        model = SpellClassifier(num_classes=31, architecture="mlp")
        model.eval()
        x = torch.randn(1, 126)
        pred = model.predict(x)
        assert pred.shape == (1,)
        assert 0 <= pred.item() < 31

    def test_predict_proba(self):
        model = SpellClassifier(num_classes=31, architecture="mlp")
        model.eval()
        x = torch.randn(1, 126)
        probs = model.predict_proba(x)
        assert probs.shape == (1, 31)
        assert abs(probs.sum().item() - 1.0) < 0.01

    def test_load_trained_model(self):
        model_path = Path(__file__).resolve().parent.parent / "models" / "alphabet" / "spell_classifier_best.pth"
        if not model_path.exists():
            pytest.skip("model checkpoint not present")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model = SpellClassifier(num_classes=31, architecture="mlp")
        model.load_state_dict(state_dict)
        model.eval()
        x = torch.randn(1, 126)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 31)

    def test_param_count(self):
        model = SpellClassifier(num_classes=31, architecture="mlp")
        params = sum(p.numel() for p in model.parameters())
        assert params > 100000  # Should be around 235K
        assert params < 1000000
