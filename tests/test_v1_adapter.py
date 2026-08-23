"""Tests for V1 adapter and V1 compatibility."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from src.config.settings import DEFAULT_MODEL_PATH


class TestV1Adapter:
    def _skip_if_no_model(self):
        if not DEFAULT_MODEL_PATH.exists():
            pytest.skip("V1 model checkpoint not found")

    def test_load(self):
        self._skip_if_no_model()
        from src.models.continuous.v1_adapter import V1SequenceAdapter
        adapter = V1SequenceAdapter()
        assert adapter.num_classes > 0
        assert len(adapter.class_names) == adapter.num_classes

    def test_predict_frame(self):
        self._skip_if_no_model()
        from src.models.continuous.v1_adapter import V1SequenceAdapter
        adapter = V1SequenceAdapter()
        frame = np.random.randn(126).astype(np.float32)
        cls, conf = adapter.predict_frame(frame)
        assert 0 <= cls < adapter.num_classes
        assert 0.0 <= conf <= 1.0

    def test_per_frame_predict(self):
        self._skip_if_no_model()
        from src.models.continuous.v1_adapter import V1SequenceAdapter
        adapter = V1SequenceAdapter()
        seq = np.random.randn(5, 126).astype(np.float32)
        preds, probs = adapter.per_frame_predict(seq)
        assert preds.shape == (5,)
        assert probs.shape == (5, adapter.num_classes)

    def test_majority_vote(self):
        self._skip_if_no_model()
        from src.models.continuous.v1_adapter import V1SequenceAdapter
        adapter = V1SequenceAdapter()
        seq = np.random.randn(10, 126).astype(np.float32)
        cls, conf = adapter.predict_majority_vote(seq)
        assert -1 <= cls < adapter.num_classes
        assert 0.0 <= conf <= 1.0
