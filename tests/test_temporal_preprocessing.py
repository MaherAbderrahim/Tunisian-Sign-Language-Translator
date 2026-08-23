"""Tests for V2 preprocessing modules."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from src.preprocessing.normalization import normalize_frame_wrist, normalize_landmarks_sequence
from src.preprocessing.temporal import (
    compute_velocity,
    compute_acceleration,
    build_enhanced_features,
    pad_sequence_2d,
    collate_padded,
)


class TestNormalization:
    def test_frame_shape(self):
        frame = np.random.randn(126).astype(np.float32)
        result = normalize_frame_wrist(frame)
        assert result.shape == (126,)

    def test_frame_range(self):
        frame = np.random.randn(126).astype(np.float32) * 10
        result = normalize_frame_wrist(frame)
        assert result.min() >= -5.0
        assert result.max() <= 5.0

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            normalize_frame_wrist(np.random.randn(128))

    def test_sequence(self):
        seq = np.random.randn(10, 126).astype(np.float32)
        result = normalize_landmarks_sequence(seq)
        assert result.shape == (10, 126)
        assert result.min() >= -5.0


class TestTemporal:
    def test_velocity(self):
        seq = np.random.randn(8, 126).astype(np.float32)
        vel = compute_velocity(seq)
        assert vel.shape == (7, 126)

    def test_acceleration(self):
        vel = np.random.randn(7, 126).astype(np.float32)
        acc = compute_acceleration(vel)
        assert acc.shape == (6, 126)

    def test_enhanced_features(self):
        seq = np.random.randn(8, 126).astype(np.float32)
        feat, length = build_enhanced_features(seq, include_velocity=True, include_acceleration=True)
        assert length == 6
        assert feat.shape[0] == 6
        assert feat.shape[1] == 126 * 3  # pos + vel + acc

    def test_enhanced_no_accel(self):
        seq = np.random.randn(8, 126).astype(np.float32)
        feat, length = build_enhanced_features(seq, include_velocity=True, include_acceleration=False)
        assert feat.shape[1] == 126 * 2

    def test_enhanced_positions_only(self):
        seq = np.random.randn(8, 126).astype(np.float32)
        feat, length = build_enhanced_features(seq, include_velocity=False, include_acceleration=False)
        assert feat.shape[1] == 126

    def test_pad_shorter(self):
        seq = np.random.randn(5, 10).astype(np.float32)
        padded, mask = pad_sequence_2d(seq, 10)
        assert padded.shape == (10, 10)
        assert mask[:5].all()
        assert not mask[5:].any()

    def test_pad_longer(self):
        seq = np.random.randn(12, 10).astype(np.float32)
        padded, mask = pad_sequence_2d(seq, 10)
        assert padded.shape == (10, 10)
        assert mask.all()

    def test_collate(self):
        batch = [
            (np.random.randn(5, 126).astype(np.float32), 0),
            (np.random.randn(8, 126).astype(np.float32), 1),
        ]
        X, M, Y = collate_padded(batch, max_len=8)
        assert X.shape == (2, 8, 126)
        assert M.shape == (2, 8)
        assert Y.shape == (2,)
