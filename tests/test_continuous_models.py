"""Tests for V2 continuous sign models."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from src.models.continuous.attention import TemporalAttention
from src.models.continuous.bilstm import BiLSTMAttentionModel
from src.models.continuous.gru import GRUAttentionModel


class TestAttention:
    def test_output_shape(self):
        attn = TemporalAttention(64)
        x = torch.randn(4, 10, 64)
        mask = torch.ones(4, 10, dtype=torch.bool)
        out = attn(x, mask=mask)
        assert out.shape == (4, 64)

    def test_masking(self):
        attn = TemporalAttention(32)
        x = torch.randn(2, 8, 32)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, :3] = True
        out = attn(x, mask=mask)
        assert out.shape == (2, 32)


class TestBiLSTM:
    def test_forward_shape(self):
        model = BiLSTMAttentionModel(num_classes=8, input_dim=126, hidden_dim=64, num_layers=1)
        x = torch.randn(2, 10, 126)
        mask = torch.ones(2, 10, dtype=torch.bool)
        logits = model(x, mask=mask)
        assert logits.shape == (2, 8)

    def test_bidirectional_true(self):
        model = BiLSTMAttentionModel(num_classes=5, input_dim=126, hidden_dim=32, bidirectional=True)
        x = torch.randn(1, 6, 126)
        logits = model(x)
        assert logits.shape == (1, 5)

    def test_bidirectional_false(self):
        model = BiLSTMAttentionModel(num_classes=5, input_dim=126, hidden_dim=32, bidirectional=False)
        x = torch.randn(1, 6, 126)
        logits = model(x)
        assert logits.shape == (1, 5)

    def test_predict(self):
        model = BiLSTMAttentionModel(num_classes=3, input_dim=126, hidden_dim=32, num_layers=1)
        x = torch.randn(2, 5, 126)
        preds = model.predict(x)
        assert preds.shape == (2,)

    def test_predict_proba(self):
        model = BiLSTMAttentionModel(num_classes=3, input_dim=126, hidden_dim=32, num_layers=1)
        x = torch.randn(2, 5, 126)
        proba = model.predict_proba(x)
        assert proba.shape == (2, 3)
        assert torch.allclose(proba.sum(-1), torch.ones(2), atol=1e-5)

    def test_predict_topk(self):
        model = BiLSTMAttentionModel(num_classes=5, input_dim=126, hidden_dim=32, num_layers=1)
        x = torch.randn(2, 5, 126)
        values, indices = model.predict_topk(x, k=3)
        assert values.shape == (2, 3)
        assert indices.shape == (2, 3)


class TestGRU:
    def test_forward_shape(self):
        model = GRUAttentionModel(num_classes=8, input_dim=126, hidden_dim=64, num_layers=1)
        x = torch.randn(2, 10, 126)
        mask = torch.ones(2, 10, dtype=torch.bool)
        logits = model(x, mask=mask)
        assert logits.shape == (2, 8)

    def test_predict(self):
        model = GRUAttentionModel(num_classes=4, input_dim=126, hidden_dim=32, num_layers=1)
        x = torch.randn(2, 5, 126)
        preds = model.predict(x)
        assert preds.shape == (2,)

    def test_parameter_count_bilstm_vs_gru(self):
        bilstm = BiLSTMAttentionModel(8, 126, 64, 1)
        gru = GRUAttentionModel(8, 126, 64, 1)
        # GRU should have fewer parameters than BiLSTM
        gru_params = sum(p.numel() for p in gru.parameters())
        bilstm_params = sum(p.numel() for p in bilstm.parameters())
        assert gru_params < bilstm_params

    def test_variable_length(self):
        model = BiLSTMAttentionModel(num_classes=3, input_dim=126, hidden_dim=32, num_layers=1)
        x1 = torch.randn(1, 3, 126)
        x2 = torch.randn(1, 8, 126)
        m1 = torch.ones(1, 3, dtype=torch.bool)
        m2 = torch.ones(1, 8, dtype=torch.bool)
        out1 = model(x1, mask=m1)
        out2 = model(x2, mask=m2)
        assert out1.shape == (1, 3)
        assert out2.shape == (1, 3)
