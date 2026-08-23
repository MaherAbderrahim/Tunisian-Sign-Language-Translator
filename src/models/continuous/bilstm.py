"""BiLSTM + Temporal Attention baseline for continuous sign recognition."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.continuous.base import ContinuousSignModel
from src.models.continuous.attention import TemporalAttention


class BiLSTMAttentionModel(ContinuousSignModel):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 126,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__(num_classes, input_dim)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = TemporalAttention(out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.attention(out, mask=mask)
