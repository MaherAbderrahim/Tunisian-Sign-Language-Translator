"""Abstract base class for V2 temporal sign models."""
from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousSignModel(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 126):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

    @abstractmethod
    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encode(x, mask=mask)
        return self.classifier(h)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x, mask=mask), dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self.eval()
        return self.forward(x, mask=mask).argmax(dim=-1)

    @torch.no_grad()
    def predict_topk(self, x: torch.Tensor, mask: torch.Tensor | None = None, k: int = 3):
        self.eval()
        logits = self.forward(x, mask=mask)
        return torch.topk(logits, k, dim=-1)
