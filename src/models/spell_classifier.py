"""
Spell Classifier - Separate model for finger spelling.

Supports two architectures:
  - "mlp": Simple MLP (from train_alphabet_models.py) — used by trained checkpoints
  - "residual": Residual MLP with skip connections — original SpellClassifier

Architecture mirrors TSLClassifier but with configurable output classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpellClassifier(nn.Module):
    """
    Classificateur MLP pour l'alphabet TSL (doigté).

    Supports two architectures selected via `architecture` parameter:
      - "mlp": Simple Sequential MLP (126→512→256→128→num_classes)
      - "residual": Residual MLP with skip connections
    """

    def __init__(
        self,
        num_classes: int = 31,
        input_size: int = 126,
        dropout_rate: float = 0.3,
        architecture: str = "mlp",
    ):
        super(SpellClassifier, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.architecture = architecture

        if architecture == "mlp":
            # Simple MLP matching train_alphabet_models.py MLPModel
            self.net = nn.Sequential(
                nn.Linear(input_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
        elif architecture == "residual":
            # Original residual architecture
            self.input_block = nn.Sequential(
                nn.Linear(input_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            )
            self.res1 = nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(512, 512), nn.BatchNorm1d(512),
            )
            self.down1 = nn.Sequential(
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            )
            self.res2 = nn.Sequential(
                nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(256, 256), nn.BatchNorm1d(256),
            )
            self.down2 = nn.Sequential(
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate / 2),
                nn.Linear(64, num_classes),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "mlp":
            return self.net(x)

        h = self.input_block(x)
        h = h + self.res1(h)
        h = F.relu(h)
        h = self.down1(h)
        h = h + self.res2(h)
        h = F.relu(h)
        h = self.down2(h)
        return self.classifier(h)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
