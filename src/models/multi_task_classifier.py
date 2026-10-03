"""
Multi-Task Classifier (Version B) - Single model for signs + alphabet.

Shared backbone with two classification heads:
  - Sign head: 57 TSL signs
  - Spell head: 31 finger spelling classes (28 letters + space + backspace + finish)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiTaskClassifier(nn.Module):
    """
    Classificateur multi-tâches: 57 signes LST + 31 lettres alphabet = 88 classes totales.

    Architecture:
    - Backbone commun partagé (126 -> 512 -> 256 -> 128)
    - Deux têtes de classification indépendantes:
      - sign_head: 128 -> 57 (signes LST)
      - spell_head: 128 -> 31 (lettres alphabet)
    """

    def __init__(
        self,
        num_sign_classes: int = 57,
        num_spell_classes: int = 31,
        input_size: int = 126,
        dropout_rate: float = 0.2,
    ):
        super(MultiTaskClassifier, self).__init__()

        self.num_sign_classes = num_sign_classes
        self.num_spell_classes = num_spell_classes

        # Shared backbone
        self.backbone = nn.Sequential(
            # Input block
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Residual block 1
            _ResidualBlock(512, 512, dropout_rate),
            # Downsample
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Residual block 2
            _ResidualBlock(256, 256, dropout_rate),
            # Downsample to features
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Sign classification head (57 classes)
        self.sign_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, num_sign_classes),
        )

        # Spell classification head (38 classes)
        self.spell_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, num_spell_classes),
        )

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, 126) landmark features
        Returns:
            sign_logits: (batch_size, num_sign_classes)
            spell_logits: (batch_size, num_spell_classes)
        """
        features = self.backbone(x)
        sign_logits = self.sign_head(features)
        spell_logits = self.spell_head(features)
        return sign_logits, spell_logits

    def predict_sign(self, x: torch.Tensor) -> torch.Tensor:
        """Prédit le signe LST (mode signe)."""
        self.eval()
        with torch.no_grad():
            sign_logits, _ = self.forward(x)
            return torch.argmax(sign_logits, dim=-1)

    def predict_spell(self, x: torch.Tensor) -> torch.Tensor:
        """Prédit la lettre alphabet (mode doigté)."""
        self.eval()
        with torch.no_grad():
            _, spell_logits = self.forward(x)
            return torch.argmax(spell_logits, dim=-1)

    def predict(self, x: torch.Tensor, mode: str = "sign") -> torch.Tensor:
        """
        Prédit selon le mode.

        Args:
            x: (batch_size, 126)
            mode: "sign" ou "spell"
        """
        if mode == "sign":
            return self.predict_sign(x)
        elif mode == "spell":
            return self.predict_spell(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def predict_proba(self, x: torch.Tensor, mode: str = "sign") -> torch.Tensor:
        """Retourne les probabilités selon le mode."""
        self.eval()
        with torch.no_grad():
            if mode == "sign":
                sign_logits, _ = self.forward(x)
                return F.softmax(sign_logits, dim=-1)
            elif mode == "spell":
                _, spell_logits = self.forward(x)
                return F.softmax(spell_logits, dim=-1)
            else:
                raise ValueError(f"Unknown mode: {mode}")


class _ResidualBlock(nn.Module):
    """Simple residual block: Linear -> BN -> ReLU -> Dropout -> Linear -> BN."""

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))
