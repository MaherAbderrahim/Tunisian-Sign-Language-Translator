"""V1 TSLClassifier adapter for per-frame inference in V2 sequences.

This module loads the existing V1 model and uses it as a frame-level
classifier.  V1 itself is never modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config.settings import DEFAULT_MODEL_PATH, TEMPORAL_INPUT_DIM


class V1SequenceAdapter:
    def __init__(self, model_path: Optional[str | Path] = None):
        from src.models.classifier import TSLClassifier
        model_path = Path(model_path or DEFAULT_MODEL_PATH)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.class_names: list[str] = ckpt["class_names"]
        self.num_classes = len(self.class_names)
        model = TSLClassifier(
            num_classes=self.num_classes,
            input_size=TEMPORAL_INPUT_DIM,
            dropout_rate=ckpt.get("dropout_rate", 0.5),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        self.model = model.eval()
        self.device = torch.device("cpu")
        self._last_sequence: Optional[np.ndarray] = None
        self._last_probs: Optional[np.ndarray] = None

    @torch.no_grad()
    def predict_frame(self, frame: np.ndarray) -> tuple[int, float]:
        if frame.shape != (126,):
            raise ValueError(f"Expected (126,), got {frame.shape}")
        t = torch.tensor(frame, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.model(t)
        probs = torch.softmax(logits, dim=-1)
        cls = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, cls].item())
        return cls, conf

    @torch.no_grad()
    def per_frame_predict(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if sequence.ndim != 2 or sequence.shape[1] != 126:
            raise ValueError(f"Expected (T, 126), got {sequence.shape}")
        t = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        logits = self.model(t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        return preds, probs

    def detect_stable_segments(
        self,
        sequence: np.ndarray,
        *,
        min_stable: int = 4,
        min_gap: int = 2,
        confidence_threshold: float = 0.5,
    ) -> list[dict]:
        preds, probs = self.per_frame_predict(sequence)
        confs = probs[np.arange(len(preds)), preds]
        stable: list[dict] = []
        i = 0
        T = len(preds)
        while i < T:
            if confs[i] < confidence_threshold:
                i += 1
                continue
            cls = preds[i]
            start = i
            while i < T and preds[i] == cls and confs[i] >= confidence_threshold:
                i += 1
            length = i - start
            if length >= min_stable:
                stable.append({
                    "class_idx": int(cls),
                    "class_name": self.class_names[cls],
                    "start": start,
                    "end": i - 1,
                    "length": length,
                    "mean_confidence": float(confs[start:i].mean()),
                })
        return stable

    def predict_sign_sequence(
        self,
        sequence: np.ndarray,
        *,
        stable_frames: int = 4,
        gap_frames: int = 2,
    ) -> list[dict]:
        return self.detect_stable_segments(
            sequence, min_stable=stable_frames, min_gap=gap_frames,
        )

    def predict_majority_vote(self, sequence: np.ndarray, window: int = 5) -> tuple[int, float]:
        preds, probs = self.per_frame_predict(sequence)
        if len(preds) == 0:
            return -1, 0.0
        from collections import Counter
        counts = Counter(preds.tolist())
        best_cls, best_count = counts.most_common(1)[0]
        return best_cls, best_count / len(preds)
