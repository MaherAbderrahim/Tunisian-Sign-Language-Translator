"""End-to-end continuous sign recognition pipeline (V2)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config.settings import (
    CONTINUOUS_CONFIDENCE_THRESHOLD,
    TEMPORAL_INPUT_DIM,
)
from src.models.continuous.base import ContinuousSignModel
from src.models.continuous.bilstm import BiLSTMAttentionModel
from src.models.continuous.gru import GRUAttentionModel
from src.models.continuous.v1_adapter import V1SequenceAdapter
from src.preprocessing.landmarks import extract_landmarks_sequence
from src.preprocessing.normalization import normalize_landmarks_sequence
from src.preprocessing.temporal import pad_sequence_2d


MODEL_REGISTRY = {
    "bilstm": BiLSTMAttentionModel,
    "gru": GRUAttentionModel,
}


class ContinuousSignPipeline:
    def __init__(
        self,
        temporal_model: Optional[ContinuousSignModel] = None,
        model_type: str = "bilstm",
        v1_adapter: Optional[V1SequenceAdapter] = None,
    ):
        self.v1_adapter = v1_adapter or V1SequenceAdapter()
        self.temporal_model = temporal_model
        self.model_type = model_type

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, model_type: str = "bilstm") -> "ContinuousSignPipeline":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        num_classes = ckpt["num_classes"]
        model_cls = MODEL_REGISTRY[model_type]
        model = model_cls(
            num_classes=num_classes,
            input_dim=ckpt.get("input_dim", TEMPORAL_INPUT_DIM),
            hidden_dim=ckpt.get("hidden_dim", 128),
            num_layers=ckpt.get("num_layers", 2),
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return cls(temporal_model=model, model_type=model_type)

    def process_video(
        self,
        video_path: str | Path,
        *,
        max_frames: Optional[int] = None,
        target_fps: Optional[int] = None,
    ) -> dict:
        result = extract_landmarks_sequence(
            video_path, max_frames=max_frames, target_fps=target_fps,
        )
        if result["status"] != "ok":
            return result
        seq = normalize_landmarks_sequence(result["landmarks"])
        result["normalized_landmarks"] = seq
        result["v1_segments"] = self.v1_adapter.predict_sign_sequence(seq)
        if self.temporal_model is not None:
            result["temporal_prediction"] = self.predict_temporal(seq)
        return result

    @torch.no_grad()
    def predict_temporal(
        self,
        normalized_sequence: np.ndarray,
        *,
        max_len: int = 32,
    ) -> dict:
        padded, mask = pad_sequence_2d(normalized_sequence, max_len)
        x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        logits = self.temporal_model(x, mask=m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        cls = int(probs.argmax())
        conf = float(probs[cls])
        if conf < CONTINUOUS_CONFIDENCE_THRESHOLD:
            return {"class_idx": cls, "class_name": "UNKNOWN", "confidence": conf, "all_probs": probs}
        class_names = getattr(self.temporal_model, "class_names", [str(i) for i in range(len(probs))])
        return {"class_idx": cls, "class_name": class_names[cls], "confidence": conf, "all_probs": probs}
