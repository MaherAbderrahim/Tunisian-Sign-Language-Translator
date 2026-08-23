"""Real-time continuous sign inference engine (V2)."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch

from src.continuous_translation.pipeline import ContinuousSignPipeline
from src.config.settings import (
    TEMPORAL_INPUT_DIM,
    CONTINUOUS_CONFIDENCE_THRESHOLD,
)


class ContinuousInferenceEngine:
    def __init__(
        self,
        pipeline: ContinuousSignPipeline,
        *,
        buffer_frames: int = 32,
        smoothing_window: int = 5,
    ):
        self.pipeline = pipeline
        self.buffer_frames = buffer_frames
        self.smoothing_window = smoothing_window
        self.landmark_buffer: deque[np.ndarray] = deque(maxlen=buffer_frames)
        self.prediction_history: deque[int] = deque(maxlen=smoothing_window)

    def reset(self):
        self.landmark_buffer.clear()
        self.prediction_history.clear()

    def feed_landmarks(self, frame_landmarks: np.ndarray) -> Optional[dict]:
        if frame_landmarks.shape != (126,):
            raise ValueError(f"Expected (126,), got {frame_landmarks.shape}")
        self.landmark_buffer.append(frame_landmarks)
        if len(self.landmark_buffer) < 10:
            return None
        seq = np.stack(list(self.landmark_buffer), axis=0)
        result = self.pipeline.predict_temporal(seq)
        self.prediction_history.append(result["class_idx"])
        result["smoothed_class"] = self._majority_vote()
        return result

    def _majority_vote(self) -> int:
        if not self.prediction_history:
            return -1
        counts: dict[int, int] = {}
        for c in self.prediction_history:
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=counts.get)
