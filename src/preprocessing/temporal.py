"""Temporal preprocessing utilities for V2 continuous sign recognition.

Provides:
- velocity / acceleration computation
- fixed-length padding with masks
- collate function for DataLoader with variable-length sequences
"""
from __future__ import annotations

from typing import Tuple, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def compute_velocity(landmarks: np.ndarray) -> np.ndarray:
    if landmarks.ndim != 2:
        raise ValueError(f"Expected 2D array, got {landmarks.ndim}D")
    return np.diff(landmarks, axis=0).astype(np.float32)


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    if velocity.ndim != 2:
        raise ValueError(f"Expected 2D array, got {velocity.ndim}D")
    return np.diff(velocity, axis=0).astype(np.float32)


def build_enhanced_features(
    landmarks: np.ndarray,
    *,
    include_velocity: bool = True,
    include_acceleration: bool = True,
) -> Tuple[np.ndarray, int]:
    parts = [landmarks]
    if include_velocity:
        vel = compute_velocity(landmarks)
        parts.append(vel)
    if include_acceleration and len(parts) > 1:
        acc = compute_acceleration(parts[-1])
        parts.append(acc)
    min_t = min(p.shape[0] for p in parts)
    return np.concatenate([p[:min_t] for p in parts], axis=1), min_t


def pad_sequence_2d(
    seq: np.ndarray,
    max_len: int,
    *,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    T = seq.shape[0]
    F = seq.shape[1]
    mask = np.zeros(max_len, dtype=bool)
    if T >= max_len:
        return seq[:max_len], np.ones(max_len, dtype=bool)
    padded = np.full((max_len, F), pad_value, dtype=seq.dtype)
    padded[:T] = seq
    mask[:T] = True
    return padded, mask


def collate_padded(
    batch: List[Tuple[np.ndarray, int]],
    max_len: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    if max_len <= 0:
        max_len = max(s.shape[0] for s in sequences)
    padded = []
    masks = []
    for s in sequences:
        p, m = pad_sequence_2d(s, max_len)
        padded.append(p)
        masks.append(m)
    X = torch.tensor(np.stack(padded), dtype=torch.float32)
    M = torch.tensor(np.stack(masks), dtype=torch.bool)
    Y = torch.tensor(labels, dtype=torch.long)
    return X, M, Y
