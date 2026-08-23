"""Normalize landmarks per frame using a wrist-centric approach (V2 extension).

The normalization is intentionally different from the V1 per-frame z-score
(normalize_features) because temporal models benefit from a coordinate
system anchored to the dominant hand's wrist.  The V1 classifier remains
completely untouched.
"""
from __future__ import annotations

import numpy as np


# ---------- single-frame normalization (wrist-centric) ---------- #

WRIST_LEFT = 0
WRIST_RIGHT = 21


def normalize_frame_wrist(frame: np.ndarray) -> np.ndarray:
    """Center a single (126,) frame on the average wrist position.

    Steps:
        1. Locate both wrists (indices 0 and 21 of the flattened vector,
           each triplet is (x, y, z)).
        2. Compute the midpoint of both wrists.
        3. Subtract this midpoint from every landmark.
        4. Scale each hand by the maximum wrist-to-farthest-landmark distance
           so that both hands occupy roughly the same dynamic range.
        5. Clamp extreme outliers to [-5, 5] (same range V1 expects).

    Returns a (126,) vector with the same layout as the V1 features.
    """
    if frame.shape != (126,):
        raise ValueError(f"Expected (126,) frame, got {frame.shape}")

    coords = frame.reshape(42, 3)  # (42, 3): 21 left + 21 right

    wrist_l = coords[WRIST_LEFT]   # (3,)
    wrist_r = coords[WRIST_RIGHT]  # (3,)
    midpoint = (wrist_l + wrist_r) / 2.0

    centered = coords - midpoint  # center both hands

    # scale by max spread of each hand
    spread_l = np.max(np.linalg.norm(coords[:21] - wrist_l, axis=-1))
    spread_r = np.max(np.linalg.norm(coords[21:] - wrist_r, axis=-1))
    max_spread = max(spread_l, spread_r, 1e-6)

    scaled = centered / max_spread
    return np.clip(scaled.reshape(126), -5.0, 5.0).astype(np.float32)


# ---------- sequence normalization ---------- #

def normalize_landmarks_sequence(seq: np.ndarray) -> np.ndarray:
    """Normalize every frame of a (T, 126) sequence independently.

    For each frame the normalization is:
        - subtract the average wrist position (both hands)
        - scale by the max spread of either hand
        - clamp to [-5, 5]

    This keeps V1-compatible feature ranges while being robust to
    camera zoom / user distance.
    """
    if seq.ndim != 2 or seq.shape[1] != 126:
        raise ValueError(f"Expected (T, 126), got {seq.shape}")

    return np.stack([normalize_frame_wrist(f) for f in seq], axis=0)


def normalize_landmarks_frame(frame: np.ndarray) -> np.ndarray:
    """Convenience wrapper for a single (126,) frame."""
    return normalize_frame_wrist(frame)
