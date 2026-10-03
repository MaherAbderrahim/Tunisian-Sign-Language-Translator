"""Video → landmark sequence extraction (V2 extension).

Provides ``extract_landmarks_sequence`` which processes a video file and
returns a (T, 126) numpy array suitable for the V2 temporal models.

The function re-uses the existing ``MediaPipeLandmarkExtractor`` (V1) as-is.
No V1 code is modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.models.landmark_extractor import MediaPipeLandmarkExtractor


def extract_landmarks_sequence(
    video_path: str | Path,
    *,
    max_frames: Optional[int] = None,
    target_fps: Optional[int] = None,
    min_detection_confidence: float = 0.5,
) -> dict:
    """Extract a (T, 126) landmark sequence from a video file.

    Parameters
    ----------
    video_path : str or Path
        Path to an mp4/avi/etc. video file.
    max_frames : int, optional
        If set, stop after this many frames (truncation).
    target_fps : int, optional
        If set, skip frames to match this target frame rate.
    min_detection_confidence : float
        Passed through to MediaPipeLandmarkExtractor.

    Returns
    -------
    dict with keys:
        - "landmarks" : np.ndarray of shape (T, 126) or None if no hands detected.
        - "fps"       : float – actual video FPS.
        - "n_frames"  : int – total frames in the video.
        - "n_valid"   : int – frames where at least one hand was detected.
        - "status"    : "ok" | "no_hands" | "error".
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {"landmarks": None, "fps": 0.0, "n_frames": 0, "n_valid": 0,
                "status": "error", "error": f"File not found: {video_path}"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"landmarks": None, "fps": 0.0, "n_frames": 0, "n_valid": 0,
                "status": "error", "error": f"Cannot open video: {video_path}"}

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # compute frame skip to reach target_fps (approximate)
    frame_skip = 1
    if target_fps and video_fps > 0:
        frame_skip = max(1, round(video_fps / target_fps))

    extractor = MediaPipeLandmarkExtractor(min_detection_confidence=min_detection_confidence)
    landmarks_list: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # extract_features flattens + pads to a fixed 126-vector, so stacking
        # is always safe even when the number of detected hands varies between
        # frames. normalize=False because normalize_landmarks_sequence does its
        # own wrist-centering downstream.
        features = extractor.extract_features(frame, normalize=False)
        if features is not None:
            landmarks_list.append(features)

        if max_frames is not None and len(landmarks_list) >= max_frames:
            break
        frame_idx += 1

    cap.release()

    if not landmarks_list:
        return {"landmarks": None, "fps": video_fps, "n_frames": total_frames,
                "n_valid": 0, "status": "no_hands"}

    return {
        "landmarks": np.stack(landmarks_list, axis=0).astype(np.float32),
        "fps": video_fps,
        "n_frames": total_frames,
        "n_valid": len(landmarks_list),
        "status": "ok",
    }


def extract_landmarks_from_frames(
    frames: list[np.ndarray],
    *,
    min_detection_confidence: float = 0.5,
) -> np.ndarray | None:
    """Extract landmarks from a list of BGR frames (in memory).

    Returns (T, 126) array or None if no hands were detected in any frame.
    """
    extractor = MediaPipeLandmarkExtractor(min_detection_confidence=min_detection_confidence)
    result = []
    for frame in frames:
        # Same 126-row contract as extract_landmarks_sequence (see above).
        features = extractor.extract_features(frame, normalize=False)
        if features is not None:
            result.append(features)
    if not result:
        return None
    return np.stack(result, axis=0).astype(np.float32)
