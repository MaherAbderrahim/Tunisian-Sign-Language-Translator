"""
MediaPipe Landmark Extractor for Hand Gesture Recognition
Uses the new MediaPipe Task API (v0.10+)
"""

import numpy as np
import cv2
from typing import Optional, List
from pathlib import Path
import mediapipe as mp

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),             # palm
]


class MediaPipeLandmarkExtractor:
    """
    Extracteur de landmarks MediaPipe pour la reconnaissance de gestes
    Extrait 21 points par main (x, y, z) = 63 features per hand
    Uses MediaPipe Task API (v0.10+)
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ):
        """
        Initialize MediaPipe HandLandmarker

        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum detection confidence
            model_path: Path to hand_landmarker.task model file
        """
        # Find model path
        if model_path is None:
            default_paths = [
                Path(__file__).parent.parent.parent / "models" / "hand_landmarker.task",
                Path(__file__).parent.parent.parent / "models" / "mediapipe" / "hand_landmarker.task",
            ]
            for p in default_paths:
                if p.exists():
                    model_path = str(p)
                    break

        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(
                "hand_landmarker.task not found. Download from: "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        # Create HandLandmarker using Task API
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        # Landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20

    def extract_landmarks(
        self,
        frame: np.ndarray,
        rgb: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from frame

        Args:
            frame: BGR or RGB image (H, W, 3)
            rgb: If True, frame is already RGB

        Returns:
            Landmarks array of shape (num_hands, 21, 3) or None if no hands detected
        """
        if not rgb:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect landmarks
        result = self.hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        all_landmarks = []
        for hand_landmarks in result.hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
            all_landmarks.append(landmarks)

        return np.array(all_landmarks)  # (num_hands, 21, 3)

    def extract_features(
        self,
        frame: np.ndarray,
        normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Extract flattened features for classification

        Args:
            frame: BGR image
            normalize: If True, normalize features

        Returns:
            Feature vector of shape (126,) for 2 hands or (63,) for 1 hand
        """
        landmarks = self.extract_landmarks(frame)

        if landmarks is None:
            return None

        # Flatten landmarks
        features = landmarks.flatten()

        # Pad to fixed size (2 hands * 21 landmarks * 3 coordinates = 126)
        if len(features) < 126:
            features = np.pad(features, (0, 126 - len(features)))

        if normalize:
            features = self._normalize_features(features)

        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features relative to wrist position and hand size
        """
        if len(features) == 126:
            features = features.reshape(2, 21, 3)
        else:
            features = features.reshape(-1, 21, 3)

        for i in range(features.shape[0]):
            hand = features[i]
            hand = hand - hand[0]  # relative to wrist
            hand_size = np.linalg.norm(hand[self.MIDDLE_TIP] - hand[self.WRIST])
            if hand_size > 0:
                hand = hand / hand_size
            features[i] = hand

        return features.flatten()

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draw landmarks on frame

        Args:
            frame: BGR image
            landmarks: Pre-extracted landmarks (optional)

        Returns:
            Frame with drawn landmarks
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.hand_landmarker.detect(mp_image)

        frame_copy = frame.copy()
        h, w, _ = frame_copy.shape

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Draw connections
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw points
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_copy, (x, y), 4, (0, 0, 255), -1)

        return frame_copy

    def get_hand_info(
        self,
        frame: np.ndarray,
    ) -> List[dict]:
        """
        Get hand information including handedness

        Args:
            frame: BGR image

        Returns:
            List of dictionaries with hand info
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.hand_landmarker.detect(mp_image)

        hands_info = []

        if result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness in zip(
                result.hand_landmarks, result.handedness
            ):
                h, w, _ = frame.shape
                xs = [lm.x for lm in hand_landmarks]
                ys = [lm.y for lm in hand_landmarks]

                hands_info.append({
                    'handedness': handedness[0].category_name,
                    'confidence': handedness[0].score,
                    'bbox': (int(min(xs) * w), int(min(ys) * h),
                             int(max(xs) * w), int(max(ys) * h)),
                    'landmarks': np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                })

        return hands_info

    def release(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'hand_landmarker') and self.hand_landmarker:
            self.hand_landmarker.close()

    def __del__(self):
        """Destructor"""
        self.release()
