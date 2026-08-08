"""
TSL Translation Pipeline - Complete translation from camera to text.

Key correctness fixes vs the original implementation:
  - Loads `class_names` order from the checkpoint (single source of truth), so
    inference class indices match training, even if the dictionary changes.
  - Confidence threshold: predictions with confidence < CONFIDENCE_THRESHOLD
    return "unknown" instead of low-confidence garbage.
  - Temporal smoothing: a sliding majority vote over the last
    TEMPORAL_SMOOTHING_WINDOW frames produces a stable prediction.
  - Bounded translation history (HISTORY_MAX_LEN).
"""

import torch
import numpy as np
import cv2
from collections import deque
from typing import Optional, Dict, List
from pathlib import Path
import logging

from src.models.landmark_extractor import MediaPipeLandmarkExtractor
from src.models.classifier import TSLClassifier, create_model
from src.languages.dictionary import TSLDictionary
from src.config.settings import (
    MODELS_DIR, NUM_CLASSES, DEFAULT_MODEL_PATH,
    CONFIDENCE_THRESHOLD, TEMPORAL_SMOOTHING_WINDOW, HISTORY_MAX_LEN,
)
from src.translation.streaming import StreamingRecognizer, SentenceBuilder
from src.translation.translator_backend import OfflineTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSLTranslator:
    """
    Pipeline complet de traduction TSL:
    Caméra → Landmarks → Classification → Dialecte Tunisien → FR/EN/AR

    Includes confidence threshold and temporal smoothing for stable output.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        smoothing_window: int = TEMPORAL_SMOOTHING_WINDOW,
        history_max_len: int = HISTORY_MAX_LEN,
        output_lang: str = "en",
        translate_to_languages: Optional[List[str]] = None,
    ):
        """
        Initialize TSL Translator.

        Args:
            model_path: Path to trained model checkpoint. If None, uses
                DEFAULT_MODEL_PATH if it exists; otherwise random weights.
            device: Inference device (default: cuda if available else cpu)
            confidence_threshold: minimum softmax confidence to accept a
                prediction (predictions below this return "unknown")
            smoothing_window: number of recent predictions used for the
                majority vote (1 = no smoothing)
            history_max_len: max number of translations kept in history
            output_lang: source language used when assembling sentences (used
                by translate_video_stream). One of "en","fr","ar","tunisian".
            translate_to_languages: extra languages to translate assembled
                sentences to via the offline translator (e.g. ["en","fr"]).
                Defaults to ["en","fr"] when None.
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")

        # Streaming sentence output language + extra translation targets
        self.output_lang = output_lang or "en"
        self.translate_to_languages = translate_to_languages or ["en", "fr"]
        self.offline_translator = OfflineTranslator()

        # Components
        self.landmark_extractor = MediaPipeLandmarkExtractor(
            max_num_hands=2,
            min_detection_confidence=0.5,
        )
        self.dictionary = TSLDictionary()

        # Inference params
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = max(1, int(smoothing_window))
        self._recent_predictions: deque = deque(maxlen=self.smoothing_window)

        # Bounded history
        self.history_max_len = int(history_max_len)
        self.translation_history: List[Dict] = []

        # Resolve model path
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else None

        # Load model + class_names order from checkpoint
        self.class_names: List[str] = self.dictionary.get_all_labels()
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Optional[Path]) -> TSLClassifier:
        """
        Load trained model. The checkpoint's `class_names` (if present) becomes
        the authoritative index->label mapping for this instance.

        Args:
            model_path: Path to model checkpoint

        Returns:
            Loaded model in eval mode
        """
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model = create_model("landmark", NUM_CLASSES)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            logger.info(f"Loaded model from {model_path}")

            # Persist class-order mapping from the checkpoint (single source of
            # truth). Fall back to the dictionary if the checkpoint is old.
            ckpt_class_names = checkpoint.get('class_names')
            if ckpt_class_names:
                self.class_names = list(ckpt_class_names)
                logger.info(f"Using checkpoint class_names ({len(self.class_names)} classes)")
            else:
                logger.warning(
                    "Checkpoint has no 'class_names' field — falling back to "
                    "dictionary order. Consider re-training with the updated "
                    "train.py to persist the mapping."
                )
        else:
            logger.warning(
                f"No trained model found (path={model_path}). Using RANDOM "
                "weights — predictions will be meaningless. Train a model with "
                "`python scripts/prepare_data.py && python -m src.models.train`."
            )
            model = create_model("landmark", NUM_CLASSES)
            model = model.to(self.device)
            model.eval()
        return model

    def translate_frame(
        self,
        frame: np.ndarray,
        return_all: bool = True,
    ) -> Dict[str, object]:
        """
        Translate a single frame, applying confidence thresholding and
        temporal smoothing (the smoothed label is the majority of the last
        `smoothing_window` accepted predictions).

        Args:
            frame: BGR image
            return_all: unused (kept for API compatibility)

        Returns:
            Dictionary with translations. On rejection (no hands or low
            confidence) returns {"error": ...} or {"tsl": None, ...}.
        """
        # 1. Extract landmarks
        features = self.landmark_extractor.extract_features(frame)
        if features is None:
            return {"error": "No hands detected"}

        # 2. Forward pass
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

        # 3. Apply confidence threshold BEFORE recording for smoothing
        if confidence < self.confidence_threshold:
            self._recent_predictions.append(None)
        else:
            self._recent_predictions.append(predicted_class)

        # 4. Temporal smoothing via majority vote over accepted predictions
        accepted = [p for p in self._recent_predictions if p is not None]
        if not accepted:
            translations = {
                "tsl": None,
                "tunisian": None,
                "french": None,
                "english": None,
                "arabic": None,
                "confidence": confidence,
                "rejected": True,
            }
            self._append_history(translations)
            return translations

        # Find the majority label among the accepted window
        counts: Dict[int, int] = {}
        for cls in accepted:
            counts[cls] = counts.get(cls, 0) + 1
        smoothed_class = max(counts, key=counts.get)

        # Confidence reported is the current-frame confidence (informational),
        # but the *label* is the smoothed majority — keeps things stable.
        tsl_label = self._class_index_to_label(smoothed_class)
        if tsl_label is None:
            translations = {
                "tsl": None,
                "tunisian": None,
                "french": None,
                "english": None,
                "arabic": None,
                "confidence": confidence,
                "rejected": True,
            }
            self._append_history(translations)
            return translations

        translations = self.dictionary.get_all_translations(tsl_label)
        translations["confidence"] = confidence
        translations["smoothed_count"] = counts[smoothed_class]
        translations["smoothed_window"] = len(accepted)
        translations["rejected"] = False

        self._append_history(translations)
        return translations

    def _class_index_to_label(self, idx: int) -> Optional[str]:
        """Convert a class index to a TSL label using the checkpoint order."""
        if idx is None or idx < 0 or idx >= len(self.class_names):
            return None
        return self.class_names[idx]

    def _append_history(self, entry: Dict) -> None:
        """Append entry to bounded history (drops oldest when full)."""
        self.translation_history.append(entry)
        if len(self.translation_history) > self.history_max_len:
            self.translation_history = self.translation_history[-self.history_max_len:]

    def translate_video_stream(
        self,
        camera_id: int = 0,
        window_name: str = "TSL Translator",
        show_landmarks: bool = True,
    ):
        """Real-time translation from video stream.

        Uses StreamingRecognizer + SentenceBuilder under the hood to recognize
        and assemble continuous signs into readable sentences.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
        logger.info("Press 'q' to quit")

        recognizer = StreamingRecognizer()
        builder = SentenceBuilder(self.dictionary, target_lang=self.output_lang)
        current_sentence_labels: List[str] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.translate_frame(frame)
                event = recognizer.feed(result)
                if event["sign"] is not None:
                    builder.add_sign(event["sign"])
                    current_sentence_labels.append(event["sign"])
                if event["sentence_boundary"] and current_sentence_labels:
                    sentence = builder.flush()
                    if sentence:
                        logger.info(f"SENTENCE: {sentence}")
                        # Multi-language translations of the assembled sentence
                        for lang in self.translate_to_languages:
                            try:
                                tr = self.offline_translator.translate(
                                    sentence, self.output_lang, lang
                                )
                                logger.info(f"  -> [{lang}] {tr}")
                            except Exception as e:
                                logger.warning(f"Translation {lang} failed: {e}")
                    current_sentence_labels = []

                # draw per-frame result
                if "error" not in result and not result.get("rejected"):
                    frame = self._draw_results(frame, result)
                if show_landmarks:
                    frame = self.landmark_extractor.draw_landmarks(frame)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        # Flush any remaining buffered signs at the end
        if current_sentence_labels:
            sentence = builder.flush()
            if sentence:
                logger.info(f"FINAL SENTENCE: {sentence}")

    def _draw_results(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw translation results on frame."""
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(frame_copy, (0, 0), (w, 130), (0, 0, 0), -1)
        tsl_label = result.get('tsl') or 'N/A'
        tunisian = result.get('tunisian') or 'N/A'
        french = result.get('french') or 'N/A'
        confidence = result.get('confidence', 0)
        cv2.putText(frame_copy, f"TSL: {tsl_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"Tunisian: {tunisian}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_copy, f"French: {french}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame_copy, f"Confidence: {confidence:.2%}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame_copy

    def translate_image(self, image_path: str) -> Dict:
        """Translate a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            return {"error": f"Cannot read image: {image_path}"}
        return self.translate_frame(frame)

    def translate_video_file(
        self,
        video_path: str,
        target_lang: str = "en",
        extra_langs: Optional[List[str]] = None,
        sample_every_n_frames: int = 1,
    ) -> List[Dict]:
        """Process a video file, returning finalized sentences with translations.

        Args:
            video_path: path to a video file
            target_lang: source language for sentence assembly
                (the assembled sentence lives in this language before being
                translated to extra_langs)
            extra_langs: extra languages to translate each sentence to via
                OfflineTranslator (defaults to ["en","fr"] when None)
            sample_every_n_frames: process 1 frame every N (for speed on long
                videos)

        Returns:
            List of dicts: {"sentence": str, "signs": [...], "translations": {lang: str}}
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video {video_path}")
            return []
        recognizer = StreamingRecognizer()
        builder = SentenceBuilder(self.dictionary, target_lang=target_lang)
        extra_langs = extra_langs or ["en", "fr"]
        sentences: List[Dict] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if sample_every_n_frames > 1 and frame_idx % sample_every_n_frames != 0:
                    frame_idx += 1
                    continue
                result = self.translate_frame(frame)
                event = recognizer.feed(result)
                if event["sign"] is not None:
                    builder.add_sign(event["sign"])
                if event["sentence_boundary"]:
                    sentence = builder.flush()
                    if sentence:
                        translations = {}
                        for lang in extra_langs:
                            translations[lang] = self.offline_translator.translate(
                                sentence, target_lang, lang
                            )
                        sentences.append({
                            "sentence": sentence,
                            "signs": list(builder.signs_seq),
                            "translations": translations,
                        })
                frame_idx += 1
        finally:
            cap.release()

        # Flush remaining
        sentence = builder.flush()
        if sentence:
            translations = {}
            for lang in extra_langs:
                translations[lang] = self.offline_translator.translate(
                    sentence, target_lang, lang
                )
            sentences.append({
                "sentence": sentence,
                "signs": [],
                "translations": translations,
            })
        logger.info(f"Video {video_path}: produced {len(sentences)} sentences from {frame_idx} frames")
        return sentences

    def batch_translate(self, frames: List[np.ndarray]) -> List[Dict]:
        """Translate multiple frames."""
        return [self.translate_frame(f) for f in frames]

    def get_history(self) -> List[Dict]:
        """Get translation history (bounded)."""
        return self.translation_history

    def clear_history(self):
        """Clear translation history."""
        self.translation_history = []
        self._recent_predictions.clear()

    def __del__(self):
        """Destructor"""
        if hasattr(self, 'landmark_extractor'):
            try:
                self.landmark_extractor.release()
            except Exception:
                pass


def translate_text_to_tsl(text: str) -> Optional[str]:
    """Translate Tunisian-dialect text to a TSL label (case-insensitive)."""
    dictionary = TSLDictionary()
    return dictionary.tunisian_to_tsl.get(text.lower())


def translate_tsl_to_text(tsl_label: str) -> Dict[str, str]:
    """Translate a TSL label to text in all languages."""
    dictionary = TSLDictionary()
    return dictionary.get_all_translations(tsl_label)
