"""
TSL Translation Pipeline - Complete translation from camera to text.

Supports two modes:
  - SIGN mode: standard 57-sign recognition (existing pipeline)
  - SPELL mode: finger spelling with 38-letter Arabic alphabet

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
import threading
import uuid

from src.models.landmark_extractor import MediaPipeLandmarkExtractor
from src.models.classifier import TSLClassifier, create_model
from src.models.spell_classifier import SpellClassifier
from src.languages.dictionary import TSLDictionary
from src.languages.alphabet import TSLAlphabet
from src.config.settings import (
    NUM_CLASSES, DEFAULT_MODEL_PATH,
    CONFIDENCE_THRESHOLD, TEMPORAL_SMOOTHING_WINDOW, HISTORY_MAX_LEN,
    ALPHABET_NUM_CLASSES, ALPHABET_MODELS_DIR, SPELL_GAP_FRAMES,
    UNSUPPORTED_SIGN_CLASSES,
)
from src.translation.streaming import StreamingRecognizer, SentenceBuilder
from src.translation.spell_assembler import SpellAssembler
from src.translation.mode_controller import ModeController
from src.translation.translator_backend import OfflineTranslator

try:
    from src.translation.llm_corrector import LLMCorrector, LLMConfig, save_config
except ImportError:
    LLMCorrector = None  # type: ignore
    LLMConfig = None  # type: ignore
    save_config = None  # type: ignore

# Configure logging only if the embedding application hasn't (avoids hijacking
# the root logger when this module is imported as a library).
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared read-only dictionary (static data; rebuilt per call was wasteful)
_shared_dictionary: Optional[TSLDictionary] = None


def get_shared_dictionary() -> TSLDictionary:
    global _shared_dictionary
    if _shared_dictionary is None:
        _shared_dictionary = TSLDictionary()
    return _shared_dictionary


class SessionState:
    """Mutable per-client state: smoothing deques, history, mode, spell buffers.

    Models and MediaPipe detectors are shared read-only across sessions; only
    this state is isolated, so concurrent API clients cannot corrupt each
    other's predictions, history, mode or finger-spelling buffers.
    """

    def __init__(
        self,
        smoothing_window: int,
        history_max_len: int,
        spell_gap_frames: int,
    ):
        self.smoothing_window = max(1, int(smoothing_window))
        self.history_max_len = int(history_max_len)
        self.recent_predictions: deque = deque(maxlen=self.smoothing_window)
        self.spell_recent_predictions: deque = deque(maxlen=self.smoothing_window)
        self.history: deque = deque(maxlen=self.history_max_len)
        self.mode_controller = ModeController()
        self.spell_assembler = SpellAssembler(gap_frames=spell_gap_frames)

    def set_smoothing_window(self, window: int) -> None:
        w = max(1, int(window))
        if w == self.smoothing_window:
            return
        self.smoothing_window = w
        self.recent_predictions = deque(self.recent_predictions, maxlen=w)
        self.spell_recent_predictions = deque(self.spell_recent_predictions, maxlen=w)


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

        # Bounded history (lives inside each session)
        self.history_max_len = int(history_max_len)

        # Finger spelling components
        self.alphabet = TSLAlphabet()
        self.spell_model: Optional[SpellClassifier] = None

        # Per-client session state + locks for shared, non-thread-safe resources
        self._sessions: Dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._detect_lock = threading.Lock()

        # Default session (single-user callers: Streamlit, demo, tests)
        self.get_session()

        # Resolve model path
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else None

        # Load model + class_names order from checkpoint
        self.class_names: List[str] = self.dictionary.get_all_labels()
        self.model = self._load_model(model_path)

        # Load spell model (separate classifier for finger spelling)
        self.spell_model = self._load_spell_model()

        # LLM corrector (optional, configurable)
        self.llm_corrector = None
        if LLMCorrector is not None:
            try:
                self.llm_corrector = LLMCorrector()
                if self.llm_corrector.enabled:
                    logger.info(f"LLM corrector enabled: {self.llm_corrector.get_config()['provider']}/{self.llm_corrector.get_config()['model']}")
                else:
                    logger.info("LLM corrector available but disabled (no API key). Set LLM_API_KEY to enable.")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM corrector: {e}")

    # ── Session management ──────────────────────────────────────────────

    def get_session(self, session_id: Optional[str] = None) -> SessionState:
        """Return (creating if needed) the state container for a client."""
        sid = session_id or "default"
        with self._sessions_lock:
            state = self._sessions.get(sid)
            if state is None:
                state = SessionState(
                    self.smoothing_window,
                    self.history_max_len,
                    SPELL_GAP_FRAMES,
                )
                self._sessions[sid] = state
            return state

    def delete_session(self, session_id: Optional[str]) -> None:
        """Drop a client's state (call on WebSocket disconnect)."""
        if not session_id or session_id == "default":
            return
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def _extract_features(self, frame: np.ndarray):
        """Landmark extraction serialized across threads (MediaPipe Tasks
        detectors are not documented as thread-safe)."""
        with self._detect_lock:
            return self.landmark_extractor.extract_features(frame)

    def _draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        with self._detect_lock:
            return self.landmark_extractor.draw_landmarks(frame)

    def set_inference_settings(
        self,
        confidence_threshold: Optional[float] = None,
        smoothing_window: Optional[int] = None,
    ) -> Dict[str, float]:
        """Update global inference settings (validated) across all sessions."""
        if confidence_threshold is not None:
            c = float(confidence_threshold)
            if not 0.0 <= c <= 1.0:
                raise ValueError("confidence_threshold must be between 0 and 1")
            self.confidence_threshold = c
        if smoothing_window is not None:
            w = int(smoothing_window)
            if not 1 <= w <= 60:
                raise ValueError("smoothing_window must be between 1 and 60")
            self.smoothing_window = w
            with self._sessions_lock:
                for state in self._sessions.values():
                    state.set_smoothing_window(w)
        return self.get_inference_settings()

    def get_inference_settings(self) -> Dict[str, float]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "smoothing_window": self.smoothing_window,
        }

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
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
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

    def _load_spell_model(self) -> Optional[SpellClassifier]:
        """Load the spell classifier if a trained checkpoint exists."""
        spell_path = ALPHABET_MODELS_DIR / "spell_classifier_best.pth"
        if not spell_path.exists():
            logger.info(
                f"No spell model found at {spell_path}. "
                "Finger spelling will be unavailable. "
                "Train with: python scripts/train_alphabet_models.py"
            )
            return None

        try:
            checkpoint = torch.load(spell_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Detect architecture from state_dict keys
            if any(k.startswith('net.') for k in state_dict.keys()):
                # Simple MLP architecture from train_alphabet_models.py
                model = SpellClassifier(num_classes=ALPHABET_NUM_CLASSES, architecture="mlp")
            elif any(k.startswith('input_block.') for k in state_dict.keys()):
                # Original SpellClassifier architecture
                model = SpellClassifier(num_classes=ALPHABET_NUM_CLASSES, architecture="residual")
            else:
                # Default: try the SpellClassifier architecture
                model = SpellClassifier(num_classes=ALPHABET_NUM_CLASSES, architecture="mlp")

            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            logger.info(f"Loaded spell model from {spell_path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load spell model: {e}")
            return None

    def translate_frame(
        self,
        frame: np.ndarray,
        return_all: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Translate a single frame, applying confidence thresholding and
        temporal smoothing (the smoothed label is the majority of the last
        `smoothing_window` accepted predictions).

        Args:
            frame: BGR image
            return_all: unused (kept for API compatibility)
            session_id: client session whose smoothing/history is updated

        Returns:
            Dictionary with translations. On rejection (no hands or low
            confidence) returns {"error": ...} or {"tsl": None, ...}.
            Every result carries "mode_label": the raw predicted class label
            (before UNSUPPORTED_SIGN_CLASSES filtering), used for mode control.
        """
        state = self.get_session(session_id)

        # 1. Extract landmarks
        features = self._extract_features(frame)
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
            state.recent_predictions.append(None)
        else:
            state.recent_predictions.append(predicted_class)

        # 4. Temporal smoothing via majority vote over accepted predictions
        accepted = [p for p in state.recent_predictions if p is not None]
        if not accepted:
            translations = {
                "tsl": None,
                "tunisian": None,
                "french": None,
                "english": None,
                "arabic": None,
                "arabic_msa": None,
                "confidence": confidence,
                "rejected": True,
                "mode_label": None,
            }
            self._append_history(translations, state)
            return translations

        # Find the majority label among the accepted window
        counts: Dict[int, int] = {}
        for cls in accepted:
            counts[cls] = counts.get(cls, 0) + 1
        smoothed_class = max(counts, key=counts.get)

        # Raw label (used for mode control even when the class itself has no
        # training data, e.g. the ta3raf mode-switch sign)
        raw_label = (
            self.class_names[smoothed_class]
            if 0 <= smoothed_class < len(self.class_names)
            else None
        )

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
                "arabic_msa": None,
                "confidence": confidence,
                "rejected": True,
                "mode_label": raw_label,
            }
            self._append_history(translations, state)
            return translations

        translations = self.dictionary.get_all_translations(tsl_label)
        translations["confidence"] = confidence
        translations["smoothed_count"] = counts[smoothed_class]
        translations["smoothed_window"] = len(accepted)
        translations["rejected"] = False
        translations["mode_label"] = raw_label

        self._append_history(translations, state)
        return translations

    def spell_frame(
        self,
        frame: np.ndarray,
        return_all: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Recognize a finger spelling letter from a single frame.

        Args:
            frame: BGR image
            session_id: client session whose smoothing is updated

        Returns:
            Dict with letter info: {letter, arabic, confidence, ...}
        """
        if self.spell_model is None:
            return {"error": "Spell model not loaded"}

        state = self.get_session(session_id)

        # 1. Extract landmarks
        features = self._extract_features(frame)
        if features is None:
            return {"error": "No hands detected"}

        # 2. Forward pass through spell model
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.spell_model(features_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

        # 3. Apply confidence threshold
        if confidence < self.confidence_threshold:
            state.spell_recent_predictions.append(None)
        else:
            state.spell_recent_predictions.append(predicted_class)

        # 4. Temporal smoothing
        accepted = [p for p in state.spell_recent_predictions if p is not None]
        if not accepted:
            return {
                "letter": None,
                "arabic": None,
                "confidence": confidence,
                "rejected": True,
            }

        counts: Dict[int, int] = {}
        for cls in accepted:
            counts[cls] = counts.get(cls, 0) + 1
        smoothed_class = max(counts, key=counts.get)

        # 5. Map to letter
        spell_labels = self.alphabet.get_all_labels()
        if smoothed_class < 0 or smoothed_class >= len(spell_labels):
            return {
                "letter": None,
                "arabic": None,
                "confidence": confidence,
                "rejected": True,
            }

        letter_label = spell_labels[smoothed_class]
        arabic_char = self.alphabet.get_arabic(letter_label)

        return {
            "letter": letter_label,
            "arabic": arabic_char,
            "confidence": confidence,
            "smoothed_count": counts[smoothed_class],
            "smoothed_window": len(accepted),
            "rejected": False,
        }

    def translate_frame_multi_mode(
        self,
        frame: np.ndarray,
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Translate a frame using the current mode (sign or spell).
        Handles mode switching via TSL control signs.

        Returns:
            Dict with either sign translations or spell letter info,
            plus mode state.
        """
        state = self.get_session(session_id)

        # First, always try to recognize as a sign (to detect mode switches)
        sign_result = self.translate_frame(frame, session_id=session_id)

        # Check if this sign triggers a mode change. Use the raw predicted
        # label (mode_label) so control signs with no training data (ta3raf)
        # still switch mode even though they translate to nothing.
        control_label = sign_result.get("mode_label") or sign_result.get("tsl")
        if control_label:
            mode_event = state.mode_controller.update(control_label)
            if mode_event["consumed"]:
                # Mode changed - return mode change event
                return {
                    "type": "mode_change",
                    "mode": mode_event["mode"],
                    "action": mode_event["action"],
                    "sign": control_label,
                }

        # If in spell mode, use spell classifier
        if state.mode_controller.is_spell_mode:
            if self.spell_model is None:
                return {
                    "type": "error",
                    "error": "Spell model not loaded",
                    "mode": "spell",
                }

            spell_result = self.spell_frame(frame, session_id=session_id)

            # Feed to assembler
            if spell_result.get("letter") and not spell_result.get("rejected"):
                assembly = state.spell_assembler.feed_letter(spell_result["letter"])
            else:
                assembly = state.spell_assembler.on_empty_frame()

            return {
                "type": "spell",
                "mode": "spell",
                "letter": spell_result.get("letter"),
                "arabic": spell_result.get("arabic"),
                "confidence": spell_result.get("confidence", 0),
                "current_word": assembly.get("current_word", ""),
                "sentence": assembly.get("sentence", ""),
                "event": assembly.get("event"),
            }

        # If in sign mode, return normal sign result
        return {
            "type": "sign",
            "mode": "sign",
            **sign_result,
        }

    def _class_index_to_label(self, idx: int) -> Optional[str]:
        """Convert a class index to a TSL label using the checkpoint order."""
        if idx is None or idx < 0 or idx >= len(self.class_names):
            return None
        label = self.class_names[idx]
        # Reject classes with no training data (MediaPipe fails on their images)
        if label in UNSUPPORTED_SIGN_CLASSES:
            return None
        return label

    def _append_history(
        self, entry: Dict, state: Optional[SessionState] = None
    ) -> None:
        """Append entry to a session's bounded history (default session)."""
        (state or self.get_session()).history.append(entry)

    def translate_video_stream(
        self,
        camera_id: int = 0,
        window_name: str = "TSL Translator",
        show_landmarks: bool = True,
    ):
        """Real-time translation from video stream.

        Uses StreamingRecognizer + SentenceBuilder under the hood to recognize
        and assemble continuous signs into readable sentences.
        Supports mode switching between sign recognition and finger spelling.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
        logger.info("Press 'q' to quit | Press 's' to toggle sign/spell mode")

        recognizer = StreamingRecognizer()
        builder = SentenceBuilder(self.dictionary, target_lang=self.output_lang)
        current_sentence_labels: List[str] = []
        state = self.get_session()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Use multi-mode translation
                result = self.translate_frame_multi_mode(frame, session_id="default")

                # Handle mode change
                if result.get("type") == "mode_change":
                    logger.info(f"MODE: {result['mode'].upper()}")
                    continue

                # Handle spell mode output
                if result.get("type") == "spell":
                    if result.get("letter") and not result.get("rejected"):
                        letter = result.get("arabic") or "?"
                        current_word = result.get("current_word", "")
                        sentence = result.get("sentence", "")
                        logger.info(f"SPELL: {letter} | Word: {current_word} | Sentence: {sentence}")

                    # Draw spell results
                    if show_landmarks:
                        frame = self._draw_landmarks(frame)
                    frame = self._draw_spell_results(frame, result)
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('s'):
                        new_mode = state.mode_controller.toggle()
                        logger.info(f"MODE: {new_mode.upper()}")
                    continue

                # Handle sign mode output
                if result.get("type") == "sign" or result.get("tsl"):
                    event = recognizer.feed(result)
                    if event["sign"] is not None:
                        builder.add_sign(event["sign"])
                        current_sentence_labels.append(event["sign"])
                    if event["sentence_boundary"] and current_sentence_labels:
                        sentence = builder.flush()
                        if sentence:
                            logger.info(f"SENTENCE: {sentence}")
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
                        frame = self._draw_landmarks(frame)
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('s'):
                        new_mode = state.mode_controller.toggle()
                        logger.info(f"MODE: {new_mode.upper()}")
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

    def _draw_spell_results(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw finger spelling results on frame."""
        frame_copy = frame.copy()
        h, w = frame.shape[:2]

        # Header
        cv2.rectangle(frame_copy, (0, 0), (w, 150), (30, 30, 30), -1)
        cv2.putText(frame_copy, "MODE: SPELL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Current letter
        arabic = result.get("arabic") or ""
        letter = result.get("letter") or ""
        confidence = result.get("confidence", 0)
        cv2.putText(frame_copy, f"Letter: {arabic} ({letter})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Current word being built
        current_word = result.get("current_word", "")
        cv2.putText(frame_copy, f"Word: {current_word}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Full sentence
        sentence = result.get("sentence", "")
        if sentence:
            cv2.putText(frame_copy, f"Sentence: {sentence}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Confidence bar
        bar_w = int(confidence * 200)
        cv2.rectangle(frame_copy, (10, 135), (10 + bar_w, 145), (0, 255, 0), -1)
        cv2.rectangle(frame_copy, (10, 135), (210, 145), (100, 100, 100), 1)

        return frame_copy

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
        # Private session: smoothing/history of this file never leaks into
        # (or is polluted by) other concurrent requests.
        vid_sid = f"video:{uuid.uuid4().hex}"

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if sample_every_n_frames > 1 and frame_idx % sample_every_n_frames != 0:
                    frame_idx += 1
                    continue
                result = self.translate_frame(frame, session_id=vid_sid)
                event = recognizer.feed(result)
                if event["sign"] is not None:
                    builder.add_sign(event["sign"])
                if event["sentence_boundary"]:
                    signs = list(builder.signs_seq)
                    sentence = builder.flush()
                    if sentence:
                        translations = {}
                        for lang in extra_langs:
                            translations[lang] = self.offline_translator.translate(
                                sentence, target_lang, lang
                            )
                        sentences.append({
                            "sentence": sentence,
                            "signs": signs,
                            "translations": translations,
                        })
                frame_idx += 1
        finally:
            cap.release()
            self.delete_session(vid_sid)

        # Flush remaining
        signs = list(builder.signs_seq)
        sentence = builder.flush()
        if sentence:
            translations = {}
            for lang in extra_langs:
                translations[lang] = self.offline_translator.translate(
                    sentence, target_lang, lang
                )
            sentences.append({
                "sentence": sentence,
                "signs": signs,
                "translations": translations,
            })
        logger.info(f"Video {video_path}: produced {len(sentences)} sentences from {frame_idx} frames")
        return sentences

    def batch_translate(self, frames: List[np.ndarray]) -> List[Dict]:
        """Translate multiple frames."""
        return [self.translate_frame(f) for f in frames]

    def get_history(self, session_id: Optional[str] = None) -> List[Dict]:
        """Get translation history for a session (bounded, copy)."""
        return list(self.get_session(session_id).history)

    def clear_history(self, session_id: Optional[str] = None) -> None:
        """Clear translation history and spell state for a session."""
        state = self.get_session(session_id)
        state.history.clear()
        state.recent_predictions.clear()
        state.spell_recent_predictions.clear()
        state.spell_assembler.reset()

    def get_mode_state(self, session_id: Optional[str] = None) -> Dict[str, str]:
        """Get current mode controller state for a session."""
        return self.get_session(session_id).mode_controller.get_state()

    def toggle_mode(self, session_id: Optional[str] = None) -> str:
        """Toggle between sign and spell mode. Returns new mode."""
        return self.get_session(session_id).mode_controller.toggle()

    def flush_spell(self, session_id: Optional[str] = None) -> str:
        """Flush the spell assembler and return the assembled sentence."""
        return self.get_session(session_id).spell_assembler.flush()

    async def correct_with_llm(self, raw_text: str) -> str:
        """Correct raw text using the LLM corrector (async)."""
        if self.llm_corrector and self.llm_corrector.enabled:
            return await self.llm_corrector.correct(raw_text)
        return raw_text

    def correct_with_llm_sync(self, raw_text: str) -> str:
        """Synchronous wrapper for LLM correction.

        Safe to call from any thread. If the calling thread already runs an
        event loop, the correction cannot be awaited there — the raw text is
        returned unchanged instead of raising RuntimeError.
        """
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.correct_with_llm(raw_text))
        logger.warning(
            "correct_with_llm_sync called from a running event loop; "
            "returning uncorrected text"
        )
        return raw_text

    async def coordinate_sentences(
        self,
        sentences: List[Dict],
        target_lang: str = "en",
    ) -> List[Dict]:
        """LLM-coordinate each finalized video sentence (async, parallel).

        Adds a "coordinated" key per sentence (natural sentence in
        ``target_lang``) and refreshes that sentence's translations so every
        language reflects the coordinated wording. When the LLM is disabled
        or fails, sentences are returned untouched (frontend falls back to
        the raw "sentence").
        """
        import asyncio
        if not sentences:
            return sentences
        if not (self.llm_corrector and self.llm_corrector.enabled):
            return sentences

        async def _one(s: Dict) -> None:
            raw = s.get("sentence") or " ".join(s.get("signs") or [])
            if not raw:
                return
            s["coordinated"] = await self.llm_corrector.coordinate(
                raw, target_lang=target_lang
            )

        await asyncio.gather(
            *(_one(s) for s in sentences), return_exceptions=True
        )
        try:
            await asyncio.to_thread(
                self._refresh_sentence_translations, sentences, target_lang
            )
        except Exception as e:
            logger.warning(f"Translation refresh after coordination failed: {e}")
        return sentences

    def _refresh_sentence_translations(
        self, sentences: List[Dict], target_lang: str
    ) -> None:
        """Re-translate each coordinated sentence into the extra languages."""
        for s in sentences:
            coordinated = s.get("coordinated")
            if not coordinated or coordinated == s.get("sentence"):
                continue
            translations = s.get("translations")
            if not translations:
                continue
            for lang in list(translations.keys()):
                try:
                    translations[lang] = self.offline_translator.translate(
                        coordinated, target_lang, lang
                    )
                except Exception:
                    pass  # keep the original translation for this language

    def configure_llm(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        """Reconfigure the LLM corrector at runtime and persist the choice.

        Every argument is optional: ``None`` means "keep the current value",
        so a partial update (e.g. only the model) can never reset the
        provider, wipe the API key or drop temperature/max_tokens.

        Switching provider resets the provider-specific fields (model,
        base_url) unless they are supplied explicitly, so a stale
        ``openrouter/free`` can never be sent to a local model.
        """
        if LLMCorrector is None or LLMConfig is None:
            logger.warning("LLM corrector not available")
            return
        from src.translation.llm_corrector import LLMProvider

        # Current effective config (falls back to the environment).
        if self.llm_corrector is not None:
            base = self.llm_corrector.config
        else:
            try:
                base = LLMConfig.from_env()
            except Exception:
                base = LLMConfig()

        if provider is None or not str(provider).strip():
            provider_enum = base.provider
        else:
            try:
                provider_enum = LLMProvider(str(provider).strip().lower())
            except ValueError as e:
                raise ValueError(
                    f"Unknown LLM provider '{provider}'. Valid options: "
                    + ", ".join(p.value for p in LLMProvider)
                ) from e

        provider_changed = provider_enum != base.provider
        if model is None or (not str(model).strip() and provider_changed):
            # No model supplied, or an empty model after a provider switch:
            # let apply_defaults() pick the new provider's own default.
            model = "" if provider_changed else base.model
        else:
            model = str(model).strip()
            if provider_changed and provider_enum != LLMProvider.CUSTOM:
                # A model that is another provider's built-in default is a
                # leftover from the previous selection, not a user choice.
                stale_defaults = {
                    str(v.get("model") or "")
                    for k, v in LLMConfig.PROVIDER_DEFAULTS.items()
                    if k != provider_enum and v.get("model")
                }
                if model in stale_defaults:
                    model = ""

        if base_url is None or not str(base_url).strip():
            # Empty/absent: keep the current endpoint, except right after a
            # provider switch where the new provider's own default applies.
            base_url = "" if provider_changed else base.base_url
        else:
            base_url = str(base_url).strip()

        # API keys are provider-specific: when the provider changes and no new
        # key is supplied, drop the old one instead of sending it elsewhere.
        if api_key is not None:
            new_key = str(api_key)
        elif provider_changed:
            new_key = ""
        else:
            new_key = base.api_key

        config = LLMConfig(
            provider=provider_enum,
            api_key=new_key,
            model=model,
            base_url=base_url,
            temperature=base.temperature if temperature is None else float(temperature),
            max_tokens=base.max_tokens if max_tokens is None else int(max_tokens),
            timeout=base.timeout if timeout is None else int(timeout),
        )
        config.apply_defaults()
        config.validate()  # raises ValueError on unsafe/invalid base_url
        if self.llm_corrector is None:
            self.llm_corrector = LLMCorrector(config)
        else:
            self.llm_corrector.configure(config)
        if save_config is not None:
            save_config(config)  # survives an API restart
        logger.info(f"LLM reconfigured: {config.provider.value}/{config.model}")


    def get_llm_config(self) -> dict:
        """Get current LLM configuration."""
        if self.llm_corrector:
            return self.llm_corrector.get_config()
        return {"provider": "none", "model": "none", "has_api_key": False}

    async def close(self):
        """Cleanup resources."""
        if self.llm_corrector:
            await self.llm_corrector.close()

    def __del__(self):
        """Destructor"""
        if hasattr(self, 'landmark_extractor'):
            try:
                self.landmark_extractor.release()
            except Exception:
                pass


def translate_text_to_tsl(text: str) -> Optional[str]:
    """Translate Tunisian-dialect text to a TSL label (case-insensitive)."""
    dictionary = get_shared_dictionary()
    return dictionary.tunisian_to_tsl.get(text.lower())


def translate_tsl_to_text(tsl_label: str) -> Dict[str, str]:
    """Translate a TSL label to text in all languages."""
    dictionary = get_shared_dictionary()
    return dictionary.get_all_translations(tsl_label)
