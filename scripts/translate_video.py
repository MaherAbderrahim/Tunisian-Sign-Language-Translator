"""
Translate Video - Main script to translate TSL video to text.

Takes a video file as input, recognizes signs and finger spelling,
and outputs the translated sentence.

Usage:
    python scripts/translate_video.py video.mp4
    python scripts/translate_video.py video.mp4 --output result.txt
    python scripts/translate_video.py video.mp4 --lang fr
    python scripts/translate_video.py video.mp4 --verbose
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from src.config.settings import (
    DEFAULT_MODEL_PATH, ALPHABET_MODELS_DIR, MODELS_DIR,
    CONFIDENCE_THRESHOLD, TEMPORAL_SMOOTHING_WINDOW,
)
from src.models.classifier import create_model
from src.models.spell_classifier import SpellClassifier
from src.models.landmark_extractor import MediaPipeLandmarkExtractor
from src.languages.dictionary import TSLDictionary
from src.languages.alphabet import TSLAlphabet
from src.translation.streaming import StreamingRecognizer, SentenceBuilder
from src.translation.spell_assembler import SpellAssembler
from src.translation.mode_controller import ModeController
from src.translation.translator_backend import OfflineTranslator
from src.config.settings import (
    NUM_CLASSES, ALPHABET_NUM_CLASSES, SPELL_GAP_FRAMES,
    STREAMING_STABLE_FRAMES, STREAMING_GAP_FRAMES,
)


class TSLVideoTranslator:
    """
    Traducteur vidéo TSL complet.
    Reconnaît les signes et le doigté, puis assemble les phrases.
    """

    def __init__(self, lang: str = "en", verbose: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.lang = lang

        # Load components
        self.dictionary = TSLDictionary()
        self.alphabet = TSLAlphabet()
        self.extractor = MediaPipeLandmarkExtractor(max_num_hands=2)
        self.offline_translator = OfflineTranslator()

        # Load sign model (57 classes)
        self.sign_model = self._load_sign_model()
        self.sign_class_names = self.dictionary.get_all_labels()

        # Load spell model (38 classes, optional)
        self.spell_model = self._load_spell_model()

        # Streaming components
        self.recognizer = StreamingRecognizer(
            stable_frames=STREAMING_STABLE_FRAMES,
            gap_frames=STREAMING_GAP_FRAMES,
        )
        self.builder = SentenceBuilder(self.dictionary, target_lang=lang)
        self.mode_controller = ModeController()
        self.spell_assembler = SpellAssembler(gap_frames=SPELL_GAP_FRAMES)

        # Temporal smoothing for spell
        self._spell_predictions = []

        # Results
        self.sentences = []
        self.all_signs = []

    def _load_sign_model(self):
        """Load the trained sign classifier."""
        if not DEFAULT_MODEL_PATH.exists():
            print(f"WARNING: Sign model not found at {DEFAULT_MODEL_PATH}")
            print("Run: python scripts/setup_all.py")
            model = create_model("landmark", NUM_CLASSES)
            model.eval()
            return model

        checkpoint = torch.load(DEFAULT_MODEL_PATH, map_location=self.device, weights_only=False)
        model = create_model("landmark", NUM_CLASSES)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(self.device)

        # Use checkpoint class names if available
        if "class_names" in checkpoint:
            self.sign_class_names = list(checkpoint["class_names"])

        if self.verbose:
            print(f"Loaded sign model: {DEFAULT_MODEL_PATH}")
        return model

    def _load_spell_model(self):
        """Load the spell classifier (optional)."""
        spell_path = ALPHABET_MODELS_DIR / "spell_classifier_best.pth"
        if not spell_path.exists():
            if self.verbose:
                print(f"Spell model not found at {spell_path} (finger spelling disabled)")
            return None

        try:
            checkpoint = torch.load(spell_path, map_location=self.device, weights_only=False)
            model = SpellClassifier(num_classes=ALPHABET_NUM_CLASSES)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.to(self.device)
            if self.verbose:
                print(f"Loaded spell model: {spell_path}")
            return model
        except Exception as e:
            print(f"WARNING: Could not load spell model: {e}")
            return None

    def translate_video(self, video_path: str) -> dict:
        """
        Process a video file and return translated sentences.

        Returns:
            {
                "video": str,
                "sentences": [{"sentence": str, "signs": list, "translations": dict}],
                "all_recognized": list,  # every recognized sign/letter
                "total_frames": int,
                "processing_time": float,
            }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if self.verbose:
            print(f"\nVideo: {video_path}")
            print(f"  FPS: {fps:.1f}, Frames: {total_frames}, Duration: {duration:.1f}s")

        t0 = time.time()
        frame_idx = 0
        current_sentence_signs = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract landmarks
                features = self.extractor.extract_features(frame)
                if features is None:
                    # No hands detected
                    event = self.recognizer.feed({"rejected": True})
                    if event["sign"] is not None:
                        self.builder.add_sign(event["sign"])
                        current_sentence_signs.append(event["sign"])
                    if event["sentence_boundary"] and current_sentence_signs:
                        sentence = self.builder.flush()
                        if sentence:
                            self._save_sentence(sentence, current_sentence_signs)
                        current_sentence_signs = []
                    frame_idx += 1
                    continue

                # Sign classification
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.sign_model(features_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    confidence, pred_idx = torch.max(probs, dim=-1)
                    confidence = confidence.item()
                    pred_idx = pred_idx.item()

                # Check mode switching
                if confidence >= CONFIDENCE_THRESHOLD:
                    tsl_label = self.sign_class_names[pred_idx]
                    mode_event = self.mode_controller.update(tsl_label)

                    if mode_event["consumed"]:
                        # Mode switched - don't process as a regular sign
                        if self.verbose:
                            print(f"  [{frame_idx}] MODE -> {mode_event['mode'].upper()}")
                        frame_idx += 1
                        continue

                # Process based on current mode
                if self.mode_controller.is_spell_mode and self.spell_model is not None:
                    # Spell mode - SpellClassifier returns just logits
                    with torch.no_grad():
                        spell_logits = self.spell_model(features_tensor)
                        spell_probs = torch.softmax(spell_logits, dim=-1)
                        spell_conf, spell_idx = torch.max(spell_probs, dim=-1)
                        spell_conf = spell_conf.item()
                        spell_idx = spell_idx.item()

                    if spell_conf >= CONFIDENCE_THRESHOLD:
                        letter_label = self.alphabet.get_all_labels()[spell_idx]
                        arabic_char = self.alphabet.get_arabic(letter_label)
                        self.spell_assembler.feed_letter(letter_label)
                        if self.verbose:
                            print(f"  [{frame_idx}] SPELL: {arabic_char} ({letter_label}) conf={spell_conf:.2f}")
                    else:
                        # Empty frame in spell mode
                        self.spell_assembler.on_empty_frame()
                else:
                    # Sign mode
                    if confidence >= CONFIDENCE_THRESHOLD:
                        tsl_label = self.sign_class_names[pred_idx]
                        result = {
                            "tsl": tsl_label,
                            "confidence": confidence,
                            "rejected": False,
                        }
                    else:
                        result = {"rejected": True}

                    event = self.recognizer.feed(result)
                    if event["sign"] is not None:
                        self.builder.add_sign(event["sign"])
                        current_sentence_signs.append(event["sign"])
                        if self.verbose:
                            print(f"  [{frame_idx}] SIGN: {event['sign']} conf={confidence:.2f}")

                    if event["sentence_boundary"] and current_sentence_signs:
                        sentence = self.builder.flush()
                        if sentence:
                            self._save_sentence(sentence, current_sentence_signs)
                        current_sentence_signs = []

                frame_idx += 1

        finally:
            cap.release()

        # Flush remaining signs
        if current_sentence_signs:
            sentence = self.builder.flush()
            if sentence:
                self._save_sentence(sentence, current_sentence_signs)

        # Flush spell assembler
        spell_sentence = self.spell_assembler.flush()
        if spell_sentence:
            self._save_sentence(spell_sentence, ["SPELL:" + spell_sentence])

        processing_time = time.time() - t0

        return {
            "video": video_path,
            "sentences": self.sentences,
            "all_recognized": self.all_signs,
            "total_frames": frame_idx,
            "processing_time": processing_time,
        }

    def _save_sentence(self, sentence: str, signs: list):
        """Save a recognized sentence with translations."""
        translations = {}
        for target_lang in ["en", "fr", "ar"]:
            if target_lang != self.lang:
                try:
                    tr = self.offline_translator.translate(sentence, self.lang, target_lang)
                    translations[target_lang] = tr
                except Exception:
                    translations[target_lang] = ""

        self.sentences.append({
            "sentence": sentence,
            "signs": signs,
            "translations": translations,
        })
        self.all_signs.extend(signs)

    def release(self):
        """Cleanup."""
        if hasattr(self, 'extractor'):
            self.extractor.release()


def main():
    parser = argparse.ArgumentParser(
        description="Translate TSL video to text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/translate_video.py video.mp4
  python scripts/translate_video.py video.mp4 --output result.txt
  python scripts/translate_video.py video.mp4 --lang fr
  python scripts/translate_video.py video.mp4 --verbose
        """,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", help="Save result to text file")
    parser.add_argument("--lang", default="en", choices=["en", "fr", "ar", "tunisian"],
                        help="Output language (default: en)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    # Run translation
    translator = TSLVideoTranslator(lang=args.lang, verbose=args.verbose)
    try:
        result = translator.translate_video(str(video_path))
    finally:
        translator.release()

    # Check for errors
    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    # Print results
    print()
    print("=" * 60)
    print("  TRANSLATION RESULT")
    print("=" * 60)
    print(f"  Video: {result['video']}")
    print(f"  Frames processed: {result['total_frames']}")
    print(f"  Processing time: {result['processing_time']:.1f}s")
    print(f"  Sentences found: {len(result['sentences'])}")

    if result["sentences"]:
        print()
        for i, sent in enumerate(result["sentences"], 1):
            print(f"  Sentence {i}: {sent['sentence']}")
            for lang, tr in sent.get("translations", {}).items():
                if tr:
                    print(f"    [{lang}] {tr}")
            if sent.get("signs"):
                print(f"    Signs: {', '.join(sent['signs'])}")
    else:
        print()
        print("  No sentences recognized.")
        if result["all_recognized"]:
            print(f"  Recognized elements: {', '.join(result['all_recognized'])}")

    print()
    print("=" * 60)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"TSL Translation Result\n")
            f.write(f"Video: {result['video']}\n")
            f.write(f"Frames: {result['total_frames']}\n")
            f.write(f"Time: {result['processing_time']:.1f}s\n")
            f.write(f"\n")
            for i, sent in enumerate(result["sentences"], 1):
                f.write(f"Sentence {i}: {sent['sentence']}\n")
                for lang, tr in sent.get("translations", {}).items():
                    if tr:
                        f.write(f"  [{lang}] {tr}\n")
                f.write(f"  Signs: {', '.join(sent.get('signs', []))}\n")
                f.write(f"\n")
        print(f"Result saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
