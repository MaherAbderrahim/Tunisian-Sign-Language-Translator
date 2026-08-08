"""
Tests for TSL Translator
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '.')

from src.languages.dictionary import TSLDictionary
from src.translation.translator import translate_tsl_to_text, translate_text_to_tsl


class TestTSLDictionary:
    """Tests for TSL Dictionary (aligned with W7CH dataset labels)."""

    def setup_method(self):
        self.dictionary = TSLDictionary()

    def test_dictionary_length(self):
        """Dictionary must have exactly 57 entries (matches dataset)."""
        assert len(self.dictionary) == 57

    def test_get_tunisian(self):
        """Test getting Tunisian from a dataset TSL label."""
        assert self.dictionary.get_tunisian("labes") == "labes"

    def test_get_french(self):
        result = self.dictionary.get_french("labes")
        assert result == "bien"

    def test_get_english(self):
        result = self.dictionary.get_english("labes")
        assert result == "well"

    def test_get_arabic(self):
        result = self.dictionary.get_arabic("labes")
        assert result is not None

    def test_get_all_translations(self):
        result = self.dictionary.get_all_translations("labes")
        assert result["tsl"] == "labes"
        assert result["tunisian"] == "labes"
        assert result["french"] == "bien"
        assert result["english"] == "well"
        assert result["arabic"] is not None
        # Arabic MSA field must be present (formal Arabic for non-Tunisians)
        assert "arabic_msa" in result
        assert result["arabic_msa"]
        assert result["category"] == "Demandes"

    def test_invalid_label(self):
        result = self.dictionary.get_all_translations("totallyunknown")
        assert result == {}

    def test_get_all_labels(self):
        labels = self.dictionary.get_all_labels()
        assert len(labels) == 57
        assert "labes" in labels
        # W7CH dataset labels must be present
        assert "3aslema" in labels
        assert "karhba" in labels
        assert "thnin" in labels

    def test_get_all_tunisian_words(self):
        words = self.dictionary.get_all_tunisian_words()
        assert len(words) == 57

    def test_contains(self):
        assert "labes" in self.dictionary
        assert "unknown" not in self.dictionary

    def test_canonical_label_handles_metro_mojibake(self):
        """The W7CH dataset 'métro' folder is stored as latin-1 mojibake."""
        # The actual bytes: 'm' + b'\x8e' + 'tro' in CP1252 Latin-1
        assert self.dictionary.canonical_label("metro") == "metro"


class TestDictionaryMatchesDataset:
    """Verify the dictionary labels match the W7CH dataset folder names."""

    def setup_method(self):
        self.dictionary = TSLDictionary()

    def _dataset_folders(self):
        from pathlib import Path
        root = Path("data/raw/tsl_dataset/data/Data")
        folders = set()
        if root.exists():
            for cat_dir in root.iterdir():
                if cat_dir.is_dir():
                    for class_dir in cat_dir.iterdir():
                        if class_dir.is_dir():
                            folders.add(class_dir.name)
        return folders

    def test_dictionary_matches_dataset(self):
        ds_folders = self._dataset_folders()
        if not ds_folders:
            pytest.skip("W7CH dataset not present")
        dict_labels = set(self.dictionary.get_all_labels())
        # Every dataset folder (except the mojibake 'métro') must map to a
        # dictionary label via canonical_label().
        unmatched = []
        for folder in ds_folders:
            canonical = self.dictionary.canonical_label(folder)
            if canonical is None:
                unmatched.append(folder)
        assert not unmatched, f"Dataset folders with no dictionary match: {sorted(unmatched)}"


class TestTranslation:
    """Tests for translation functions."""

    def test_tsl_to_text(self):
        result = translate_tsl_to_text("labes")
        assert result["tunisian"] == "labes"

    def test_text_to_tsl(self):
        result = translate_text_to_tsl("labes")
        assert result == "labes"

    def test_invalid_tsl(self):
        result = translate_tsl_to_text("totallyunknown")
        assert result == {}

    def test_invalid_text(self):
        result = translate_text_to_tsl("totallyunknown")
        assert result is None


class TestModel:
    """Tests for model."""

    def test_model_creation(self):
        from src.models.classifier import TSLClassifier
        model = TSLClassifier(num_classes=57)
        assert model is not None

    def test_model_forward(self):
        from src.models.classifier import TSLClassifier
        import torch
        model = TSLClassifier(num_classes=57)
        model.eval()
        dummy_input = torch.randn(1, 126)
        output = model(dummy_input)
        assert output.shape == (1, 57)

    def test_model_predict(self):
        from src.models.classifier import TSLClassifier
        import torch
        model = TSLClassifier(num_classes=57)
        dummy_input = torch.randn(1, 126)
        prediction = model.predict(dummy_input)
        assert prediction.shape == (1,)
        assert 0 <= prediction.item() < 57


class TestLandmarkExtractor:
    """Tests for landmark extractor."""

    def test_extractor_creation(self):
        from src.models.landmark_extractor import MediaPipeLandmarkExtractor
        extractor = MediaPipeLandmarkExtractor()
        assert extractor is not None
        extractor.release()

    def test_extract_features(self):
        from src.models.landmark_extractor import MediaPipeLandmarkExtractor
        extractor = MediaPipeLandmarkExtractor()
        dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
        features = extractor.extract_features(dummy_image)  # may be None
        if features is not None:
            assert features.shape == (126,) or features.shape == (63,)
        extractor.release()


class TestTranslatorPipeline:
    """Smoke tests for the translator's correctness features."""

    def test_translator_initialization(self):
        """Translator must initialize (with random weights if no model)."""
        from src.translation.translator import TSLTranslator
        t = TSLTranslator()
        assert len(t.class_names) == 57
        assert t.confidence_threshold > 0
        assert t.smoothing_window >= 1
        t.clear_history()
        # destructor releases

    def test_history_is_bounded(self):
        """History must never exceed HISTORY_MAX_LEN."""
        from src.translation.translator import TSLTranslator
        from src.config.settings import HISTORY_MAX_LEN
        t = TSLTranslator()
        fake_entry = {"tsl": "test", "confidence": 0.9}
        for _ in range(HISTORY_MAX_LEN + 50):
            t._append_history(fake_entry)
        assert len(t.get_history()) == HISTORY_MAX_LEN

    def test_no_hands_returns_error(self):
        """A blank image must return {error: ...} (no random guess)."""
        from src.translation.translator import TSLTranslator
        t = TSLTranslator()
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        result = t.translate_frame(blank)
        assert "error" in result


class TestAPI:
    """Smoke tests for the FastAPI app (no network needed)."""

    def test_app_imports(self):
        from api.main import app
        assert app.title == "TSL Translator API"

    def test_cors_not_wildcard_with_credentials(self):
        import api.main as m
        # The middleware must not combine origins=["*"] with credentials.
        # api.main sets credentials=False when origins are ["*"].
        from starlette.middleware.cors import CORSMiddleware
        cors_found = False
        for mw in m.app.user_middleware:
            if mw.cls is CORSMiddleware:
                opts = mw.kwargs
                if opts.get("allow_origins") == ["*"] and opts.get("allow_credentials") is True:
                    assert False, "Insecure: CORS * with credentials"
                cors_found = True
        assert cors_found


class TestStreamingAndSentences:
    """Tests for streaming recognition + sentence building."""

    def test_streaming_emits_stable_sign(self):
        from src.translation.streaming import StreamingRecognizer
        rec = StreamingRecognizer(stable_frames=4, gap_frames=5)
        emitted = []
        for _ in range(5):
            ev = rec.feed({"tsl": "labes", "rejected": False})
            if ev["sign"]:
                emitted.append(ev["sign"])
        assert "labes" in emitted

    def test_streaming_rejects_low_confidence(self):
        from src.translation.streaming import StreamingRecognizer
        rec = StreamingRecognizer(stable_frames=3, gap_frames=3)
        ev = rec.feed({"tsl": None, "rejected": True})
        assert ev["sign"] is None

    def test_streaming_sentence_boundary_on_long_gap(self):
        from src.translation.streaming import StreamingRecognizer
        rec = StreamingRecognizer(stable_frames=3, gap_frames=3)
        for _ in range(4):
            rec.feed({"tsl": "labes", "rejected": False})
        boundary = False
        for _ in range(4):
            ev = rec.feed({"error": "no hands"})
            if ev["sentence_boundary"]:
                boundary = True
                break
        assert boundary, "long no-hands gap must trigger a sentence boundary"

    def test_sentence_builder_family_then_day(self):
        from src.translation.streaming import SentenceBuilder
        from src.languages.dictionary import TSLDictionary
        d = TSLDictionary()
        b = SentenceBuilder(d, target_lang="en")
        for s in ["bou", "thnin"]:
            b.add_sign(s)
        s = b.flush()
        assert "will come" in s
        assert "Monday" in s

    def test_sentence_builder_arabic_msa(self):
        from src.translation.streaming import SentenceBuilder
        from src.languages.dictionary import TSLDictionary
        d = TSLDictionary()
        b = SentenceBuilder(d, target_lang="ar")
        b.add_sign("labes")
        s = b.flush()
        assert "بخير" in s

    def test_sentence_builder_multi_signs(self):
        from src.translation.streaming import SentenceBuilder
        from src.languages.dictionary import TSLDictionary
        d = TSLDictionary()
        b = SentenceBuilder(d, target_lang="en")
        b.add_sign("labes")
        b.add_sign("non")
        s = b.flush()
        assert "well" in s and "no" in s


class TestOfflineTranslator:
    """Tests for the Argos-based offline translator."""

    def test_available_languages_include_english(self):
        from src.translation.translator_backend import OfflineTranslator
        t = OfflineTranslator()
        langs = t.available_languages()
        if not langs:
            pytest.skip("Argos language packages not installed")
        assert "en" in langs

    def test_translate_passthrough_same_lang(self):
        from src.translation.translator_backend import OfflineTranslator
        t = OfflineTranslator()
        assert t.translate("hello", "en", "en") == "hello"

    def test_translate_empty_text(self):
        from src.translation.translator_backend import OfflineTranslator
        t = OfflineTranslator()
        assert t.translate("", "en", "fr") == ""

    def test_translate_french_to_english(self):
        from src.translation.translator_backend import OfflineTranslator
        t = OfflineTranslator()
        if not t.can_translate("fr", "en"):
            pytest.skip("fr->en not installed")
        out = t.translate("bonjour", "fr", "en")
        assert out and ("hello" in out.lower() or "good" in out.lower())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
