"""Tests for alphabet labels consistency."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.languages.alphabet import SPELL_LABELS, INDEX_TO_LABEL, LABEL_TO_INDEX, TSLAlphabet


class TestAlphabetLabels:
    def test_spell_labels_count(self):
        assert len(SPELL_LABELS) == 31

    def test_index_to_label_consistency(self):
        for i, label in enumerate(SPELL_LABELS):
            assert INDEX_TO_LABEL[i] == label

    def test_label_to_index_consistency(self):
        for i, label in enumerate(SPELL_LABELS):
            assert LABEL_TO_INDEX[label] == i

    def test_round_trip(self):
        for label in SPELL_LABELS:
            idx = LABEL_TO_INDEX[label]
            assert INDEX_TO_LABEL[idx] == label

    def test_tsl_alphabet_matches(self):
        alpha = TSLAlphabet()
        assert alpha.get_all_labels() == SPELL_LABELS

    def test_no_duplicates(self):
        assert len(SPELL_LABELS) == len(set(SPELL_LABELS))

    def test_class_names_txt_matches(self):
        class_names_path = Path(__file__).resolve().parent.parent / "data" / "alphabet" / "processed" / "class_names.txt"
        if class_names_path.exists():
            with open(class_names_path) as f:
                file_labels = [l.strip() for l in f.readlines()]
            assert file_labels == SPELL_LABELS
