"""
TSL Alphabet - Mapping between Tunisian Sign Language finger spelling and text.

The trained spell model uses 31 classes (28 Arabic letters + Space + Backspace + Finish).
Labels and ordering MUST match the training data exactly (from ArSL-31-Pilot dataset).

CRITICAL: Do NOT change the ordering of SPELL_LABELS — it must match the training data.
"""

from typing import Dict, List, Optional


# ── TRAINING ORDER — single source of truth ──
# This order matches the ArSL-31-Pilot dataset as parsed by prepare_arsl.py.
# The trained model's output index i corresponds to SPELL_LABELS[i].
SPELL_LABELS = [
    "alef", "ba", "ta", "tha", "jim", "ha", "kha", "dal", "dhal", "ra", "zay",
    "sin", "shin", "sad", "dad", "ta_h", "za", "ayn", "ghayn", "fa", "qaf",
    "kaf", "lam", "mim", "nun", "ha2", "waw", "ya",
    "space", "backspace", "finish",
]

# Translation table for each label
_SPELL_INFO: Dict[str, Dict[str, str]] = {
    "alef":      {"arabic": "أ",  "french": "a",    "english": "a"},
    "ba":        {"arabic": "ب",  "french": "b",    "english": "b"},
    "ta":        {"arabic": "ت",  "french": "t",    "english": "t"},
    "tha":       {"arabic": "ث",  "french": "th",   "english": "th"},
    "jim":       {"arabic": "ج",  "french": "j",    "english": "j"},
    "ha":        {"arabic": "ح",  "french": "h",    "english": "h"},
    "kha":       {"arabic": "خ",  "french": "kh",   "english": "kh"},
    "dal":       {"arabic": "د",  "french": "d",    "english": "d"},
    "dhal":      {"arabic": "ذ",  "french": "dh",   "english": "dh"},
    "ra":        {"arabic": "ر",  "french": "r",    "english": "r"},
    "zay":       {"arabic": "ز",  "french": "z",    "english": "z"},
    "sin":       {"arabic": "س",  "french": "s",    "english": "s"},
    "shin":      {"arabic": "ش",  "french": "sh",   "english": "sh"},
    "sad":       {"arabic": "ص",  "french": "s'",   "english": "s'"},
    "dad":       {"arabic": "ض",  "french": "d'",   "english": "d'"},
    "ta_h":      {"arabic": "ط",  "french": "t'",   "english": "t'"},
    "za":        {"arabic": "ظ",  "french": "z'",   "english": "z'"},
    "ayn":       {"arabic": "ع",  "french": "3",    "english": "3"},
    "ghayn":     {"arabic": "غ",  "french": "gh",   "english": "gh"},
    "fa":        {"arabic": "ف",  "french": "f",    "english": "f"},
    "qaf":       {"arabic": "ق",  "french": "q",    "english": "q"},
    "kaf":       {"arabic": "ك",  "french": "k",    "english": "k"},
    "lam":       {"arabic": "ل",  "french": "l",    "english": "l"},
    "mim":       {"arabic": "م",  "french": "m",    "english": "m"},
    "nun":       {"arabic": "ن",  "french": "n",    "english": "n"},
    "ha2":       {"arabic": "ه",  "french": "h'",   "english": "h'"},
    "waw":       {"arabic": "و",  "french": "w",    "english": "w"},
    "ya":        {"arabic": "ي",  "french": "y",    "english": "y"},
    "space":     {"arabic": " ",  "french": " ",    "english": " "},
    "backspace": {"arabic": "⌫", "french": "⌫",   "english": "⌫"},
    "finish":    {"arabic": "⏎", "french": "⏎",   "english": "⏎"},
}

# Convenience: index -> label mapping (for model output decoding)
INDEX_TO_LABEL = {i: label for i, label in enumerate(SPELL_LABELS)}

# Convenience: label -> index mapping (for encoding)
LABEL_TO_INDEX = {label: i for i, label in enumerate(SPELL_LABELS)}


class TSLAlphabet:
    """
    TSL finger spelling alphabet.

    IMPORTANT: The label ordering matches the trained model exactly.
    Do NOT sort or reorder labels — the model's output index i maps to SPELL_LABELS[i].
    """

    def __init__(self):
        self._labels = list(SPELL_LABELS)

        # Build lookup tables (in training order)
        self.label_to_arabic: Dict[str, str] = {
            label: _SPELL_INFO[label]["arabic"] for label in self._labels
        }
        self.label_to_french: Dict[str, str] = {
            label: _SPELL_INFO[label]["french"] for label in self._labels
        }
        self.label_to_english: Dict[str, str] = {
            label: _SPELL_INFO[label]["english"] for label in self._labels
        }

        # Reverse lookups
        self.arabic_to_label: Dict[str, str] = {
            _SPELL_INFO[label]["arabic"]: label for label in self._labels
        }
        self.sign_to_index: Dict[str, int] = dict(LABEL_TO_INDEX)

    def get_letter(self, label: str) -> Optional[Dict[str, str]]:
        if label not in _SPELL_INFO:
            return None
        return {"label": label, **_SPELL_INFO[label]}

    def get_arabic(self, label: str) -> Optional[str]:
        return self.label_to_arabic.get(label)

    def get_translations(self, label: str) -> Optional[Dict[str, str]]:
        if label not in _SPELL_INFO:
            return None
        return {"label": label, **_SPELL_INFO[label]}

    def get_label_from_arabic(self, arabic_char: str) -> Optional[str]:
        return self.arabic_to_label.get(arabic_char)

    def get_all_labels(self) -> List[str]:
        """Get all labels in TRAINING ORDER (model output index -> label)."""
        return list(self._labels)

    def get_label_for_index(self, index: int) -> Optional[str]:
        """Get label for a model output index."""
        if 0 <= index < len(self._labels):
            return self._labels[index]
        return None

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, label: str) -> bool:
        return label in _SPELL_INFO
