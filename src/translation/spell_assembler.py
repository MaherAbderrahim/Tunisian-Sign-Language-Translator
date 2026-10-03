"""
Spell Assembler - Accumulates letters into words during finger spelling.

Handles:
  - Letter accumulation into the current word
  - Word finalization on pause (no hand detected)
  - Space sign for explicit word separation
  - Backspace sign for correction
  - Full sentence flush
"""

from typing import List, Optional, Dict
from src.languages.alphabet import TSLAlphabet


# Special control signs for finger spelling
# MUST match SPELL_LABELS in alphabet.py (lowercase)
SPELL_SPACE = "space"
SPELL_BACKSPACE = "backspace"


class SpellAssembler:
    """
    Assemble des lettres en mots pendant le mode doigté.

    Cycle de vie:
    1. Lettres détectées stables → ajoutées au mot courant
    2. Pause (main basse / gap) → mot finalisé
    3. "SPACE" → séparateur de mot explicite
    4. "BACKSPACE" → supprime la dernière lettre
    5. "flush()" → retourne la phrase complète et reset
    """

    def __init__(self, gap_frames: int = 8):
        """
        Args:
            gap_frames: Nombre de frames vides avant de finaliser le mot courant.
        """
        self.alphabet = TSLAlphabet()
        self.gap_frames = gap_frames

        # State
        self.current_word: List[str] = []  # Lettres du mot courant (labels alphabet)
        self.completed_words: List[str] = []  # Mots finalisés (strings arabe)
        self._gap_count: int = 0  # Compteur de frames vides consécutives

    def feed_letter(self, letter_label: str) -> Dict[str, object]:
        """
        Ajoute une lettre au mot courant.

        Args:
            letter_label: Label de la lettre (ex: "alef", "ba", "digit_1")

        Returns:
            Dict avec l'état courant:
            {
                "current_word": str,        # Mot en cours (arabe)
                "completed_words": list,    # Mots finalisés
                "event": str,               # "letter_added" | "space" | "backspace" | None
                "sentence": str,            # Phrase complète
            }
        """
        self._gap_count = 0  # Reset gap on any activity

        # Handle special control signs
        if letter_label == SPELL_SPACE:
            return self.feed_space()
        if letter_label == SPELL_BACKSPACE:
            return self.feed_backspace()

        # Validate letter
        arabic_char = self.alphabet.get_arabic(letter_label)
        if arabic_char is None:
            return self._state("unknown_letter")

        # Add letter to current word
        self.current_word.append(letter_label)
        return self._state("letter_added")

    def feed_space(self) -> Dict[str, object]:
        """Finalise le mot courant et commence un nouveau."""
        if self.current_word:
            word = self._build_word(self.current_word)
            self.completed_words.append(word)
            self.current_word = []
        return self._state("space")

    def feed_backspace(self) -> Dict[str, object]:
        """Supprime la dernière lettre du mot courant."""
        if self.current_word:
            self.current_word.pop()
        return self._state("backspace")

    def on_empty_frame(self) -> Dict[str, object]:
        """
        Appelé quand aucune main n'est détectée.
        Incrémente le compteur de gap. Si le gap dépasse le seuil,
        finalise le mot courant.
        """
        self._gap_count += 1
        if self._gap_count >= self.gap_frames and self.current_word:
            word = self._build_word(self.current_word)
            self.completed_words.append(word)
            self.current_word = []
            self._gap_count = 0
            return self._state("word_finalized")
        return self._state(None)

    def flush(self) -> str:
        """
        Finalise le mot courant et retourne la phrase complète.
        Reset l'état.
        """
        if self.current_word:
            word = self._build_word(self.current_word)
            self.completed_words.append(word)
            self.current_word = []

        sentence = " ".join(self.completed_words)
        self.completed_words = []
        self._gap_count = 0
        return sentence

    def get_current_word(self) -> str:
        """Retourne le mot en cours (arabe)."""
        return self._build_word(self.current_word)

    def get_sentence(self) -> str:
        """Retourne la phrase complète sans reset."""
        parts = list(self.completed_words)
        if self.current_word:
            parts.append(self._build_word(self.current_word))
        return " ".join(parts)

    def reset(self):
        """Reset complet de l'assembler."""
        self.current_word = []
        self.completed_words = []
        self._gap_count = 0

    def _build_word(self, letters: List[str]) -> str:
        """Construit un mot à partir d'une liste de labels lettres."""
        return "".join(
            self.alphabet.get_arabic(label) or "?" for label in letters
        )

    def _state(self, event: Optional[str]) -> Dict[str, object]:
        """Retourne l'état courant."""
        return {
            "current_word": self.get_current_word(),
            "completed_words": list(self.completed_words),
            "event": event,
            "sentence": self.get_sentence(),
        }
