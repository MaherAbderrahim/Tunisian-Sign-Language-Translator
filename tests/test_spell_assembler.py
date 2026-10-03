"""Tests for SpellAssembler."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.translation.spell_assembler import SpellAssembler


class TestSpellAssembler:
    def test_basic_letter(self):
        asm = SpellAssembler(gap_frames=3)
        asm.feed_letter("ba")
        result = asm.flush()
        assert result != ""

    def test_space(self):
        asm = SpellAssembler(gap_frames=3)
        asm.feed_letter("ba")
        asm.feed_letter("waw")
        asm.feed_letter("space")
        asm.feed_letter("jim")
        result = asm.flush()
        assert " " in result

    def test_backspace(self):
        asm = SpellAssembler(gap_frames=3)
        asm.feed_letter("ha")
        asm.feed_letter("lam")
        asm.feed_letter("backspace")
        asm.feed_letter("ayn")
        result = asm.flush()
        # Should have 2 letters, not 3
        assert result != ""

    def test_empty_flush(self):
        asm = SpellAssembler(gap_frames=3)
        result = asm.flush()
        assert result == ""

    def test_repeated_letters(self):
        asm = SpellAssembler(gap_frames=3)
        asm.feed_letter("sin")
        asm.feed_letter("sin")
        asm.feed_letter("sin")
        result = asm.flush()
        assert result != ""

    def test_reset(self):
        asm = SpellAssembler(gap_frames=3)
        asm.feed_letter("ba")
        asm.reset()
        result = asm.flush()
        assert result == ""

    def test_multiple_words(self):
        asm = SpellAssembler(gap_frames=3)
        asm.feed_letter("ba")
        asm.feed_letter("waw")
        asm.feed_letter("nun")
        asm.feed_letter("space")
        asm.feed_letter("jim")
        asm.feed_letter("waw")
        asm.feed_letter("fa")
        result = asm.flush()
        assert " " in result
