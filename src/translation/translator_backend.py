"""
Offline translation backend using Argos Translate (no API key, no network).

Supports: ar, en, fr, es, de (and any pair installed via argostranslate).
Falls back gracefully (returns the original text) if a language pair isn't
installed.

Usage:
    from src.translation.translator_backend import OfflineTranslator
    t = OfflineTranslator()
    t.translate("bonjour", "fr", "en")  # -> "hello"
    t.available_languages()              # -> ["en", "ar", "fr", "es", "de"]
"""

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

try:
    import argostranslate.translate as _argos_translate
    import argostranslate.package as _argos_package
    _ARGOS_AVAILABLE = True
except ImportError:
    _ARGOS_AVAILABLE = False
    logger.warning(
        "argostranslate not installed — OfflineTranslator will be a passthrough. "
        "Install with: pip install argostranslate && python -m "
        "argostranslate.package update && ...download desired pairs."
    )


# Aliases so callers can use friendly names; maps to ISO 639-1 codes.
_LANG_ALIASES = {
    "english": "en", "eng": "en",
    "arabic": "ar", "arabe": "ar", "arabic_msa": "ar", "msa": "ar",
    "french": "fr", "francais": "fr", "fran\u00e7ais": "fr",
    "spanish": "es", "espanol": "es", "espa\u00f1ol": "es",
    "german": "de", "deutsch": "de",
    "tunisian": "ar",  # tunisian dialect is mapped to Arabic for translation
    "dialect": "ar",
}


def _normalize_code(lang: str) -> str:
    if not lang:
        return ""
    lang = lang.strip().lower()
    if lang in _LANG_ALIASES:
        return _LANG_ALIASES[lang]
    # fall back to first 2 chars (e.g. "eng" -> "en", "fra" -> "fr")
    if len(lang) >= 2:
        return lang[:2]
    return lang


class OfflineTranslator:
    """Thin wrapper around argostranslate with caching and passthrough fallback."""

    def __init__(self):
        self._langs_cache: Optional[List] = None
        self._pair_cache = {}
        if _ARGOS_AVAILABLE:
            # Make sure the package index is up to date with installed pkgs.
            try:
                _argos_package.update_package_index()
            except Exception as e:
                logger.warning(f"Argos index update failed: {e}")

    def available_languages(self) -> List[str]:
        """Return ISO codes of installed languages (e.g. ['en','ar','fr'])."""
        if not _ARGOS_AVAILABLE:
            return []
        if self._langs_cache is None:
            self._langs_cache = list(_argos_translate.get_installed_languages())
        return sorted(l.code for l in self._langs_cache if l.code)

    def _get_pair(self, src: str, dst: str):
        key = (src, dst)
        if key in self._pair_cache:
            return self._pair_cache[key]
        if not _ARGOS_AVAILABLE:
            return None
        langs = _argos_translate.get_installed_languages()
        src_lang = next((l for l in langs if l.code == src), None)
        dst_lang = next((l for l in langs if l.code == dst), None)
        if src_lang is None or dst_lang is None:
            self._pair_cache[key] = None
            return None
        try:
            translator = src_lang.get_translation(dst_lang)
        except Exception as e:
            logger.warning(f"Argos cannot build translator {src}->{dst}: {e}")
            translator = None
        self._pair_cache[key] = translator
        return translator

    def can_translate(self, src: str, dst: str) -> bool:
        src = _normalize_code(src)
        dst = _normalize_code(dst)
        if not src or not dst or src == dst:
            return False
        return self._get_pair(src, dst) is not None

    def translate(self, text: str, src: str, dst: str) -> str:
        """
        Translate `text` from `src` to `dst`. Returns the original text if the
        pair isn't installed (graceful passthrough), so callers always get a
        usable string.

        Args:
            text: input text
            src: source language code or alias (e.g. "fr", "arabic", "tunisian")
            dst: target language code or alias
        """
        if not text:
            return text
        src = _normalize_code(src)
        dst = _normalize_code(dst)
        if not src or not dst or src == dst:
            return text
        translator = self._get_pair(src, dst)
        if translator is None:
            # If a direct pair isn't installed, try pivoting through English.
            if src != "en" and dst != "en":
                en_text = self.translate(text, src, "en")
                if en_text and en_text != text:
                    return self.translate(en_text, "en", dst)
            logger.info(f"Argos: no pair {src}->{dst}, returning original text")
            return text
        try:
            return translator.translate(text)
        except Exception as e:
            logger.warning(f"Argos translate failed {src}->{dst}: {e}")
            return text


# Module-level singleton for convenience
_default = None


def get_default_translator() -> OfflineTranslator:
    global _default
    if _default is None:
        _default = OfflineTranslator()
    return _default


def translate(text: str, src: str, dst: str) -> str:
    """Convenience function using the module-level singleton."""
    return get_default_translator().translate(text, src, dst)
