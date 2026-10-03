"""
Streaming TSL recognition + sentence building.

Two cooperating components:

StreamingRecognizer
    Consumes per-frame predictions (commentaar TSLTranslator.translate_frame),
    buffers the last N accepted predictions, detects stable "plateaus" and
    emits a single finalized sign when the prediction has been stable for
    `stable_frames` frames. Emits None on a no-hand / low-confidence gap.

SentenceBuilder
    Accumulates finalized signs into a phrase. Heuristically inserts TSL
    function words / transition tokens to produce a readable sentence in the
    target language:
      - detect "begin/end of sentence" by a NO-HAND gap >= gap_frames
      - when signs are sequential, inject connectors ("and", "then")
      - when a Famille label is followed by a Jours label, build
        "[Family] will come on [Day]"-like phrasing
      - emit the full sentence on a sentence boundary (long gap or explicit
        close)

The SentenceBuilder operates in the *source* language (the Tunisian words the
dictionary provides); offline translation to other languages (FR/EN/AR-MSA)
is then done by translator_backend.Argos.

The grammar is intentionally heuristic — real TSL grammar differs from spoken
Tunisian; we approximate so the output is readable instead of a bare list of
isolated labels.
"""

from collections import deque
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


# Connector tokens per output language (used by SentenceBuilder when stitching).
_CONNECTORS = {
    "en": {"and": "and", "then": "then", "is": "is", "says": "says", "the": "the", "has": "has"},
    "fr": {"and": "et", "then": "puis", "is": "est", "says": "dit", "the": "le", "has": "a"},
    "ar": {"and": "و", "then": "ثم", "is": "هو", "says": "يقول", "the": "ال", "has": "لديه"},
    "tunisian": {"and": "w", "then": "men ba3d", "is": "mch", "says": "y9oul", "the": "le", "has": "3and"},
}

# Sentence framing templates per output language.
# Position-holders: {SUBJECT} {OBJECT} {VERB} {DAY} {FAMILY} {DESTINATION} {TIME}
TEMPLATES = {
    "en": {
        "family_then_day":      "{FAMILY} will come on {DAY}",
        "family_then_dest":     "{FAMILY} goes to {DEST}",
        "demande_only":         "They say: {DEMANDE}",
        "transport_then_dest":  "Take {TRANSPORT} to {DEST}",
        "day_then_demande":      "On {DAY}: {DEMANDE}",
        "plain":                "They say: {SIGN}",
    },
    "fr": {
        "family_then_day":      "{FAMILY} viendra le {DAY}",
        "family_then_dest":     "{FAMILY} va à {DEST}",
        "demande_only":         "Ils demandent : {DEMANDE}",
        "transport_then_dest":  "Prenez le {TRANSPORT} vers {DEST}",
        "day_then_demande":     "Le {DAY} : {DEMANDE}",
        "plain":                "Ils disent : {SIGN}",
    },
    "ar": {
        "family_then_day":      "{FAMILY} سيأتي يوم {DAY}",
        "family_then_dest":     "{FAMILY} يذهب إلى {DEST}",
        "demande_only":         "هم يطلبون: {DEMANDE}",
        "transport_then_dest":  "خذ {TRANSPORT} إلى {DEST}",
        "day_then_demande":     "يوم {DAY}: {DEMANDE}",
        "plain":                "هم يقولون: {SIGN}",
    },
    "tunisian": {
        "family_then_day":      "{FAMILY} yeji el {DAY}",
        "family_then_dest":     "{FAMILY} ymshi lel {DEST}",
        "demande_only":         "ya9oul: {DEMANDE}",
        "transport_then_dest":  "khodh {TRANSPORT} lel {DEST}",
        "day_then_demande":     "el {DAY}: {DEMANDE}",
        "plain":                "ya9oul: {SIGN}",
    },
}


class StreamingRecognizer:
    """Buffer per-frame signs, emit stable signs + sentence boundaries.

    Lifecycle:
        for frame in stream:
            result = translator.translate_frame(frame)
            event = recognizer.feed(result)
            if event.sign is not None:
                sentence_builder.add_sign(event.sign)
            if event.sentence_boundary:
                sentence = sentence_builder.flush(target_lang)
    """

    def __init__(
        self,
        stable_frames: int = 4,
        gap_frames: int = 6,
        buffer_size: int = 12,
    ):
        """
        Args:
            stable_frames: minimum consecutive frames of the same accepted
                prediction to emit a finalized sign
            gap_frames: number of consecutive empty/low-confidence frames to
                trigger a sentence boundary (None followed by a flush)
            buffer_size: max number of frame results kept in the window
        """
        self.stable_frames = stable_frames
        self.gap_frames = gap_frames
        self.buffer = deque(maxlen=buffer_size)
        self._current_run_sign: Optional[str] = None
        self._current_run_count: int = 0
        self._gap_count: int = 0
        self._last_emitted_sign: Optional[str] = None

    def feed(self, result: Dict) -> Dict:
        """Feed a translate_frame result and emit an event.

        Returns dict with:
          'sign': finalized sign label or None
          'sentence_boundary': True if a long-enough gap was detected (caller
              should flush the sentence builder)
          'is_stable': True if a new stable sign was emitted
        """
        # Determine the "label" for this frame. result may be:
        #   - {"error": ...}    (no hands)
        #   - {"tsl": None, "rejected": True, ...}  (low confidence)
        #   - {"tsl": "labes", "rejected": False, ...}  (accepted)
        tsl = result.get("tsl") if isinstance(result, dict) else None
        rejected = result.get("rejected", True) if isinstance(result, dict) else True
        has_error = isinstance(result, dict) and "error" in result

        empty = has_error or rejected or tsl is None
        sign = None if empty else tsl

        event = {"sign": None, "sentence_boundary": False, "is_stable": False}
        self.buffer.append(sign)

        if empty:
            self._gap_count += 1
            # If we had an active run, finalize it now too (a short gap is fine),
            # unless it was already emitted by the stability path.
            if (
                self._current_run_sign is not None
                and self._current_run_count >= self.stable_frames
                and self._current_run_sign != self._last_emitted_sign
            ):
                event["sign"] = self._current_run_sign
                event["is_stable"] = True
                self._last_emitted_sign = self._current_run_sign
            self._current_run_sign = None
            self._current_run_count = 0
            # Fire the boundary exactly once, on the transition into the gap.
            if self._gap_count == self.gap_frames:
                event["sentence_boundary"] = True
            return event

        # Non-empty frame
        self._gap_count = 0
        if sign == self._current_run_sign:
            self._current_run_count += 1
        else:
            # A new sign starts. If the previous run was stable enough, emit it
            if self._current_run_sign is not None and self._current_run_count >= self.stable_frames:
                # Only emit if it differs from the previous emission (avoid duplicates)
                if self._current_run_sign != self._last_emitted_sign:
                    event["sign"] = self._current_run_sign
                    event["is_stable"] = True
                    self._last_emitted_sign = self._current_run_sign
            self._current_run_sign = sign
            self._current_run_count = 1

        # If the run just became stable enough, emit
        if self._current_run_count == self.stable_frames and self._current_run_sign != self._last_emitted_sign:
            event["sign"] = self._current_run_sign
            event["is_stable"] = True
            self._last_emitted_sign = self._current_run_sign
        return event

    def reset(self):
        self.buffer.clear()
        self._current_run_sign = None
        self._current_run_count = 0
        self._gap_count = 0
        self._last_emitted_sign = None


class SentenceBuilder:
    """Accumulate finalized signs and produce readable sentences.

    Uses dictionary category info and per-language templates to stitch signs
    into phrases. A "sentence boundary" (long no-hand gap) flushes the current
    sequence as a finalized sentence.

    Args:
        dictionary: a TSLDictionary (used for category lookup + per-lang
            translations of each sign)
        target_lang: target output language — "en", "fr", "ar", or "tunisian"
    """

    def __init__(
        self,
        dictionary,
        target_lang: str = "en",
    ):
        if target_lang not in TEMPLATES:
            target_lang = "en"
        self.dictionary = dictionary
        self.target_lang = target_lang
        self.signs_seq: List[str] = []  # canonical TSL labels accumulated
        self.completed_sentences: List[str] = []
        self._connectors = _CONNECTORS.get(target_lang, _CONNECTORS["en"])
        self._templates = TEMPLATES.get(target_lang, TEMPLATES["en"])

    def add_sign(self, tsl_label: str) -> None:
        # Dedup consecutive identical signs (a single long-held sign should
        # produce one word, not three).
        if self.signs_seq and self.signs_seq[-1] == tsl_label:
            return
        self.signs_seq.append(tsl_label)

    def flush(self) -> str:
        """Build a sentence from the accumulated signs and clear them.

        Returns the built sentence. If no signs were collected, returns "".
        """
        if not self.signs_seq:
            return ""
        sentence = self._build_sentence(self.signs_seq)
        self.completed_sentences.append(sentence)
        self.signs_seq = []
        return sentence

    def get_completed_sentences(self) -> List[str]:
        return list(self.completed_sentences)

    def reset(self) -> None:
        self.signs_seq = []
        self.completed_sentences = []

    def _localize(self, tsl_label: str) -> str:
        """Translate a single TSL label to the target language."""
        trans = self.dictionary.get_all_translations(tsl_label)
        if not trans:
            return tsl_label
        if self.target_lang in ("en", "english"):
            return trans.get("english") or trans.get("french") or tsl_label
        if self.target_lang in ("fr", "french"):
            return trans.get("french") or tsl_label
        if self.target_lang in ("ar", "arabic", "msa"):
            return trans.get("arabic_msa") or trans.get("arabic") or tsl_label
        if self.target_lang == "tunisian":
            return trans.get("tunisian") or tsl_label
        return tsl_label

    def _build_sentence(self, signs: List[str]) -> str:
        """Build a readable sentence from a list of canonical labels.

        Heuristics:
          1. Single sign -> "They say: {SIGN}"
          2. 2-sign sequences pick a template based on category combo
          3. >2 signs -> join the localized signs with connectors
        """
        if not signs:
            return ""
        if len(signs) == 1:
            return self._templates["plain"].format(SIGN=self._localize(signs[0]))

        cats = [self.dictionary.get_category(s) for s in signs]

        # Look for common 2-patterns by scanning the sequence.
        if len(signs) == 2:
            c1, c2 = cats
            s1, s2 = signs
            loc1, loc2 = self._localize(s1), self._localize(s2)
            if c1 == "Famille" and c2 == "Jours":
                return self._templates["family_then_day"].format(FAMILY=loc1, DAY=loc2)
            if c1 == "Famille" and c2 == "Destinations":
                return self._templates["family_then_dest"].format(FAMILY=loc1, DEST=loc2)
            if c1 == "Transport" and c2 == "Destinations":
                return self._templates["transport_then_dest"].format(TRANSPORT=loc1, DEST=loc2)
            if c1 == "Jours" and c2 == "Demandes":
                return self._templates["day_then_demande"].format(DAY=loc1, DEMANDE=loc2)
            if c1 == "Demandes" and c2 == "Jours":
                return self._templates["day_then_demande"].format(DAY=loc2, DEMANDE=loc1)
            # default 2-sign: loc1 + connector + loc2
            return f"{loc1} {self._connectors['and']} {loc2}"

        # >2 signs: localize each and join with the connector.
        localized = [self._localize(s) for s in signs]
        sep = f" {self._connectors['and']} "
        return sep.join(localized)
