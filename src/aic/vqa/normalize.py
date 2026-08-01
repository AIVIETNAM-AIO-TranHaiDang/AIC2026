"""Language-general answer normalisation (note 18 §3.5).

Turns free-text answers into a comparison key so independently-read candidates
can be voted together: casefold + NFC, number words <-> digits, a diacritic
fold for a looser tie-breaker tier, colour-synonym folding, and per-answer-type
unit stripping. Every language-specific table (the number-word lexicon, the
compound-number words, the colour lexicon, the unit list) is config
(:class:`aic.config.VqaNormalizeConfig`), so the same code serves any language
by swapping those; the defaults are the Vietnamese set the AIC corpus needs.
Display keeps diacritics; only the comparison keys fold. This is where most
silent wrongness hides, so it is exhaustively table-tested.
"""

from __future__ import annotations

import re
import unicodedata

from aic.config import VqaNormalizeConfig


def nfc_casefold(text: str) -> str:
    """NFC-normalise and casefold (diacritics preserved)."""
    return unicodedata.normalize("NFC", text).strip().casefold()


def fold_diacritics(text: str) -> str:
    """Strip Vietnamese diacritics for a looser comparison tier.

    NFD then drop combining marks; đ/Đ have no combining form, so map them
    explicitly. Used only as a tie-breaker below exact-with-diacritics
    matching, never for display.
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def parse_number(text: str, cfg: VqaNormalizeConfig) -> int | None:
    """Parse a cardinal 0-99 (or a bare integer) to int, else None.

    Digits always parse. Words parse through ``cfg.number_words`` (single
    words) and, when ``cfg.ten_word``/``cfg.tens_word`` are set, the
    Vietnamese-shaped compound rule ("ten_word <unit>" -> 10+unit; "<n>
    tens_word [unit]" -> n*10[+unit]). Only a string that is ENTIRELY a number
    expression parses, so "một số" ("some") and "mười một người" do not become
    numbers here (unit stripping happens before this in normalize_answer). A
    language with a different number grammar sets those two words empty and
    relies on the single-word lexicon plus digits.
    """
    text = text.strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    words = text.split()
    units = cfg.number_words
    ten, tens = cfg.ten_word, cfg.tens_word
    if len(words) == 1:
        word = words[0]
        if ten and word == ten:
            return 10
        return units.get(word)
    if len(words) == 2:
        a, b = words
        # "<ten> <unit>" -> 10 + unit (mười lăm = 15, mười một = 11)
        if ten and a == ten and b in units:
            return 10 + units[b]
        # "<n> <tens>" -> n * 10 (hai mươi = 20)
        if tens and a in units and b == tens and 2 <= units[a] <= 9:
            return units[a] * 10
        return None
    if len(words) == 3:
        a, b, c = words
        # "<n> <tens> <unit>" -> n*10 + unit (hai mươi ba = 23, ... mốt = 21)
        if tens and a in units and b == tens and c in units and 2 <= units[a] <= 9:
            return units[a] * 10 + units[c]
        return None
    return None


def _canonical_color(text: str, cfg: VqaNormalizeConfig) -> str | None:
    """Fold a colour surface form to its canonical name, else None."""
    for canonical, variants in cfg.color_lexicon.items():
        for variant in variants:
            if nfc_casefold(variant) == text:
                return canonical
    return None


def _strip_units(text: str, answer_type: str, cfg: VqaNormalizeConfig) -> str:
    """Drop a trailing unit for this answer_type (longest unit first)."""
    units = sorted(cfg.unit_strip.get(answer_type, []), key=len, reverse=True)
    for unit in units:
        unit_cf = nfc_casefold(unit)
        if text == unit_cf:
            continue  # the whole answer is the unit; keep it
        if text.endswith(unit_cf):
            stripped = text[: -len(unit_cf)].strip()
            if stripped:
                return stripped
    return text


def normalize_answer(text: str, answer_type: str, cfg: VqaNormalizeConfig) -> str:
    """Comparison key for an answer: casefold + type-specific canonicalisation.

    - ``number``: strip units, then number-word (via cfg) or digit -> the
      decimal string ("ba" -> "3", "mười lăm" -> "15"); unparseable stays as
      normalised text (the type validator then demotes it).
    - ``color``: fold to the canonical colour when it hits the lexicon.
    - everything else: NFC casefold, diacritics kept.
    Empty input returns "".
    """
    key = nfc_casefold(text)
    if not key:
        return ""
    stripped = _strip_units(key, answer_type, cfg)
    if answer_type == "number":
        parsed = parse_number(stripped, cfg)
        return str(parsed) if parsed is not None else stripped
    if answer_type == "color":
        canonical = _canonical_color(stripped, cfg)
        return canonical if canonical is not None else stripped
    return stripped


def is_valid_for_type(
    normalized: str, answer_type: str, cfg: VqaNormalizeConfig
) -> bool:
    """Whether a normalised answer is well-formed for its type (else demote).

    Never drops — the operator still sees the raw card — but a malformed
    answer must not win the vote over a well-formed one.
    """
    if not normalized:
        return False
    if answer_type == "number":
        return bool(re.fullmatch(r"-?\d+", normalized))
    if answer_type == "color":
        return normalized in cfg.color_lexicon
    return True
