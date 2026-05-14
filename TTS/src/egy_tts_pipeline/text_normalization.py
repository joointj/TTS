from __future__ import annotations

import re
from collections import Counter

from .config import NormalizationConfig

ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([،؛!؟.,:])")
SPACE_AFTER_PUNCT_RE = re.compile(r"([،؛!؟.,:])(?=\S)")

DIALECT_MARKERS = {
    "عايز",
    "عاوزه",
    "عاوزة",
    "عايزين",
    "محتاج",
    "محتاجة",
    "دلوقتي",
    "امبارح",
    "بكرة",
    "معلش",
    "لسه",
    "إزاي",
    "ازاي",
    "ليه",
    "فين",
    "شوية",
    "أوي",
    "اوي",
    "كده",
}

BORROWED_TOKENS = {
    "الواي",
    "فاي",
    "الباسوورد",
    "الأبلكيشن",
    "الابلكيشن",
    "الكود",
    "اللينك",
    "أونلاين",
    "اونلاين",
    "أوبر",
    "اوبر",
}


def normalize_egyptian_text(text: str, config: NormalizationConfig) -> str:
    value = text.strip()
    if config.remove_diacritics:
        value = ARABIC_DIACRITICS_RE.sub("", value)
    if config.strip_tatweel:
        value = value.replace("ـ", "")
    if config.normalize_alef:
        value = re.sub(r"[أإآٱ]", "ا", value)
    if config.normalize_alef_maqsura:
        value = value.replace("ى", "ي")
    if config.normalize_punctuation_spacing:
        value = SPACE_BEFORE_PUNCT_RE.sub(r"\1", value)
        value = SPACE_AFTER_PUNCT_RE.sub(r"\1 ", value)
    if config.normalize_whitespace:
        value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def preflight_tts_text(text: str) -> str:
    value = text.replace("“", '"').replace("”", '"').replace("…", "...")
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def inspect_text_features(text: str) -> dict[str, object]:
    tokens = [token for token in text.replace("؟", "").replace("،", "").split() if token]
    token_counter = Counter(tokens)
    repeated_token_count = sum(1 for token, count in token_counter.items() if count > 1)
    markers = sorted({token for token in tokens if token in DIALECT_MARKERS})
    borrowed = sorted({token for token in tokens if token in BORROWED_TOKENS})
    has_latin = any("a" <= char.lower() <= "z" for char in text)
    has_digits = any(char.isdigit() for char in text)
    return {
        "char_count": len(text),
        "word_count": len(tokens),
        "dialect_markers": markers,
        "borrowed_tokens": borrowed,
        "contains_latin": has_latin,
        "contains_digits": has_digits,
        "repeated_token_count": repeated_token_count,
    }
