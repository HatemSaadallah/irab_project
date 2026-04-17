"""Character-level tokenizer for Arabic.

Not really a "tokenizer" in the subword sense — just a char → id mapping
plus helpers for the per-character label alignment.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

from .labels import (
    CHAR_TO_ID, DIAC_LABELS, DIAC_TO_ID, ID_TO_CHAR, PAD_ID, UNK_ID, VOCAB_SIZE,
    canonicalize_diac,
)


TATWEEL = "\u0640"


def normalize(text: str) -> str:
    """NFC normalize + strip tatweel. Idempotent."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(TATWEEL, "")
    return text


def is_arabic_letter(ch: str) -> bool:
    """True if the character is in Arabic letter range (U+0621–U+064A, plus wasla)."""
    return ("\u0621" <= ch <= "\u064A") or ch == "\u0671"


def is_diacritic(ch: str) -> bool:
    """True if the character is an Arabic diacritic (U+064B–U+0652)."""
    return "\u064B" <= ch <= "\u0652"


def text_to_diac_labels(diacritized: str) -> Tuple[str, List[int]]:
    """Split diacritized text into (bare_text, per_bare_char_diac_class).

    For each character in the BARE output, emit a diacritic class ID:
    - 0 if no diacritic follows this character
    - 1..14 for each known diacritic combination

    Whitespace and punctuation get class 0.
    """
    bare_chars = []
    diac_ids = []
    i = 0
    while i < len(diacritized):
        c = diacritized[i]
        if is_arabic_letter(c):
            bare_chars.append(c)
            # Collect any trailing diacritics
            diacs = ""
            j = i + 1
            while j < len(diacritized) and is_diacritic(diacritized[j]):
                diacs += diacritized[j]
                j += 1
            diacs = canonicalize_diac(diacs)
            diac_ids.append(DIAC_TO_ID.get(diacs, 0))
            i = j
        elif is_diacritic(c):
            # Stray diacritic — skip (shouldn't happen if input is clean)
            i += 1
        else:
            bare_chars.append(c)
            diac_ids.append(0)
            i += 1
    return "".join(bare_chars), diac_ids


def strip_diacritics(text: str) -> str:
    """Remove all diacritics from text, keeping letters + spaces + punctuation."""
    return re.sub(r"[\u064B-\u0652]+", "", text)


def compute_word_offsets(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) char offsets for each whitespace-separated word in text."""
    offsets = []
    pos = 0
    for word in text.split():
        offsets.append((pos, pos + len(word)))
        pos += len(word) + 1  # +1 for the space
    return offsets


def encode_chars(text: str, max_len: int = 512) -> Tuple[List[int], List[int]]:
    """Character → ID encoding with padding.

    Returns (char_ids, attention_mask) both of length max_len.
    attention_mask is 1 for real chars, 0 for padding.
    """
    # Truncate BEFORE padding
    truncated = text[:max_len]
    ids = [CHAR_TO_ID.get(c, UNK_ID) for c in truncated]
    mask = [1] * len(ids)
    # Pad to max_len
    pad_needed = max_len - len(ids)
    if pad_needed > 0:
        ids.extend([PAD_ID] * pad_needed)
        mask.extend([0] * pad_needed)
    return ids, mask


def decode_diacritized(bare_text: str, diac_ids: List[int]) -> str:
    """Inverse of text_to_diac_labels: given bare text + per-char class, reconstruct diacritized text."""
    out = []
    for i, c in enumerate(bare_text):
        out.append(c)
        if i < len(diac_ids) and is_arabic_letter(c):
            cls = diac_ids[i]
            if 0 < cls < len(DIAC_LABELS):
                out.append(DIAC_LABELS[cls])
    return "".join(out)
