"""Synthetic error injection for training the error detection head.

Takes gold-diacritized text and produces (corrupted_text, error_spans)
pairs. Error types:
  - hamza_drop: أ/إ/آ → ا
  - taa_marbuta_swap: ة → ه
  - case_swap: rotate final diacritic to a wrong case
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ..models.labels import (
    DAMMA, DAMMATAN, ERR_TO_ID, FATHA, FATHATAN, KASRA, KASRATAN,
)
from ..models.tokenizer import (
    compute_word_offsets, strip_diacritics, text_to_diac_labels,
)
from .schema import MTLExample

TA_MARBUTA = "\u0629"

ERROR_TYPES = ("hamza_drop", "taa_marbuta_swap", "case_swap")


def inject_hamza_drop(diacritized: str, rng: random.Random) -> Optional[Tuple[str, int, int]]:
    """Drop hamza on a random word that has one. Returns (corrupted_text, start_bare, end_bare).

    The bare positions are in the BARE (undiacritized) text — that's what the
    error head sees during inference.
    """
    words = diacritized.split()
    cands = [i for i, w in enumerate(words) if any(c in w for c in "أإآ")]
    if not cands:
        return None
    i = rng.choice(cands)
    words[i] = words[i].replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    corrupted = " ".join(words)

    # Compute the bare offset of the corrupted word
    bare_corrupted = strip_diacritics(corrupted)
    bare_words = bare_corrupted.split()
    offsets = compute_word_offsets(bare_corrupted)
    if i >= len(offsets):
        return None
    s, e = offsets[i]
    return corrupted, s, e


def inject_taa_marbuta_swap(diacritized: str, rng: random.Random) -> Optional[Tuple[str, int, int]]:
    words = diacritized.split()
    cands = [i for i, w in enumerate(words) if TA_MARBUTA in w]
    if not cands:
        return None
    i = rng.choice(cands)
    words[i] = words[i].replace(TA_MARBUTA, "ه")
    corrupted = " ".join(words)

    bare_corrupted = strip_diacritics(corrupted)
    offsets = compute_word_offsets(bare_corrupted)
    if i >= len(offsets):
        return None
    s, e = offsets[i]
    return corrupted, s, e


def inject_case_swap(diacritized: str, rng: random.Random) -> Optional[Tuple[str, int, int]]:
    """Rotate the final diacritic of a random word to a wrong case."""
    words = diacritized.split()
    # Only consider case-bearing word-final diacritics
    case_marks = {DAMMA, FATHA, KASRA, DAMMATAN, FATHATAN, KASRATAN}
    cands = [i for i, w in enumerate(words) if w and w[-1] in case_marks]
    if not cands:
        return None
    i = rng.choice(cands)
    last = words[i][-1]
    rotation = {
        DAMMA: FATHA, FATHA: KASRA, KASRA: DAMMA,
        DAMMATAN: FATHATAN, FATHATAN: KASRATAN, KASRATAN: DAMMATAN,
    }
    words[i] = words[i][:-1] + rotation[last]
    corrupted = " ".join(words)

    bare_corrupted = strip_diacritics(corrupted)
    offsets = compute_word_offsets(bare_corrupted)
    if i >= len(offsets):
        return None
    s, e = offsets[i]
    return corrupted, s, e


_INJECTORS = {
    "hamza_drop": inject_hamza_drop,
    "taa_marbuta_swap": inject_taa_marbuta_swap,
    "case_swap": inject_case_swap,
}

# Map error type → BIO class prefix
_ERR_TYPE_MAP = {
    "hamza_drop": "hamza",
    "taa_marbuta_swap": "taa",
    "case_swap": "case",
}


def corrupt_to_example(
    diacritized: str, error_type: str, rng: random.Random, source_id: str = "",
) -> Optional[MTLExample]:
    """Corrupt one gold sentence, return an MTLExample for error head training."""
    injector = _INJECTORS.get(error_type)
    if injector is None:
        raise ValueError(f"Unknown error type: {error_type}")
    result = injector(diacritized, rng)
    if result is None:
        return None
    corrupted, s_bare, e_bare = result

    bare_text = strip_diacritics(corrupted)
    # Build per-character BIO error labels
    err_labels = [ERR_TO_ID["O"]] * len(bare_text)
    err_prefix = _ERR_TYPE_MAP[error_type]
    if s_bare < len(err_labels):
        err_labels[s_bare] = ERR_TO_ID[f"B-{err_prefix}"]
        for k in range(s_bare + 1, min(e_bare, len(err_labels))):
            err_labels[k] = ERR_TO_ID[f"I-{err_prefix}"]

    word_offsets = compute_word_offsets(bare_text)
    n_words = len(word_offsets)

    return MTLExample(
        bare_text=bare_text,
        diac_labels=[0] * len(bare_text),
        mask_diac=False,
        word_offsets=word_offsets,
        irab_labels=[10] * n_words,  # IRAB_PAD_ID
        mask_irab=False,
        err_labels=err_labels,
        mask_err=True,
        source=f"synth_{_ERR_TYPE_MAP[error_type]}",
        sent_id=f"synth_{source_id}_{error_type}",
    )


def generate_synthetic_examples(
    gold_sentences: List[str],
    error_types: Tuple[str, ...] = ERROR_TYPES,
    per_type: int = 2000,
    seed: int = 42,
) -> List[MTLExample]:
    """Generate synthetic corruptions from a pool of gold-diacritized sentences."""
    rng = random.Random(seed)
    out: List[MTLExample] = []
    by_type: Dict[str, int] = {t: 0 for t in error_types}

    # Shuffle a copy so we sample varied inputs
    pool = list(gold_sentences)
    rng.shuffle(pool)

    pool_idx = 0
    attempts = 0
    max_attempts = per_type * len(error_types) * 10

    while any(by_type[t] < per_type for t in error_types) and attempts < max_attempts:
        if pool_idx >= len(pool):
            rng.shuffle(pool)
            pool_idx = 0
        sent = pool[pool_idx]
        pool_idx += 1
        attempts += 1

        for etype in error_types:
            if by_type[etype] >= per_type:
                continue
            ex = corrupt_to_example(sent, etype, rng, source_id=str(pool_idx))
            if ex is not None:
                out.append(ex)
                by_type[etype] += 1

    return out
