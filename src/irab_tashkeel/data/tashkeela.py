"""Tashkeela corpus loader.

Supports three sources (in priority order):
1. Local text files (one sentence per line or whole books)
2. HuggingFace streaming dataset (Misraj/Sadeed_Tashkeela)
3. Kaggle-attached Tashkeela dataset under /kaggle/input/

Emits MTLExample objects with mask_diac=True, other masks False.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Optional

from ..models.tokenizer import (
    compute_word_offsets, normalize, text_to_diac_labels,
)
from .schema import MTLExample


MIN_LEN = 30
MAX_LEN = 500


def _split_into_sentences(text: str) -> Iterator[str]:
    """Split on Arabic/Latin sentence terminators; yield clean sentences."""
    # . ؟ ! ؛ followed by whitespace or EOL
    for chunk in re.split(r"[.؟!؛]+\s*", text):
        chunk = chunk.strip()
        if MIN_LEN <= len(chunk) <= MAX_LEN:
            yield chunk


def load_from_local_files(
    directory: Path | str, max_sentences: Optional[int] = None
) -> List[str]:
    """Load all .txt files from a directory, sentence-split them."""
    directory = Path(directory)
    out = []
    for f in directory.rglob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        text = normalize(text)
        for sent in _split_into_sentences(text):
            out.append(sent)
            if max_sentences and len(out) >= max_sentences:
                return out
    return out


def load_from_huggingface(max_sentences: int = 30000, dataset_name: str = "Misraj/Sadeed_Tashkeela") -> List[str]:
    """Stream N sentences from a HuggingFace dataset.

    Returns a list of diacritized strings. Requires `datasets` and internet.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets") from None

    ds = load_dataset(dataset_name, split="train", streaming=True)
    out = []
    for i, row in enumerate(ds):
        if len(out) >= max_sentences:
            break
        # Try the common field names
        text = row.get("text") or row.get("diacritized") or row.get("sentence") or ""
        text = normalize(text)
        if MIN_LEN <= len(text) <= MAX_LEN:
            out.append(text)
    return out


def _auto_detect_source() -> Optional[Path]:
    """Check common locations for a local Tashkeela copy."""
    candidates = [
        Path("/kaggle/input/tashkeela-arabic-diacritization-corpus"),
        Path("/kaggle/input/tashkeela"),
        Path("data/tashkeela"),
        Path("./tashkeela"),
    ]
    for p in candidates:
        if p.exists() and any(p.rglob("*.txt")):
            return p
    return None


def load_tashkeela_sentences(
    source: Optional[Path | str] = None,
    max_sentences: int = 30000,
    use_huggingface: bool = True,
) -> List[str]:
    """Return a list of diacritized Tashkeela sentences.

    Args:
        source: Path to a directory of .txt files. If None, auto-detects.
        max_sentences: Maximum number of sentences to return.
        use_huggingface: If source is None and no local copy is found, try HF streaming.
    """
    if source:
        return load_from_local_files(source, max_sentences=max_sentences)

    auto = _auto_detect_source()
    if auto is not None:
        return load_from_local_files(auto, max_sentences=max_sentences)

    if use_huggingface:
        return load_from_huggingface(max_sentences=max_sentences)

    raise FileNotFoundError(
        "No local Tashkeela found and use_huggingface=False. "
        "Provide a path or enable HF streaming."
    )


def sentences_to_examples(sentences: List[str]) -> List[MTLExample]:
    """Convert diacritized sentences into MTLExamples (diac head only)."""
    examples = []
    for i, sent in enumerate(sentences):
        bare_text, diac_ids = text_to_diac_labels(sent)
        if not bare_text.strip():
            continue
        word_offsets = compute_word_offsets(bare_text)
        n_words = len(word_offsets)
        examples.append(MTLExample(
            bare_text=bare_text,
            diac_labels=diac_ids,
            mask_diac=True,
            word_offsets=word_offsets,
            irab_labels=[10] * n_words,  # IRAB_PAD_ID
            mask_irab=False,
            err_labels=[0] * len(bare_text),
            mask_err=False,
            source="tashkeela",
            sent_id=f"tashkeela_{i}",
        ))
    return examples


def load_tashkeela_examples(
    source: Optional[Path | str] = None,
    max_sentences: int = 30000,
    use_huggingface: bool = True,
) -> List[MTLExample]:
    """One-call helper: load sentences, convert to examples."""
    sentences = load_tashkeela_sentences(source, max_sentences, use_huggingface)
    return sentences_to_examples(sentences)
