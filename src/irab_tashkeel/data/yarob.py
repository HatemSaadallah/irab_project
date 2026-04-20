"""Yarob i'rab loader.

Yarob (https://github.com/linuxscout/yarob) ships ~500-1000 manually annotated
sentences with per-word Arabic i'rab strings. The format is a free-text "WORD: IRAB"
layout intended for human readers — we parse it tolerantly and only keep examples
where the word count matches the sentence cleanly.

File format (consistent across `examples`, `examples-divers`):

    sentence text ending with a colon:
    word1: full i'rab description for this word.
    word2: another i'rab description.

Sentences are separated by blank lines. Some lines pack multiple "WORD: IRAB"
chunks; some i'rab descriptions wrap to a continuation line (no colon).

Repository: https://github.com/linuxscout/yarob
"""

from __future__ import annotations

import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ..models.labels import IRAB_TO_ID
from ..models.tokenizer import (
    compute_word_offsets, strip_diacritics, text_to_diac_labels,
)
from .schema import MTLExample


YAROB_REPO = "https://github.com/linuxscout/yarob.git"
YAROB_FILES = ("examples", "examples-divers", "إعراب في إعراب.txt")
# example-part-quran and examples-part-jazeera have messier headers (Quran braces,
# quoted sentences, prefix prose) — skip until we have a more robust parser.

# Unicode bidirectional control + invisible chars that pollute the إعراب file.
_INVISIBLE_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\u00A0]")
# Arabic + ASCII colon variants
_COLON_RE = re.compile(r"\s*[:：]\s*")
# Leading numbering like "1- ", "84 - ", "84 ـ "
_LEAD_NUM_RE = re.compile(r"^\s*\d+\s*[-ـ]\s*")
# Trailing punctuation/quote chars to strip from the sentence header
_TRAIL_PUNCT = " :.,؟!\"'«»{}[]()،؛"


def clone_yarob(target_dir: Path) -> Path:
    """Shallow-clone the yarob repository if not already present."""
    target_dir = Path(target_dir)
    if target_dir.exists() and (target_dir / "data-source").exists():
        return target_dir
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", YAROB_REPO, str(target_dir)],
        check=True, capture_output=True,
    )
    return target_dir


def _clean_text(s: str) -> str:
    """Strip bidi controls, NFC-normalize, collapse whitespace."""
    s = _INVISIBLE_RE.sub("", s)
    s = unicodedata.normalize("NFC", s)
    return s


def _normalize_word(w: str) -> str:
    """For matching i'rab words to sentence words: strip diacritics + leading 'ال'."""
    bare = strip_diacritics(_clean_text(w)).strip(_TRAIL_PUNCT)
    return bare


def _parse_irab_chunks(body_text: str) -> List[Tuple[str, str]]:
    """Extract (word, irab_text) pairs from a multi-line i'rab body.

    Handles multiple chunks per line and treats lines without a colon as
    continuations of the previous chunk's i'rab.
    """
    body_text = _clean_text(body_text)

    # Walk through colons. Each colon marks the boundary between a word
    # (everything since the last sentence-end marker) and its i'rab.
    # We tokenize loosely: split on Arabic/ASCII colon, then post-process.

    pairs: List[List[str]] = []  # list of [word, irab]
    # Approach: use a regex to find sequences "word_token : irab_until_next_word_token_colon_or_end".
    # word_token = a short Arabic-letter sequence (1–10 chars), no internal spaces.

    pattern = re.compile(
        r"(?P<word>[\u0621-\u064A\u0670\u0671\uFEFB-\uFEFC]+)"  # arabic letters (no diacritics here)
        r"\s*[:：]\s*"
        r"(?P<irab>.+?)"
        r"(?=(?:\s+[\u0621-\u064A\u0670\u0671\uFEFB-\uFEFC]+\s*[:：])|\s*$)",
        flags=re.DOTALL,
    )
    # Strip diacritics from the body for matching purposes — but we want to keep
    # the diacritics in the source for the original word display. Instead, scan
    # the diacritized text directly with a tolerant pattern.

    # Use a more permissive pattern allowing diacritics inside the word.
    pattern_diac = re.compile(
        r"(?P<word>[\u0621-\u064A\u0670\u0671\u064B-\u0652]+)"
        r"\s*[:：]\s*"
        r"(?P<irab>.+?)"
        r"(?=(?:\s+[\u0621-\u064A\u0670\u0671\u064B-\u0652]+\s*[:：])|\s*$)",
        flags=re.DOTALL,
    )

    text = re.sub(r"\s+", " ", body_text).strip()
    out: List[Tuple[str, str]] = []
    for m in pattern_diac.finditer(text):
        word = m.group("word").strip()
        irab = m.group("irab").strip().rstrip(".،؛: ")
        if word and irab:
            out.append((word, irab))
    return out


def _split_examples(text: str) -> Iterable[Tuple[str, str]]:
    """Split file text into (header_line, body_text) per example, by blank-line separators."""
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # Skip metadata-only blocks: ref:, type:, #-prefixed.
        first = lines[0].strip()
        if first.startswith(("ref:", "type:", "#")) or first.lower().startswith("done"):
            continue
        # Header is the first line; body is the rest joined.
        yield lines[0], "\n".join(lines[1:])


def _parse_file(path: Path) -> List[MTLExample]:
    text = _clean_text(path.read_text(encoding="utf-8", errors="replace"))
    examples: List[MTLExample] = []

    for header, body in _split_examples(text):
        # Clean header: drop numbering, quotes, trailing colon/punct.
        header = _LEAD_NUM_RE.sub("", header).strip()
        header = header.strip(_TRAIL_PUNCT).strip()
        if not header:
            continue
        # Skip headers that look like prose/Quran framing (e.g. start with "قال").
        if header.startswith("قال "):
            continue

        # The header is the diacritized sentence — derive bare text + diac labels.
        diac_text = _clean_text(header)
        try:
            bare_text, diac_ids = text_to_diac_labels(diac_text)
        except Exception:
            continue
        bare_text = bare_text.strip()
        if not bare_text:
            continue

        bare_words = bare_text.split()
        if len(bare_words) < 2 or len(bare_words) > 30:
            continue

        irab_pairs = _parse_irab_chunks(body)
        if not irab_pairs:
            continue

        # Positional alignment: i'rab pairs in order should map one-to-one with words.
        # Tolerate count mismatches by skipping (we want clean training data).
        if len(irab_pairs) != len(bare_words):
            continue

        # Verify the i'rab pair words actually match (after stripping diacritics)
        # the corresponding sentence words. If most match, accept.
        match_count = 0
        for (irab_word, _), bare_word in zip(irab_pairs, bare_words):
            if _normalize_word(irab_word) == _normalize_word(bare_word):
                match_count += 1
        if match_count < max(1, int(0.7 * len(bare_words))):
            continue

        # Build labels.
        word_offsets = compute_word_offsets(bare_text)
        irab_targets = [pair[1] for pair in irab_pairs]
        irab_labels = [IRAB_TO_ID["other"]] * len(word_offsets)

        # Recompute diac_ids against the cleaned bare_text length — the strip()
        # above may have changed lengths. Rebuild from the original diac_text.
        bare_recomputed, diac_recomputed = text_to_diac_labels(diac_text)
        if bare_recomputed.strip() != bare_text or len(diac_recomputed) != len(bare_recomputed):
            # Fall back to no-diac supervision.
            diac_ids = [0] * len(bare_text)
            mask_diac = False
        else:
            # Align: pick the diac labels for the stripped bare_text region.
            offset = bare_recomputed.find(bare_text)
            if offset < 0:
                diac_ids = [0] * len(bare_text)
                mask_diac = False
            else:
                diac_ids = diac_recomputed[offset : offset + len(bare_text)]
                mask_diac = True

        if len(diac_ids) != len(bare_text):
            diac_ids = [0] * len(bare_text)
            mask_diac = False

        examples.append(MTLExample(
            bare_text=bare_text,
            diac_labels=diac_ids,
            mask_diac=mask_diac,
            word_offsets=word_offsets,
            irab_labels=irab_labels,
            mask_irab=True,
            err_labels=[0] * len(bare_text),
            mask_err=False,
            source="yarob",
            sent_id=path.name,
            irab_targets=irab_targets,
        ))

    return examples


def load_yarob_examples(
    repo_dir: Path | str = "data/yarob_src",
    download_if_missing: bool = True,
) -> List[MTLExample]:
    """One-call helper: clone yarob (if needed), parse all supported files."""
    repo_dir = Path(repo_dir)
    if download_if_missing:
        try:
            clone_yarob(repo_dir)
        except subprocess.CalledProcessError as e:
            print(f"⚠ yarob clone failed ({e.stderr.decode().strip() if e.stderr else e}); skipping yarob")
            return []

    data_dir = repo_dir / "data-source"
    if not data_dir.exists():
        return []

    all_examples: List[MTLExample] = []
    for fname in YAROB_FILES:
        path = data_dir / fname
        if not path.exists():
            continue
        try:
            ex = _parse_file(path)
            all_examples.extend(ex)
        except Exception as e:
            print(f"⚠ failed to parse {fname}: {e}")
    return all_examples
