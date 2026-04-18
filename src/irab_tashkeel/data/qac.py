"""Quranic Arabic Corpus parser.

Input: the quran-morphology.txt file (tab-separated segments, one per line).
Output: list of per-verse dicts with words + derived i'rab roles.
"""

from __future__ import annotations

import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..models.labels import IRAB_TO_ID, IRAB_PAD_ID
from ..models.tokenizer import (
    compute_word_offsets, is_arabic_letter, text_to_diac_labels,
)
from .schema import MTLExample


QAC_URL = "https://raw.githubusercontent.com/mustafa0x/quran-morphology/master/quran-morphology.txt"


def download_qac(target_path: Path) -> Path:
    """Download QAC morphology if not already present."""
    target_path = Path(target_path)
    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(QAC_URL, target_path)
    return target_path


def _parse_features(feat_str: str) -> Dict[str, str]:
    """Parse the pipe-separated feature string into a dict."""
    out = {}
    for f in feat_str.split("|"):
        if ":" in f:
            k, v = f.split(":", 1)
            out[k] = v
        elif f:
            out[f] = "True"
    return out


def parse_qac(path: Path) -> List[Dict]:
    """Parse QAC morphology file into a list of verse dicts.

    Each verse dict has:
        ref: "CH:V"
        segments: list of per-segment dicts (form, tag, features, word_idx, seg_idx)
    """
    verses = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            loc, form, tag, feat_str = parts[:4]
            m = re.match(r"\(?(\d+):(\d+):(\d+):(\d+)\)?", loc)
            if not m:
                continue
            ch, v, w, seg = (int(x) for x in m.groups())
            verses[(ch, v)].append({
                "chapter": ch, "verse": v, "word_idx": w, "seg_idx": seg,
                "form": form, "tag": tag, "features": _parse_features(feat_str),
            })

    result = []
    for key in sorted(verses.keys()):
        result.append({"ref": f"{key[0]}:{key[1]}", "segments": verses[key]})
    return result


def _aggregate_segments_to_words(segments: List[Dict]) -> List[Dict]:
    """Group segments by word_idx, merge into one entry per word."""
    by_word = defaultdict(list)
    for s in segments:
        by_word[s["word_idx"]].append(s)

    words = []
    for w_idx in sorted(by_word.keys()):
        segs = sorted(by_word[w_idx], key=lambda s: s["seg_idx"])
        surface = "".join(s["form"] for s in segs)
        # The "main" segment is usually the stem (not prefix/suffix).
        # Find the first N/V/ADJ/PN/... segment; fall back to first.
        content_tags = {"N", "V", "PN", "ADJ", "PRON", "DEM", "REL", "IMPN", "IMPV"}
        content_seg = next((s for s in segs if s["tag"] in content_tags), segs[0])
        words.append({
            "surface": surface,
            "tag": content_seg["tag"],
            "features": content_seg["features"],
            "all_segments": segs,
        })
    return words


def qac_word_to_irab_role(word: Dict, prev: Optional[Dict] = None) -> str:
    """Derive a coarse i'rab role from a QAC word.

    Uses POS + case + immediate context. See docs/DATA.md for the full mapping.
    """
    tag = word["tag"]
    case = word["features"].get("CASE", "")

    if tag in {"V", "IMPV"}:
        return "fiil"
    if tag == "P":
        return "harf_jarr"
    if tag in {"CONJ", "SUB", "CIRC", "REM"}:
        return "harf_atf"
    if tag == "NEG":
        return "harf_nafy"
    if tag in {"REL", "DEM", "PRON", "PRO"}:
        return "mabni_noun"
    if tag in {"N", "PN", "IMPN", "ADJ"}:
        if case == "NOM":
            return "N_marfu"
        if case == "ACC":
            return "N_mansub"
        if case == "GEN":
            prev_tag = prev.get("tag") if prev else None
            if prev_tag == "P":
                return "ism_majrur"
            return "mudaf_ilayh"
    return "other"


def qac_verses_to_examples(verses: List[Dict], max_verses: Optional[int] = None) -> List[MTLExample]:
    """Convert parsed QAC verses into MTLExample objects.

    Each verse becomes one example with both mask_diac=True and mask_irab=True.
    """
    examples = []
    for verse in verses[:max_verses] if max_verses else verses:
        words = _aggregate_segments_to_words(verse["segments"])
        if not words:
            continue

        # Reconstruct diacritized text
        diac_text = " ".join(w["surface"] for w in words)
        bare_text, diac_ids = text_to_diac_labels(diac_text)

        # Word offsets in the bare text
        bare_words = bare_text.split()
        if len(bare_words) != len(words):
            # Misalignment — word count mismatch between diacritized and bare forms.
            # Usually caused by segments containing spaces. Skip.
            continue
        word_offsets = compute_word_offsets(bare_text)

        # Derive i'rab roles
        irab_ids = []
        for i, word in enumerate(words):
            prev = words[i - 1] if i > 0 else None
            role = qac_word_to_irab_role(word, prev)
            irab_ids.append(IRAB_TO_ID.get(role, IRAB_TO_ID["other"]))

        examples.append(MTLExample(
            bare_text=bare_text,
            diac_labels=diac_ids,
            mask_diac=True,
            word_offsets=word_offsets,
            irab_labels=irab_ids,
            mask_irab=True,
            err_labels=[0] * len(bare_text),
            mask_err=False,
            source="qac",
            sent_id=verse["ref"],
        ))
    return examples


def load_qac_examples(
    path: Path | str, max_verses: Optional[int] = None, download_if_missing: bool = True
) -> List[MTLExample]:
    """One-call helper: download (if needed), parse, convert to examples."""
    path = Path(path)
    if download_if_missing and not path.exists():
        download_qac(path)
    verses = parse_qac(path)
    return qac_verses_to_examples(verses, max_verses=max_verses)
