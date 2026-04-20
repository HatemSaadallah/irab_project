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


def _extract_case(feats: Dict) -> str:
    """QAC stores case as either CASE:NOM or as a bare flag NOM/ACC/GEN. Normalize."""
    case = feats.get("CASE", "")
    if case:
        return case
    for c in ("NOM", "ACC", "GEN"):
        if c in feats:
            return c
    return ""


def _extract_mood(feats: Dict) -> str:
    mood = feats.get("MOOD", "")
    if mood:
        return mood
    for m in ("IND", "SUB", "JUS"):
        if m in feats:
            return m
    return ""


def qac_word_to_irab_role(word: Dict, prev: Optional[Dict] = None) -> str:
    """Derive a coarse i'rab role from a QAC word.

    Uses POS + case + immediate context. See docs/DATA.md for the full mapping.
    """
    tag = word["tag"]
    case = _extract_case(word["features"])

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


# Map QAC CASE values to the Arabic role phrase used in "ضمير مبني في محل ..."
_CASE_TO_MAHALL = {
    "NOM": "رفع",
    "ACC": "نصب",
    "GEN": "جر",
}


def qac_word_to_irab_string(word: Dict, prev: Optional[Dict] = None) -> str:
    """Render a full traditional Arabic i'rab string for a QAC word.

    Deterministic templater driven by POS tag + CASE / MOOD / particle subtype.
    Covers the high-frequency patterns; falls back to a generic phrase for the
    long tail. Intended as the per-word seq2seq target for QAC examples.
    """
    tag = word["tag"]
    feats = word["features"]
    case = _extract_case(feats)

    # --- Verbs ---
    if tag == "V":
        if "PERF" in feats:
            return "فعل ماضٍ مبني على الفتح"
        if "IMPV" in feats:
            return "فعل أمر مبني على السكون"
        if "IMPF" in feats:
            mood = _extract_mood(feats)
            if mood == "IND":
                return "فعل مضارع مرفوع وعلامة رفعه الضمة الظاهرة"
            if mood == "SUB":
                return "فعل مضارع منصوب وعلامة نصبه الفتحة الظاهرة"
            if mood == "JUS":
                return "فعل مضارع مجزوم وعلامة جزمه السكون"
            return "فعل مضارع مرفوع وعلامة رفعه الضمة الظاهرة"
        return "فعل"
    if tag == "IMPV":
        return "فعل أمر مبني على السكون"

    # --- Particles (tag == 'P' in QAC carries the subtype as a flag feature) ---
    if tag == "P":
        if "DET" in feats:
            return "أداة تعريف مبنية على السكون لا محل لها من الإعراب"
        if "CONJ" in feats:
            return "حرف عطف مبني لا محل له من الإعراب"
        if "NEG" in feats:
            return "حرف نفي مبني لا محل له من الإعراب"
        if "INTG" in feats:
            return "حرف استفهام مبني لا محل له من الإعراب"
        if "EMPH" in feats:
            return "حرف توكيد مبني لا محل له من الإعراب"
        if "VOC" in feats:
            return "حرف نداء مبني لا محل له من الإعراب"
        if "FUT" in feats:
            return "حرف استقبال مبني لا محل له من الإعراب"
        if "COND" in feats:
            return "حرف شرط مبني لا محل له من الإعراب"
        if "EXP" in feats:
            return "حرف استثناء مبني لا محل له من الإعراب"
        if "P" in feats:
            return "حرف جر مبني لا محل له من الإعراب"
        return "حرف مبني لا محل له من الإعراب"

    if tag == "CONJ":
        return "حرف عطف مبني لا محل له من الإعراب"
    if tag == "SUB":
        return "حرف مصدري مبني لا محل له من الإعراب"
    if tag in {"CIRC", "REM"}:
        return "حرف عطف مبني لا محل له من الإعراب"
    if tag == "NEG":
        return "حرف نفي مبني لا محل له من الإعراب"
    if tag == "INTG":
        return "حرف استفهام مبني لا محل له من الإعراب"
    if tag == "EMPH":
        return "حرف توكيد مبني لا محل له من الإعراب"
    if tag == "VOC":
        return "حرف نداء مبني لا محل له من الإعراب"

    # --- Pronouns / demonstratives / relatives (mabni nouns) ---
    if tag in {"PRON", "PRO"}:
        mahall = _CASE_TO_MAHALL.get(case, "رفع")
        return f"ضمير مبني في محل {mahall}"
    if tag == "DEM":
        mahall = _CASE_TO_MAHALL.get(case, "رفع")
        return f"اسم إشارة مبني في محل {mahall}"
    if tag == "REL":
        mahall = _CASE_TO_MAHALL.get(case, "رفع")
        return f"اسم موصول مبني في محل {mahall}"

    # --- Nouns / proper nouns / adjectives ---
    if tag in {"N", "PN", "IMPN", "ADJ"}:
        if tag == "PN" or "PN" in feats:
            prefix = "اسم علم"
        elif tag == "ADJ" or "ADJ" in feats:
            prefix = "صفة"
        else:
            prefix = "اسم"

        if case == "NOM":
            return f"{prefix} مرفوع وعلامة رفعه الضمة الظاهرة"
        if case == "ACC":
            return f"{prefix} منصوب وعلامة نصبه الفتحة الظاهرة"
        if case == "GEN":
            # In QAC many prepositions appear as PREFIX segments on the same
            # word as the noun. Check this word's own segments first, then
            # fall back to the previous word's tail segment.
            has_prep_prefix = False
            for seg in word.get("all_segments", []):
                seg_feats = seg.get("features", {})
                if seg["tag"] == "P" and "P" in seg_feats and "PREF" in seg_feats:
                    has_prep_prefix = True
                    break
            if not has_prep_prefix and prev is not None:
                prev_tag = prev.get("tag")
                prev_feats = prev.get("features", {})
                if prev_tag == "P" and "P" in prev_feats:
                    has_prep_prefix = True
            if has_prep_prefix:
                return f"{prefix} مجرور بحرف الجر وعلامة جره الكسرة الظاهرة"
            return f"{prefix} مجرور وعلامة جره الكسرة الظاهرة (مضاف إليه)"
        return prefix

    return "كلمة"


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

        # Derive i'rab roles (coarse class) AND detailed Arabic strings
        irab_ids = []
        irab_targets = []
        for i, word in enumerate(words):
            prev = words[i - 1] if i > 0 else None
            role = qac_word_to_irab_role(word, prev)
            irab_ids.append(IRAB_TO_ID.get(role, IRAB_TO_ID["other"]))
            irab_targets.append(qac_word_to_irab_string(word, prev))

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
            irab_targets=irab_targets,
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
