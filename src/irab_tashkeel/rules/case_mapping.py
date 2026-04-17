"""Map (case, state, number, POS) → final diacritic + bilingual explanation.

Used by the rule-based engine when it has high confidence about a word's
grammatical role. The mapping implements standard Arabic case-ending rules
with a handful of special cases (diptotes, dual, sound plurals).
"""

from __future__ import annotations

import re
from typing import Tuple

from ..models.labels import (
    ALEF_WASLA, DAMMA, DAMMATAN, FATHA, FATHATAN, KASRA, KASRATAN, SUKUN,
)

ALIF = "\u0627"
TA_MARBUTA = "\u0629"


def _strip_final_diac(s: str) -> str:
    return re.sub(r"[\u064B-\u0652]+$", "", s)


def apply_case_ending(
    stem_diac: str,
    case: str,       # "marfu" | "mansub" | "majrur" | "majzum" | "mabni" | "na"
    state: str = "i",  # "d"(def) | "i"(indef) | "c"(construct) | "na"
    pos: str = "N",  # "N" | "V" | "ADJ" | "P" | "PART" | ...
    is_diptote: bool = False,
) -> Tuple[str, str]:
    """Apply the appropriate final diacritic.

    Returns (final_form, arabic_explanation).
    """
    # Verbs
    if pos == "V":
        if case == "mabni":
            return stem_diac, "فعل مبني"
        base = _strip_final_diac(stem_diac)
        if case == "marfu":
            return base + DAMMA, "فعل مضارع مرفوع بالضمة"
        if case == "mansub":
            return base + FATHA, "فعل مضارع منصوب بالفتحة"
        if case == "majzum":
            return base + SUKUN, "فعل مضارع مجزوم بالسكون"
        return stem_diac, ""

    # Particles and mabnī categories
    if pos in {"P", "PART", "PRON", "CONJ"}:
        return stem_diac, "مبني لا محل له من الإعراب"

    # Nouns / adjectives
    base = _strip_final_diac(stem_diac)
    bare = re.sub(r"[\u064B-\u0652]+", "", stem_diac)

    # Sound plurals (trust the stem if it already has them)
    if bare.endswith("ون") or bare.endswith("ين"):
        return stem_diac, f"جمع مذكر سالم — {_case_name_ar(case)}"
    # Dual
    if bare.endswith("ان") and case == "marfu":
        return stem_diac, "مثنى مرفوع بالألف"
    if bare.endswith("ين") and case in {"mansub", "majrur"}:
        return stem_diac, f"مثنى {_case_name_ar(case)} بالياء"

    definite_or_construct = state in {"d", "c"}

    if case == "marfu":
        marker = DAMMA if definite_or_construct else DAMMATAN
        return base + marker, f"مرفوع بـ{_marker_name_ar(marker)}"
    if case == "mansub":
        if is_diptote:
            return base + FATHA, "منصوب بالفتحة (ممنوع من الصرف)"
        if bare.endswith(TA_MARBUTA) or bare.endswith(ALIF) or bare.endswith("ء"):
            return base + FATHATAN, "منصوب بالفتحتين (بلا ألف)"
        if definite_or_construct:
            return base + FATHA, "منصوب بالفتحة"
        return base + FATHATAN + ALIF, "منصوب بالفتحتين مع ألف التنوين"
    if case == "majrur":
        if is_diptote:
            return base + FATHA, "مجرور بالفتحة نيابة عن الكسرة (ممنوع من الصرف)"
        marker = KASRA if definite_or_construct else KASRATAN
        return base + marker, f"مجرور بـ{_marker_name_ar(marker)}"
    if case == "mabni":
        return stem_diac, "مبني"
    return stem_diac, ""


def _case_name_ar(case: str) -> str:
    return {"marfu": "مرفوع", "mansub": "منصوب", "majrur": "مجرور",
            "majzum": "مجزوم", "mabni": "مبني", "na": ""}.get(case, case)


def _marker_name_ar(m: str) -> str:
    return {
        DAMMA: "الضمة", FATHA: "الفتحة", KASRA: "الكسرة", SUKUN: "السكون",
        DAMMATAN: "تنوين الضم", FATHATAN: "تنوين الفتح", KASRATAN: "تنوين الكسر",
    }.get(m, m)
