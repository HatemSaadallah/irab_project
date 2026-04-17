"""Rule-based orthographic corrections.

Deterministic fixes that precede the neural model. Currently handles:
- Hamza qaṭʿ on word-initial alif (الى → إلى)
- Tāʾ marbūṭa mistakes (جيده → جيدة)
- Tatweel removal

These are NOT sufficient to catch all orthographic errors — they're a
targeted set based on common mistakes Arabic learners make.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Whitelist of common mis-hamza'd words. We deliberately EXCLUDE ambiguous
# cases (like ان, which could be أَنْ or إِنَّ).
HAMZA_FIXES = {
    "الى": "إلى",
    "اذا": "إذا",
    "اليك": "إليك", "الينا": "إلينا", "اليكم": "إليكم", "اليه": "إليه", "اليها": "إليها",
    "اولاد": "أولاد", "اولى": "أولى", "اوائل": "أوائل",
    "اسرة": "أسرة", "اسر": "أسر",
    "اكثر": "أكثر", "اقل": "أقل", "اكبر": "أكبر", "اصغر": "أصغر",
    "اصبح": "أصبح", "اصبحت": "أصبحت", "امسى": "أمسى",
    "احمد": "أحمد", "احد": "أحد",
    "ارض": "أرض", "اسماء": "أسماء",
    "اعلى": "أعلى", "اسفل": "أسفل",
    "ان": None,       # explicitly mark as ambiguous
    "اذ": None,       # likewise
}

# Common tāʾ-marbūṭa vs hāʾ confusions
TAA_MARBUTA_FIXES = {
    "جيده": "جيدة", "جميله": "جميلة", "قصيره": "قصيرة", "طويله": "طويلة",
    "كبيره": "كبيرة", "صغيره": "صغيرة", "واضحه": "واضحة", "مفيده": "مفيدة",
    "مدرسه": "مدرسة", "مكتبه": "مكتبة", "سياره": "سيارة", "فكره": "فكرة",
    "شجره": "شجرة", "زهره": "زهرة", "قصه": "قصة", "لعبه": "لعبة",
    "قريه": "قرية", "مدينه": "مدينة", "دوله": "دولة", "قاعه": "قاعة",
    "غرفه": "غرفة", "ساعه": "ساعة", "لحظه": "لحظة", "عائله": "عائلة",
}

TATWEEL = "\u0640"


@dataclass
class OrthographicCorrection:
    position: int       # word index in the output tokens list
    original: str
    corrected: str
    type: str           # "hamza_qate" | "taa_marbuta"
    explanation_ar: str
    explanation_en: str


@dataclass
class OrthographicResult:
    original: str
    corrected: str
    tokens: List[str]
    corrections: List[OrthographicCorrection]


def orthographic_correct(text: str) -> OrthographicResult:
    """Apply deterministic orthographic corrections.

    Returns the corrected text plus a per-correction explanation trail.
    """
    # NFC + tatweel strip
    text = unicodedata.normalize("NFC", text).replace(TATWEEL, "")

    tokens = text.split()
    corrected = list(tokens)
    corrections: List[OrthographicCorrection] = []

    for i, tok in enumerate(tokens):
        # Hamza
        if tok in HAMZA_FIXES:
            fix = HAMZA_FIXES[tok]
            if fix is not None:
                corrected[i] = fix
                corrections.append(OrthographicCorrection(
                    position=i, original=tok, corrected=fix, type="hamza_qate",
                    explanation_ar=f"«{tok}» تُكتب بهمزة القطع: «{fix}»",
                    explanation_en=f"'{tok}' should carry hamza qaṭʿ: '{fix}'",
                ))
                continue

        # Tāʾ marbūṭa
        if tok in TAA_MARBUTA_FIXES:
            fix = TAA_MARBUTA_FIXES[tok]
            corrected[i] = fix
            corrections.append(OrthographicCorrection(
                position=i, original=tok, corrected=fix, type="taa_marbuta",
                explanation_ar=f"«{tok}» بالتاء المربوطة: «{fix}»",
                explanation_en=f"'{tok}' uses tāʾ marbūṭa: '{fix}'",
            ))

    return OrthographicResult(
        original=text,
        corrected=" ".join(corrected),
        tokens=corrected,
        corrections=corrections,
    )
