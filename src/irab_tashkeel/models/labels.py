"""Label vocabularies for the three task heads.

Modifying these requires retraining — the model's output dimensions are tied
to the lengths of these lists.
"""

# ---------------------------------------------------------------------------
# Diacritic labels (per-character)
# ---------------------------------------------------------------------------
# The 8 basic Arabic diacritics plus their 7 shadda-combined variants.
# Index 0 = "no diacritic" (the baseline class).
FATHA       = "\u064E"
DAMMA       = "\u064F"
KASRA       = "\u0650"
SUKUN       = "\u0652"
FATHATAN    = "\u064B"
DAMMATAN    = "\u064C"
KASRATAN    = "\u064D"
SHADDA      = "\u0651"

DIAC_LABELS = [
    "",             # 0 = no diacritic
    FATHA,          # 1 = fatḥa ( َ )
    DAMMA,          # 2 = ḍamma ( ُ )
    KASRA,          # 3 = kasra ( ِ )
    FATHATAN,       # 4 = tanwīn fatḥ ( ً )
    DAMMATAN,       # 5 = tanwīn ḍamm ( ٌ )
    KASRATAN,       # 6 = tanwīn kasr ( ٍ )
    SUKUN,          # 7 = sukūn ( ْ )
    FATHA + SHADDA,     # 8 = shadda + fatḥa
    DAMMA + SHADDA,     # 9
    KASRA + SHADDA,     # 10
    FATHATAN + SHADDA,  # 11
    DAMMATAN + SHADDA,  # 12
    KASRATAN + SHADDA,  # 13
    SUKUN + SHADDA,     # 14  (rare; valid for gemination + no-vowel)
]
DIAC_TO_ID = {d: i for i, d in enumerate(DIAC_LABELS)}

# Some corpora spell compound diacritics in shadda-first order. Canonicalize.
def canonicalize_diac(s: str) -> str:
    """Ensure diacritics are in vowel-then-shadda order.

    Some input has `ّ + َ` (shadda then fatha). We normalize to `َ + ّ`.
    """
    if SHADDA in s and len(s) == 2 and s[0] == SHADDA:
        return s[1] + s[0]
    return s


# ---------------------------------------------------------------------------
# I'rab role labels (per-word)
# ---------------------------------------------------------------------------
IRAB_LABELS = [
    "other",         # 0 = default / unclassified
    "fiil",          # 1 = verb (فعل)
    "harf_jarr",     # 2 = preposition (حرف جر)
    "harf_atf",      # 3 = coordinator (حرف عطف)
    "harf_nafy",     # 4 = negation particle (حرف نفي)
    "mabni_noun",    # 5 = indeclinable noun (pronouns, demonstratives, relatives)
    "N_marfu",       # 6 = noun in nominative (fāʿil, mubtadaʾ, khabar collapsed)
    "N_mansub",      # 7 = noun in accusative (mafʿūl bih, ḥāl, khabar-kāna)
    "ism_majrur",    # 8 = noun in genitive after preposition
    "mudaf_ilayh",   # 9 = noun in genitive as iḍāfa dependent
    "<pad>",         # 10 = padding
]
IRAB_TO_ID = {r: i for i, r in enumerate(IRAB_LABELS)}
IRAB_PAD_ID = IRAB_TO_ID["<pad>"]


# ---------------------------------------------------------------------------
# Error detection labels (per-character, BIO tagging)
# ---------------------------------------------------------------------------
ERR_LABELS = [
    "O",         # 0 = no error
    "B-hamza",   # 1 = beginning of hamza-qaṭʿ error (e.g., الى → إلى)
    "I-hamza",   # 2 = inside hamza error
    "B-taa",     # 3 = beginning of tāʾ marbūṭa error (e.g., جيده → جيدة)
    "I-taa",     # 4
    "B-case",    # 5 = beginning of case ending error
    "I-case",    # 6
]
ERR_TO_ID = {e: i for i, e in enumerate(ERR_LABELS)}


# ---------------------------------------------------------------------------
# Character vocabulary
# ---------------------------------------------------------------------------
# Arabic letters + diacritics + hamza variants + standard punctuation + pad/unk.
# We keep diacritics in the char vocab so the model can process already-
# diacritized input (partial diacritization is common in real-world text).

ARABIC_LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويءؤئىةأإآ")
ARABIC_DIACRITICS = list("ًٌٍَُِّْ")
DAGGER_ALIF = "\u0670"  # ـٰ (superscript alif)
ALEF_WASLA = "\u0671"   # ٱ
ARABIC_EXTRAS = [DAGGER_ALIF, ALEF_WASLA]
SPECIALS = ["<pad>", "<unk>"]
WHITESPACE = [" "]
# Some Quranic texts have these; we include them to avoid <unk> spam
PUNCTUATION = [".", "،", "؟", "!", "؛", ":", "-"]

CHAR_VOCAB = (
    SPECIALS
    + WHITESPACE
    + sorted(set(ARABIC_LETTERS + ARABIC_DIACRITICS + ARABIC_EXTRAS))
    + PUNCTUATION
)
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}
PAD_ID = CHAR_TO_ID["<pad>"]
UNK_ID = CHAR_TO_ID["<unk>"]
VOCAB_SIZE = len(CHAR_VOCAB)
