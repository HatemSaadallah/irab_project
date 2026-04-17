"""I3rab Dependency Treebank parser.

Format (CoNLL-like, tab-separated):
    idx  word  lemma  POS  irab_role  head  case

Sentences separated by blank lines.
Comments (# sent_id = ..., # text = ...) are preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..models.labels import IRAB_TO_ID
from ..models.tokenizer import compute_word_offsets
from .schema import MTLExample


# Minimal demo data (3 sentences) shipped for development when the real
# treebank isn't available.
DEMO_I3RAB = """# sent_id = i3rab_demo_1
# text = يقرأ الطالب الكتاب في المكتبة
1\tيقرأ\tقرأ\tV\tfiil\t0\tmarfu
2\tالطالب\tطالب\tN\tN_marfu\t1\tmarfu
3\tالكتاب\tكتاب\tN\tN_mansub\t1\tmansub
4\tفي\tفي\tP\tharf_jarr\t1\tna
5\tالمكتبة\tمكتبة\tN\tism_majrur\t4\tmajrur

# sent_id = i3rab_demo_2
# text = العلم نور والجهل ظلام
1\tالعلم\tعلم\tN\tN_marfu\t0\tmarfu
2\tنور\tنور\tN\tN_marfu\t1\tmarfu
3\tو\tو\tCONJ\tharf_atf\t0\tna
4\tالجهل\tجهل\tN\tN_marfu\t3\tmarfu
5\tظلام\tظلام\tN\tN_marfu\t4\tmarfu

# sent_id = i3rab_demo_3
# text = كتاب الطالب جديد
1\tكتاب\tكتاب\tN\tN_marfu\t0\tmarfu
2\tالطالب\tطالب\tN\tmudaf_ilayh\t1\tmajrur
3\tجديد\tجديد\tN\tN_marfu\t1\tmarfu
"""


def parse_i3rab(content: str) -> List[Dict]:
    """Parse an I3rab-format string into sentence dicts.

    Returns list of dicts with keys: sent_id, text, tokens.
    Each token is a dict with idx, word, lemma, pos, irab_role, head, case.
    """
    sentences = []
    current_tokens: List[Dict] = []
    current_id: Optional[str] = None
    current_text: Optional[str] = None

    def flush():
        nonlocal current_tokens, current_id, current_text
        if current_tokens:
            sentences.append({
                "sent_id": current_id,
                "text": current_text or " ".join(t["word"] for t in current_tokens),
                "tokens": current_tokens,
            })
        current_tokens = []

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# sent_id"):
            flush()
            if "=" in stripped:
                current_id = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("# text"):
            if "=" in stripped:
                current_text = stripped.split("=", 1)[1].strip()
        elif stripped and not stripped.startswith("#"):
            parts = stripped.split("\t")
            if len(parts) >= 7:
                current_tokens.append({
                    "idx": int(parts[0]),
                    "word": parts[1],
                    "lemma": parts[2],
                    "pos": parts[3],
                    "irab_role": parts[4],
                    "head": int(parts[5]),
                    "case": parts[6],
                })
        elif not stripped and current_tokens:
            # Blank line = sentence separator
            flush()
    flush()
    return sentences


def i3rab_sentences_to_examples(sentences: List[Dict]) -> List[MTLExample]:
    """Convert parsed I3rab sentences to MTLExamples.

    I3rab labels i'rab roles but doesn't give us diacritized text directly.
    We set mask_diac=False and rely on QAC/Tashkeela for diacritization signal.
    """
    examples = []
    for sent in sentences:
        tokens = sent["tokens"]
        if not tokens:
            continue

        bare_text = " ".join(t["word"] for t in tokens)
        word_offsets = compute_word_offsets(bare_text)

        if len(word_offsets) != len(tokens):
            # Tokenization misalignment (e.g., word containing a space). Skip.
            continue

        irab_ids = [IRAB_TO_ID.get(t["irab_role"], IRAB_TO_ID["other"]) for t in tokens]

        examples.append(MTLExample(
            bare_text=bare_text,
            diac_labels=[0] * len(bare_text),   # no diacritics
            mask_diac=False,
            word_offsets=word_offsets,
            irab_labels=irab_ids,
            mask_irab=True,
            err_labels=[0] * len(bare_text),
            mask_err=False,
            source="i3rab",
            sent_id=sent.get("sent_id"),
        ))
    return examples


def load_i3rab_examples(path: Optional[Path | str] = None) -> List[MTLExample]:
    """Load I3rab examples from path, or use the demo data if no path given."""
    if path is None:
        return i3rab_sentences_to_examples(parse_i3rab(DEMO_I3RAB))

    path = Path(path)
    if not path.exists():
        import warnings
        warnings.warn(f"I3rab file not found at {path}, falling back to demo data.")
        return i3rab_sentences_to_examples(parse_i3rab(DEMO_I3RAB))

    content = path.read_text(encoding="utf-8")
    return i3rab_sentences_to_examples(parse_i3rab(content))
