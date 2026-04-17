"""Unified schema for multi-task examples.

All data sources (Tashkeela, QAC, I3rab, synthetic) map to `MTLExample`.
The three boolean masks indicate which heads can be supervised by each example.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MTLExample:
    """One multi-task learning example.

    Attributes:
        bare_text: The undiacritized input (the model's input).
        diac_labels: Per-character diacritic class IDs (length = len(bare_text)).
        mask_diac: If True, train the diacritization head on this example.
        word_offsets: List of (start_char_idx, end_char_idx) for each word.
        irab_labels: Per-word i'rab class ID (length = len(word_offsets)).
        mask_irab: If True, train the i'rab head.
        err_labels: Per-character BIO error class (length = len(bare_text)).
        mask_err: If True, train the error detection head.
        source: Data source tag, e.g. "tashkeela", "qac", "i3rab", "synth_hamza".
        sent_id: Optional identifier (verse reference, sentence ID, etc).
    """

    bare_text: str
    diac_labels: List[int]
    mask_diac: bool
    word_offsets: List[Tuple[int, int]]
    irab_labels: List[int]
    mask_irab: bool
    err_labels: List[int]
    mask_err: bool
    source: str
    sent_id: Optional[str] = None

    def __post_init__(self):
        # Validate lengths to catch alignment bugs early.
        if self.mask_diac:
            assert len(self.diac_labels) == len(self.bare_text), (
                f"diac_labels length {len(self.diac_labels)} != text length "
                f"{len(self.bare_text)} for {self.sent_id}"
            )
        if self.mask_irab:
            assert len(self.irab_labels) == len(self.word_offsets), (
                f"irab_labels length {len(self.irab_labels)} != word count "
                f"{len(self.word_offsets)} for {self.sent_id}"
            )
        if self.mask_err:
            assert len(self.err_labels) == len(self.bare_text), (
                f"err_labels length {len(self.err_labels)} != text length "
                f"{len(self.bare_text)} for {self.sent_id}"
            )


@dataclass
class PredictionResult:
    """Output of Predictor.predict(). Model-agnostic (works for neural or rule)."""

    input_text: str
    diacritized: str
    words: List[dict]           # each: {surface, diacritized, role, role_ar, role_en, confidence, ...}
    errors: List[dict]          # each: {start, end, type, text, description}
    tier: int = 1               # classification tier (1/2/3) from the rule engine
    tier_flags: List[str] = field(default_factory=list)

    def to_json(self):
        return {
            "input": self.input_text,
            "diacritized": self.diacritized,
            "words": self.words,
            "errors": self.errors,
            "tier": self.tier,
            "tier_flags": self.tier_flags,
        }
