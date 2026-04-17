"""Sentence-tier classifier — detects Tier 2 / Tier 3 constructions.

Used by the inference pipeline to decide when to fall back to neural-only
(Tier 2) vs skip the rule engine entirely (Tier 3).

See docs/ARCHITECTURE.md §"The hybrid path" for the tier definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# Tier-2 markers: rule engine detects them, but case assignment changes are
# hard to model; fall back to neural for affected words.
KANA_FAMILY = {
    "كان", "كانت", "يكون", "تكون", "ليس", "ليست",
    "أصبح", "أصبحت", "صار", "صارت", "يصير",
    "أمسى", "أمست", "ظل", "ظلت", "بات", "باتت",
}
INNA_FAMILY = {"إن", "أن", "كأن", "لكن", "لعل", "ليت"}
JUSSIVE_PARTICLES = {"لم", "لما", "لا"}   # لا only jussive in imperative context

# Tier-3 markers: rule engine gives up entirely; neural handles the whole sentence.
RELATIVE_PRONOUNS = {"الذي", "التي", "الذين", "اللاتي", "اللواتي", "من", "ما", "أي"}
# NB: من, ما, أي are also interrogatives/common particles — expect false positives.
CONDITIONAL_PARTICLES = {"إذا", "إن", "لو", "لولا", "لوما", "كلما", "متى", "أين"}
# We exclude إن here since it overlaps with the inna family — ambiguous without parsing.


@dataclass
class TierResult:
    tier: int                   # 1, 2, or 3
    flags: List[str]            # diagnostic strings like "kana:كان"


def classify_tier(tokens: List[str]) -> TierResult:
    """Classify a tokenized sentence into tier 1/2/3."""
    flags: List[str] = []
    tier = 1
    token_set = set(tokens)

    # --- Tier 3 (most restrictive) ---
    for rel in RELATIVE_PRONOUNS & token_set:
        flags.append(f"relative:{rel}")
        tier = 3
    for cond in CONDITIONAL_PARTICLES & token_set:
        if cond not in INNA_FAMILY:
            flags.append(f"conditional:{cond}")
            tier = 3

    # --- Tier 2 ---
    for k in KANA_FAMILY & token_set:
        flags.append(f"kana:{k}")
        tier = max(tier, 2)
    for inna in INNA_FAMILY & token_set:
        flags.append(f"inna:{inna}")
        tier = max(tier, 2)
    for j in JUSSIVE_PARTICLES & token_set:
        flags.append(f"jussive:{j}")
        tier = max(tier, 2)

    return TierResult(tier=tier, flags=flags)
