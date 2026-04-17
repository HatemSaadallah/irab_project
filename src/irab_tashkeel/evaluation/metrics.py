"""Evaluation metrics for diacritization and i'rab tasks."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from ..models.tokenizer import is_arabic_letter, is_diacritic


def _letter_diac_pairs(text: str) -> List[Tuple[str, str]]:
    """Split text into (letter, diacritic_string) pairs."""
    pairs = []
    i = 0
    while i < len(text):
        if is_arabic_letter(text[i]):
            letter = text[i]
            diacs = ""
            j = i + 1
            while j < len(text) and is_diacritic(text[j]):
                diacs += text[j]
                j += 1
            pairs.append((letter, diacs))
            i = j
        else:
            i += 1
    return pairs


def der(predicted: str, gold: str, include_case_ending: bool = True) -> float:
    """Diacritic Error Rate.

    Standard (Fadel et al. 2019) definition: percentage of characters where
    predicted diacritic != gold diacritic, counted only over chars where
    gold has a diacritic.

    If include_case_ending=False, we skip the final character of each
    whitespace-separated word (which is where case endings live).
    """
    p_pairs = _letter_diac_pairs(predicted)
    g_pairs = _letter_diac_pairs(gold)

    if len(p_pairs) != len(g_pairs):
        return 1.0

    if not include_case_ending:
        # Strip final letter of each word — tricky because we don't have spaces in pairs.
        # Use the gold string to find word boundaries, then map to pair indices.
        gold_word_ends = _word_end_indices(gold)
        pair_indices_to_skip = set(gold_word_ends)
    else:
        pair_indices_to_skip = set()

    err = tot = 0
    for i, ((_, pd), (_, gd)) in enumerate(zip(p_pairs, g_pairs)):
        if i in pair_indices_to_skip:
            continue
        if gd:
            tot += 1
            if pd != gd:
                err += 1
    return err / tot if tot > 0 else 0.0


def _word_end_indices(gold: str) -> List[int]:
    """Return the pair indices that correspond to word-final letters in gold."""
    ends = []
    pair_idx = -1
    for i, c in enumerate(gold):
        if is_arabic_letter(c):
            pair_idx += 1
        elif c.isspace() and pair_idx >= 0:
            ends.append(pair_idx)
    if pair_idx >= 0:
        ends.append(pair_idx)
    return ends


def wer_diac(predicted: str, gold: str) -> float:
    """Word Error Rate with diacritics: % of words where any diacritic differs."""
    p_words = predicted.split()
    g_words = gold.split()
    if len(p_words) != len(g_words):
        return 1.0
    wrong = sum(1 for p, g in zip(p_words, g_words) if p != g)
    return wrong / len(g_words) if g_words else 0.0


def irab_accuracy(predicted_roles: List[str], gold_roles: List[str]) -> float:
    """Simple per-word accuracy for i'rab labels."""
    if len(predicted_roles) != len(gold_roles):
        return 0.0
    if not gold_roles:
        return 0.0
    correct = sum(1 for p, g in zip(predicted_roles, gold_roles) if p == g)
    return correct / len(gold_roles)


def irab_per_class_f1(
    predicted_roles: List[str], gold_roles: List[str],
) -> Dict[str, Dict[str, float]]:
    """Precision/recall/F1 per i'rab class."""
    assert len(predicted_roles) == len(gold_roles)
    classes = set(predicted_roles) | set(gold_roles)
    out = {}
    for c in classes:
        tp = sum(1 for p, g in zip(predicted_roles, gold_roles) if p == c and g == c)
        fp = sum(1 for p, g in zip(predicted_roles, gold_roles) if p == c and g != c)
        fn = sum(1 for p, g in zip(predicted_roles, gold_roles) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def error_span_f1(
    predicted_spans: List[Dict], gold_spans: List[Dict],
) -> Dict[str, float]:
    """Exact span-match F1 for error detection.

    Spans match iff (start, end, type) all equal.
    """
    pred_set = {(s["start"], s["end"], s["type"]) for s in predicted_spans}
    gold_set = {(s["start"], s["end"], s["type"]) for s in gold_spans}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def summary(der_list: List[float], wer_list: List[float]) -> Dict[str, float]:
    """Aggregate a list of per-sentence metrics."""
    import statistics as st
    if not der_list:
        return {}
    return {
        "mean_der": st.mean(der_list),
        "median_der": st.median(der_list),
        "mean_wer": st.mean(wer_list) if wer_list else 0.0,
        "n": len(der_list),
        "catastrophic_rate": sum(1 for d in der_list if d > 0.5) / len(der_list),
    }
