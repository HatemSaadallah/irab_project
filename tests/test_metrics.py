"""Tests for evaluation metrics."""

from irab_tashkeel.evaluation.metrics import (
    der, error_span_f1, irab_accuracy, irab_per_class_f1, wer_diac,
)


def test_der_identical_is_zero():
    gold = "ذَهَبَ الطَّالِبُ"
    assert der(gold, gold) == 0.0


def test_der_one_wrong_case():
    gold = "ذَهَبَ الطَّالِبُ"
    pred = "ذَهَبَ الطَّالِبَ"  # wrong case on last letter
    d = der(pred, gold)
    # Exactly 1 of N diacritic positions is wrong
    assert 0 < d < 1


def test_der_completely_wrong():
    gold = "ذَهَبَ الطَّالِبُ"
    pred = "ذهب الطالب"  # all diacritics missing
    d = der(pred, gold)
    assert d == 1.0  # 100% error (every diacritic-bearing position is wrong)


def test_wer_diac_identical():
    gold = "ذَهَبَ الطَّالِبُ"
    assert wer_diac(gold, gold) == 0.0


def test_wer_diac_one_wrong_word():
    gold = "ذَهَبَ الطَّالِبُ"
    pred = "ذَهَبَ الطَّالِبَ"  # second word differs
    assert wer_diac(pred, gold) == 0.5


def test_irab_accuracy_basic():
    assert irab_accuracy(["fiil", "faail"], ["fiil", "faail"]) == 1.0
    assert irab_accuracy(["fiil", "faail"], ["fiil", "mubtada"]) == 0.5
    assert irab_accuracy([], []) == 0.0


def test_irab_per_class_f1():
    pred = ["fiil", "fiil", "N_marfu", "N_mansub"]
    gold = ["fiil", "N_marfu", "N_marfu", "N_marfu"]
    result = irab_per_class_f1(pred, gold)
    # fiil: 1 TP, 1 FP, 0 FN → prec=0.5, rec=1.0
    assert abs(result["fiil"]["precision"] - 0.5) < 1e-6
    assert abs(result["fiil"]["recall"] - 1.0) < 1e-6
    # N_marfu: 1 TP, 0 FP, 2 FN → prec=1.0, rec=1/3
    assert abs(result["N_marfu"]["precision"] - 1.0) < 1e-6
    assert abs(result["N_marfu"]["recall"] - 1 / 3) < 1e-6


def test_error_span_f1_exact_match():
    pred = [{"start": 5, "end": 8, "type": "hamza"}]
    gold = [{"start": 5, "end": 8, "type": "hamza"}]
    result = error_span_f1(pred, gold)
    assert result["f1"] == 1.0
    assert result["tp"] == 1


def test_error_span_f1_wrong_type():
    pred = [{"start": 5, "end": 8, "type": "hamza"}]
    gold = [{"start": 5, "end": 8, "type": "taa"}]
    result = error_span_f1(pred, gold)
    assert result["f1"] == 0.0


def test_error_span_f1_partial_overlap_counts_as_miss():
    pred = [{"start": 5, "end": 9, "type": "hamza"}]   # 1 char longer
    gold = [{"start": 5, "end": 8, "type": "hamza"}]
    result = error_span_f1(pred, gold)
    # Spans don't match exactly → 0 TP
    assert result["tp"] == 0
    assert result["f1"] == 0.0
