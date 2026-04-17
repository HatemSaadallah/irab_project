"""Tests for data loaders and synthetic error injection."""

import random

from irab_tashkeel.data.i3rab import DEMO_I3RAB, load_i3rab_examples, parse_i3rab
from irab_tashkeel.data.qac import qac_word_to_irab_role
from irab_tashkeel.data.schema import MTLExample
from irab_tashkeel.data.synthetic import (
    corrupt_to_example, generate_synthetic_examples,
    inject_hamza_drop, inject_taa_marbuta_swap, inject_case_swap,
)
from irab_tashkeel.data.tashkeela import sentences_to_examples
from irab_tashkeel.models.labels import ERR_TO_ID, IRAB_TO_ID


# ---------- I3rab ----------
def test_parse_i3rab_demo():
    sentences = parse_i3rab(DEMO_I3RAB)
    assert len(sentences) == 3
    assert sentences[0]["sent_id"] == "i3rab_demo_1"
    assert len(sentences[0]["tokens"]) == 5
    assert sentences[0]["tokens"][0]["word"] == "يقرأ"
    assert sentences[0]["tokens"][0]["irab_role"] == "fiil"


def test_load_i3rab_examples_demo():
    examples = load_i3rab_examples()
    assert len(examples) == 3
    for ex in examples:
        assert ex.source == "i3rab"
        assert ex.mask_irab is True
        assert ex.mask_diac is False
        assert len(ex.irab_labels) == len(ex.word_offsets)


# ---------- QAC mapper ----------
def test_qac_word_to_irab_role_verb():
    word = {"tag": "V", "features": {"TENSE": "PERF"}}
    assert qac_word_to_irab_role(word) == "fiil"


def test_qac_word_to_irab_role_nom_noun():
    word = {"tag": "N", "features": {"CASE": "NOM"}}
    assert qac_word_to_irab_role(word) == "N_marfu"


def test_qac_word_to_irab_role_gen_after_prep():
    word = {"tag": "N", "features": {"CASE": "GEN"}}
    prev = {"tag": "P", "features": {}}
    assert qac_word_to_irab_role(word, prev) == "ism_majrur"


def test_qac_word_to_irab_role_gen_idafa():
    word = {"tag": "N", "features": {"CASE": "GEN"}}
    prev = {"tag": "N", "features": {"CASE": "NOM"}}
    assert qac_word_to_irab_role(word, prev) == "mudaf_ilayh"


# ---------- Synthetic errors ----------
def test_inject_hamza_drop_on_valid_word():
    rng = random.Random(42)
    result = inject_hamza_drop("ذَهَبَ إِلَى المَدْرَسَةِ", rng)
    assert result is not None
    corrupted, start, end = result
    # "إلى" should become "الى"
    assert "إ" not in corrupted
    assert start < end
    assert end <= len(corrupted)


def test_inject_hamza_drop_no_candidates():
    rng = random.Random(42)
    result = inject_hamza_drop("كتب الولد", rng)
    assert result is None


def test_inject_taa_marbuta_swap():
    rng = random.Random(42)
    result = inject_taa_marbuta_swap("كَتَبَ الطَّالِبَةِ", rng)
    assert result is not None
    corrupted, _, _ = result
    assert "ة" not in corrupted
    assert "ه" in corrupted


def test_inject_case_swap():
    rng = random.Random(42)
    gold = "الكِتَابُ جَمِيلٌ"
    result = inject_case_swap(gold, rng)
    assert result is not None
    corrupted, start, end = result
    # The corrupted text should differ from gold
    assert corrupted != gold


def test_corrupt_to_example_produces_valid_mtl():
    rng = random.Random(42)
    ex = corrupt_to_example("ذَهَبَ الطَّالِبُ إِلَى المَدْرَسَةِ", "hamza_drop", rng, "t1")
    assert ex is not None
    assert isinstance(ex, MTLExample)
    assert ex.mask_err is True
    assert ex.mask_diac is False
    assert ex.mask_irab is False
    # At least one non-O error label
    assert any(label != ERR_TO_ID["O"] for label in ex.err_labels)


def test_generate_synthetic_examples_count():
    gold = ["ذَهَبَ الطَّالِبُ إِلَى المَدْرَسَةِ",
            "قَرَأَ الوَلَدُ الكِتَابَ الجَدِيدَ"]
    examples = generate_synthetic_examples(gold, per_type=3, seed=42)
    # Up to 9 total (3 × 3 types), but could be less if some injections fail
    assert 0 < len(examples) <= 9
    sources = {ex.source for ex in examples}
    # At least two types should succeed
    assert len(sources) >= 2


# ---------- Tashkeela helpers ----------
def test_sentences_to_examples():
    sents = ["ذَهَبَ الطَّالِبُ", "الحَمْدُ للهِ"]
    examples = sentences_to_examples(sents)
    assert len(examples) == 2
    for ex in examples:
        assert ex.mask_diac is True
        assert ex.mask_irab is False
        assert ex.source == "tashkeela"
        assert len(ex.diac_labels) == len(ex.bare_text)
