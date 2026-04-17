"""Tests for the inference pipeline."""

import torch

from irab_tashkeel.data.schema import PredictionResult
from irab_tashkeel.inference.predictor import Predictor
from irab_tashkeel.models.encoder import EncoderConfig
from irab_tashkeel.models.full_model import FullModel, ModelConfig
from irab_tashkeel.models.labels import DIAC_LABELS, ERR_LABELS, IRAB_LABELS, VOCAB_SIZE
from irab_tashkeel.rules.orthographic import orthographic_correct
from irab_tashkeel.rules.tiers import classify_tier


# ---------- Rules ----------
def test_orthographic_fixes_hamza():
    result = orthographic_correct("ذهب الولد الى البيت")
    assert "إلى" in result.tokens
    assert len(result.corrections) == 1
    assert result.corrections[0].type == "hamza_qate"


def test_orthographic_fixes_taa_marbuta():
    result = orthographic_correct("هذه فكره جيده")
    assert "فكرة" in result.tokens
    assert "جيدة" in result.tokens
    assert len(result.corrections) == 2


def test_orthographic_no_fixes_on_clean_text():
    result = orthographic_correct("كتب الولد الدرس")
    assert result.corrections == []


def test_orthographic_ambiguous_not_fixed():
    # "ان" is ambiguous (أن vs إن) — should NOT be auto-fixed
    result = orthographic_correct("قال ان الجو بارد")
    # "ان" should remain as-is
    assert "ان" in result.tokens
    # No hamza correction for it
    types = [c.type for c in result.corrections]
    assert "hamza_qate" not in types or all(c.original != "ان" for c in result.corrections)


# ---------- Tier classifier ----------
def test_tier_1_simple_verbal():
    tier = classify_tier("ذهب الطالب إلى المدرسة".split())
    assert tier.tier == 1
    assert tier.flags == []


def test_tier_2_kana():
    tier = classify_tier("كان الطالب مجتهدا".split())
    assert tier.tier == 2
    assert any("kana" in f for f in tier.flags)


def test_tier_3_relative():
    tier = classify_tier("الذي يدرس ينجح".split())
    assert tier.tier == 3
    assert any("relative" in f for f in tier.flags)


def test_tier_3_conditional():
    tier = classify_tier("إذا اجتهدت نجحت".split())
    assert tier.tier == 3


# ---------- End-to-end (with random weights, so outputs are gibberish but shapes work) ----------
def _make_tiny_predictor() -> Predictor:
    cfg = ModelConfig(
        encoder=EncoderConfig(vocab_size=VOCAB_SIZE, hidden=64, n_heads=4, n_layers=2, max_len=128),
        n_diac=len(DIAC_LABELS), n_irab=len(IRAB_LABELS), n_err=len(ERR_LABELS),
    )
    model = FullModel(cfg).eval()
    return Predictor(model, device=torch.device("cpu"))


def test_predictor_returns_prediction_result():
    predictor = _make_tiny_predictor()
    result = predictor.predict("ذهب الطالب إلى المدرسة")
    assert isinstance(result, PredictionResult)
    assert result.input_text == "ذهب الطالب إلى المدرسة"
    assert isinstance(result.diacritized, str)
    assert len(result.words) == 4
    assert result.tier == 1


def test_predictor_handles_tier_2():
    predictor = _make_tiny_predictor()
    result = predictor.predict("كان الطالب مجتهدا")
    assert result.tier == 2
    assert any("kana" in f for f in result.tier_flags)


def test_predictor_words_have_required_fields():
    predictor = _make_tiny_predictor()
    result = predictor.predict("العلم نور")
    required = {"surface", "diacritized", "role", "role_ar", "role_en",
                "diac_confidence", "irab_confidence"}
    for w in result.words:
        assert required.issubset(w.keys())


def test_predictor_runs_orthographic_correction_first():
    predictor = _make_tiny_predictor()
    # The orthographic corrector should convert "الى" → "إلى" BEFORE the model sees it.
    # The corrected word shows up in the orthographic_correct output which the predictor
    # applies. We verify by stripping diacritics from the output and checking bare form.
    from irab_tashkeel.models.tokenizer import strip_diacritics
    result = predictor.predict("ذهب الولد الى البيت")
    bare_output = strip_diacritics(result.diacritized)
    # Corrected form should appear in the bare output
    assert "إلى" in bare_output, f"Expected 'إلى' in '{bare_output}'"
    # Also verify the correction was recorded
    hamza_corrections = [e for e in result.errors if e.get("type") == "hamza_qate"]
    assert len(hamza_corrections) >= 1


def test_predictor_handles_empty_string_gracefully():
    predictor = _make_tiny_predictor()
    result = predictor.predict("   ")
    # Should not crash; may return empty
    assert result.input_text.strip() == ""
