"""Tests for model components."""

import tempfile

import torch

from irab_tashkeel.models.encoder import CharTransformer, EncoderConfig
from irab_tashkeel.models.full_model import FullModel, ModelConfig
from irab_tashkeel.models.heads import DiacHead, ErrorHead, IrabHead
from irab_tashkeel.models.labels import DIAC_LABELS, ERR_LABELS, IRAB_LABELS, VOCAB_SIZE
from irab_tashkeel.models.tokenizer import (
    compute_word_offsets, decode_diacritized, encode_chars, normalize,
    strip_diacritics, text_to_diac_labels,
)


# ---------- Tokenizer ----------
def test_normalize_strips_tatweel():
    assert normalize("كتــاب") == "كتاب"


def test_strip_diacritics():
    assert strip_diacritics("كَتَبَ") == "كتب"


def test_text_to_diac_labels_roundtrip():
    original = "ذَهَبَ الطَّالِبُ"
    bare, ids = text_to_diac_labels(original)
    assert bare == "ذهب الطالب"
    assert len(ids) == len(bare)
    recon = decode_diacritized(bare, ids)
    assert recon == original


def test_compute_word_offsets():
    text = "ذهب الطالب إلى المدرسة"
    offsets = compute_word_offsets(text)
    assert len(offsets) == 4
    assert offsets[0] == (0, 3)
    assert text[offsets[1][0]:offsets[1][1]] == "الطالب"


def test_encode_chars_pads_correctly():
    ids, mask = encode_chars("ذهب", max_len=10)
    assert len(ids) == 10
    assert len(mask) == 10
    # First 3 positions real, rest padding
    assert sum(mask) == 3


# ---------- Encoder ----------
def test_char_transformer_forward_shape():
    config = EncoderConfig(vocab_size=VOCAB_SIZE, hidden=64, n_heads=4, n_layers=2, max_len=128)
    encoder = CharTransformer(config).eval()
    ids = torch.randint(0, VOCAB_SIZE, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.long)
    with torch.no_grad():
        out = encoder(ids, mask)
    assert out.shape == (2, 20, 64)


# ---------- Heads ----------
def test_diac_head_shape():
    head = DiacHead(hidden=64, n_classes=15).eval()
    h = torch.randn(3, 20, 64)
    with torch.no_grad():
        logits = head(h)
    assert logits.shape == (3, 20, 15)


def test_irab_head_shape_and_mask():
    head = IrabHead(hidden=64, n_classes=11).eval()
    h = torch.randn(2, 30, 64)
    word_offsets = [[(0, 3), (4, 10)], [(0, 5), (6, 15), (16, 25)]]
    with torch.no_grad():
        logits, mask = head(h, word_offsets)
    assert logits.shape == (2, 3, 11)   # max 3 words in batch
    assert mask.shape == (2, 3)
    # First sample has 2 words, so mask[0, 2] should be 0
    assert mask[0, 0].item() == 1
    assert mask[0, 1].item() == 1
    assert mask[0, 2].item() == 0


def test_error_head_shape():
    head = ErrorHead(hidden=64, n_classes=7).eval()
    h = torch.randn(2, 15, 64)
    with torch.no_grad():
        logits = head(h)
    assert logits.shape == (2, 15, 7)


# ---------- Full model ----------
def _tiny_model():
    cfg = ModelConfig(
        encoder=EncoderConfig(vocab_size=VOCAB_SIZE, hidden=64, n_heads=4, n_layers=2, max_len=128),
        n_diac=len(DIAC_LABELS), n_irab=len(IRAB_LABELS), n_err=len(ERR_LABELS),
    )
    return FullModel(cfg)


def test_full_model_forward_shapes():
    model = _tiny_model().eval()
    ids = torch.randint(0, VOCAB_SIZE, (2, 30))
    mask = torch.ones(2, 30, dtype=torch.long)
    offsets = [[(0, 3), (4, 10)], [(0, 5), (6, 15)]]
    with torch.no_grad():
        out = model(ids, mask, offsets)
    assert out["diac"].shape == (2, 30, len(DIAC_LABELS))
    assert out["irab"].shape == (2, 2, len(IRAB_LABELS))
    assert out["err"].shape == (2, 30, len(ERR_LABELS))


def test_full_model_save_load_roundtrip():
    model = _tiny_model().eval()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    model.save(path)
    reloaded = FullModel.load(path, map_location="cpu").eval()

    # Weights should be identical
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2)

    # Config preserved
    assert reloaded.n_params() == model.n_params()


def test_full_model_param_count_reasonable():
    cfg = ModelConfig(
        encoder=EncoderConfig(vocab_size=VOCAB_SIZE, hidden=256, n_heads=8, n_layers=6, max_len=256),
        n_diac=len(DIAC_LABELS), n_irab=len(IRAB_LABELS), n_err=len(ERR_LABELS),
    )
    model = FullModel(cfg)
    # Should be in the millions, not thousands or billions
    n = model.n_params()
    assert 1_000_000 < n < 20_000_000
