"""Predictor — the main inference API.

Wraps the trained FullModel and exposes a clean .predict() method that:
1. Orthographically normalizes the input (rule-based)
2. Classifies the sentence tier (rule-based)
3. Runs the neural model
4. Produces a unified PredictionResult with per-word explanations
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..data.schema import PredictionResult
from ..models.full_model import FullModel
from ..models.irab_tokenizer import IrabTokenizer
from ..models.labels import (
    DIAC_LABELS, ERR_LABELS, IRAB_LABELS,
)
from ..models.tokenizer import (
    compute_word_offsets, encode_chars, is_arabic_letter, normalize,
)
from ..rules.orthographic import orthographic_correct
from ..rules.tiers import classify_tier
from .explanations import describe_error, role_to_ar, role_to_en


class Predictor:
    """High-level inference wrapper around FullModel."""

    def __init__(
        self,
        model: FullModel,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.6,
        irab_tokenizer: Optional[IrabTokenizer] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        self.max_len = model.config.encoder.max_len
        self.irab_tokenizer = irab_tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.6,
        irab_tokenizer_path: Optional[Union[str, Path]] = "data/irab_spm.model",
    ) -> "Predictor":
        """Load a Predictor from a model checkpoint.

        If `irab_tokenizer_path` exists and the model has an i'rab decoder,
        the tokenizer is loaded automatically.
        """
        model = FullModel.load(path, map_location="cpu")
        tok = None
        if model.irab_decoder is not None and irab_tokenizer_path:
            tok_path = Path(irab_tokenizer_path)
            if tok_path.exists():
                tok = IrabTokenizer.load(tok_path)
        return cls(
            model, device=device,
            confidence_threshold=confidence_threshold,
            irab_tokenizer=tok,
        )

    @torch.no_grad()
    def _run_model(self, bare_text: str) -> Dict:
        """Run the neural model on bare text; return raw predictions + confidences."""
        char_ids, mask = encode_chars(bare_text, max_len=self.max_len)
        char_ids_t = torch.tensor([char_ids], dtype=torch.long, device=self.device)
        mask_t = torch.tensor([mask], dtype=torch.long, device=self.device)
        word_offsets = [compute_word_offsets(bare_text)]

        out = self.model(char_ids_t, mask_t, word_offsets)

        diac_probs = torch.softmax(out["diac"][0], dim=-1)
        diac_pred = diac_probs.argmax(dim=-1).cpu().tolist()
        diac_conf = diac_probs.max(dim=-1).values.cpu().tolist()

        irab_probs = torch.softmax(out["irab"][0], dim=-1)
        irab_pred = irab_probs.argmax(dim=-1).cpu().tolist()
        irab_conf = irab_probs.max(dim=-1).values.cpu().tolist()

        err_probs = torch.softmax(out["err"][0], dim=-1)
        err_pred = err_probs.argmax(dim=-1).cpu().tolist()

        irab_texts: List[str] = []
        if self.model.irab_decoder is not None and self.irab_tokenizer is not None:
            generated = self.model.generate_irab(char_ids_t, mask_t, word_offsets)
            for token_ids in generated[0]:
                irab_texts.append(self.irab_tokenizer.decode(token_ids))

        return {
            "diac_pred": diac_pred, "diac_conf": diac_conf,
            "irab_pred": irab_pred, "irab_conf": irab_conf,
            "irab_texts": irab_texts,
            "err_pred": err_pred,
            "word_offsets": word_offsets[0],
        }

    def _reconstruct_diacritized(self, bare_text: str, diac_pred: List[int]) -> str:
        """Insert predicted diacritics into the bare text."""
        out = []
        for i, c in enumerate(bare_text):
            out.append(c)
            if i < len(diac_pred) and is_arabic_letter(c):
                cls = diac_pred[i]
                if 0 < cls < len(DIAC_LABELS):
                    out.append(DIAC_LABELS[cls])
        return "".join(out)

    def _extract_error_spans(self, bare_text: str, err_pred: List[int]) -> List[Dict]:
        """Turn BIO-tagged error predictions into span dicts."""
        spans: List[Dict] = []
        in_span = False
        start = 0
        etype = ""
        for i, cls in enumerate(err_pred[: len(bare_text)]):
            label = ERR_LABELS[cls]
            if label.startswith("B-"):
                if in_span:
                    spans.append({
                        "start": start, "end": i, "type": etype,
                        "text": bare_text[start:i], "description": describe_error(etype),
                    })
                in_span = True
                start = i
                etype = label[2:]
            elif label == "O" and in_span:
                spans.append({
                    "start": start, "end": i, "type": etype,
                    "text": bare_text[start:i], "description": describe_error(etype),
                })
                in_span = False
        if in_span:
            spans.append({
                "start": start, "end": len(err_pred), "type": etype,
                "text": bare_text[start : len(err_pred)], "description": describe_error(etype),
            })
        return spans

    def predict(self, text: str) -> PredictionResult:
        """Main entry point. Normalizes, runs the model, returns a rich result."""
        text = normalize(text)

        # Step 1: orthographic correction
        ortho = orthographic_correct(text)
        bare_text = ortho.corrected

        # Step 2: tier classification
        tier_result = classify_tier(bare_text.split())

        # Step 3: neural inference
        pred = self._run_model(bare_text)

        # Step 4: reconstruct outputs
        diacritized = self._reconstruct_diacritized(bare_text, pred["diac_pred"])
        err_spans = self._extract_error_spans(bare_text, pred["err_pred"])

        # Fold orthographic corrections into the error list (they're a form of error too)
        for c in ortho.corrections:
            err_spans.append({
                "start": -1, "end": -1,   # unknown char span; we have word index
                "type": c.type,
                "text": c.original,
                "corrected": c.corrected,
                "description": c.explanation_en,
                "description_ar": c.explanation_ar,
            })

        # Step 5: per-word info
        words_info = []
        for wi, (s, e) in enumerate(pred["word_offsets"]):
            word_surface = bare_text[s:e]
            # Word-level diacritized form
            wdiac = []
            for i in range(s, e):
                wdiac.append(bare_text[i])
                if i < len(pred["diac_pred"]) and is_arabic_letter(bare_text[i]):
                    cls = pred["diac_pred"][i]
                    if 0 < cls < len(DIAC_LABELS):
                        wdiac.append(DIAC_LABELS[cls])
            word_diac = "".join(wdiac)

            if wi < len(pred["irab_pred"]):
                role = IRAB_LABELS[pred["irab_pred"][wi]]
                irab_conf = pred["irab_conf"][wi]
            else:
                role, irab_conf = "<pad>", 0.0

            # Per-word diacritization confidence (mean across chars)
            char_confs = [pred["diac_conf"][i] for i in range(s, min(e, len(pred["diac_conf"])))]
            diac_conf = float(np.mean(char_confs)) if char_confs else 0.0

            irab_text = pred["irab_texts"][wi] if wi < len(pred["irab_texts"]) else ""

            words_info.append({
                "index": wi,
                "surface": word_surface,
                "diacritized": word_diac,
                "role": role,
                "role_ar": role_to_ar(role),
                "role_en": role_to_en(role),
                "irab_text": irab_text,
                "diac_confidence": diac_conf,
                "irab_confidence": float(irab_conf),
                "low_confidence": float(irab_conf) < self.confidence_threshold,
            })

        return PredictionResult(
            input_text=text,
            diacritized=diacritized,
            words=words_info,
            errors=err_spans,
            tier=tier_result.tier,
            tier_flags=tier_result.flags,
        )
