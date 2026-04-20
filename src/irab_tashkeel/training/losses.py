"""Multi-task loss with per-sample masking.

Heads:
    diac    — per-character cross-entropy
    irab    — per-word coarse classification (auxiliary)
    irab_seq — per-word seq2seq cross-entropy over BPE i'rab strings
    err     — per-character BIO cross-entropy

Only samples where mask_<head>=True contribute to each head's loss.
For irab_seq, an additional per-word mask filters out words with no string target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.irab_tokenizer import PAD_ID as IRAB_TOK_PAD_ID
from ..models.labels import IRAB_PAD_ID


@dataclass
class LossConfig:
    alpha_diac: float = 1.0
    beta_irab: float = 0.1        # auxiliary classification (down-weighted)
    delta_irab_seq: float = 0.5   # main per-word seq2seq objective
    gamma_err: float = 0.3
    label_smoothing: float = 0.1


class MultiTaskLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the weighted sum of head losses.

        Returns a dict with "total" and per-head breakdowns (for logging).
        Zero-tensor losses are returned for heads with no supervision in batch.
        """
        device = out["diac"].device
        zero = torch.zeros(1, device=device).squeeze()
        losses = {
            "diac": zero.clone(),
            "irab": zero.clone(),
            "irab_seq": zero.clone(),
            "err": zero.clone(),
        }

        # --- Diacritization loss ---
        mask_diac = batch["mask_diac"]
        if mask_diac.any():
            idx = mask_diac.nonzero(as_tuple=True)[0]
            diac_logits = out["diac"][idx]                    # (b, L, C)
            diac_labels = batch["diac_labels"][idx]           # (b, L)
            attn = batch["attention_mask"][idx]
            labels = diac_labels.masked_fill(attn == 0, -100)
            losses["diac"] = F.cross_entropy(
                diac_logits.reshape(-1, diac_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )

        # --- I'rab classification loss (auxiliary) ---
        mask_irab = batch["mask_irab"]
        if mask_irab.any():
            idx = mask_irab.nonzero(as_tuple=True)[0]
            irab_logits = out["irab"][idx]                    # (b, W, C)
            irab_labels = batch["irab_labels"][idx]
            losses["irab"] = F.cross_entropy(
                irab_logits.reshape(-1, irab_logits.size(-1)),
                irab_labels.reshape(-1),
                ignore_index=IRAB_PAD_ID,
                label_smoothing=self.config.label_smoothing,
            )

        # --- I'rab seq2seq loss ---
        if "irab_seq_logits" in out and mask_irab.any():
            idx = mask_irab.nonzero(as_tuple=True)[0]
            seq_logits = out["irab_seq_logits"][idx]          # (b, W, T-1, V)
            seq_labels = out["irab_seq_labels"][idx]          # (b, W, T-1)
            # Words with no per-word string target: mask out.
            word_sup = batch["irab_seq_word_mask"][idx]       # (b, W)
            # Set labels for unsupervised words/positions to PAD (ignored by CE).
            labels = seq_labels.clone()
            labels[~word_sup] = IRAB_TOK_PAD_ID
            losses["irab_seq"] = F.cross_entropy(
                seq_logits.reshape(-1, seq_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IRAB_TOK_PAD_ID,
                label_smoothing=self.config.label_smoothing,
            )

        # --- Error detection loss ---
        mask_err = batch["mask_err"]
        if mask_err.any():
            idx = mask_err.nonzero(as_tuple=True)[0]
            err_logits = out["err"][idx]                      # (b, L, C)
            err_labels = batch["err_labels"][idx]
            attn = batch["attention_mask"][idx]
            labels = err_labels.masked_fill(attn == 0, -100)
            losses["err"] = F.cross_entropy(
                err_logits.reshape(-1, err_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )

        total = (
            self.config.alpha_diac * losses["diac"]
            + self.config.beta_irab * losses["irab"]
            + self.config.delta_irab_seq * losses["irab_seq"]
            + self.config.gamma_err * losses["err"]
        )
        losses["total"] = total
        return losses
