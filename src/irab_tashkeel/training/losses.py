"""Multi-task loss with per-sample masking.

Only samples where mask_<head>=True contribute to each head's loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.labels import IRAB_PAD_ID


@dataclass
class LossConfig:
    alpha_diac: float = 1.0
    beta_irab: float = 0.5
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
        """Compute weighted sum of the three head losses.

        Returns a dict with "total" and per-head breakdowns (for logging).
        Zero-tensor losses are returned for heads with no supervision in batch.
        """
        device = out["diac"].device
        zero = torch.zeros(1, device=device).squeeze()
        losses = {"diac": zero.clone(), "irab": zero.clone(), "err": zero.clone()}

        # --- Diacritization loss ---
        mask_diac = batch["mask_diac"]
        if mask_diac.any():
            idx = mask_diac.nonzero(as_tuple=True)[0]
            diac_logits = out["diac"][idx]                    # (b, L, C)
            diac_labels = batch["diac_labels"][idx]           # (b, L)
            attn = batch["attention_mask"][idx]               # (b, L)
            # Set labels at padded positions to -100 so CE ignores them
            labels = diac_labels.masked_fill(attn == 0, -100)
            losses["diac"] = F.cross_entropy(
                diac_logits.reshape(-1, diac_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )

        # --- I'rab loss ---
        mask_irab = batch["mask_irab"]
        if mask_irab.any():
            idx = mask_irab.nonzero(as_tuple=True)[0]
            irab_logits = out["irab"][idx]                    # (b, W, C)
            irab_labels = batch["irab_labels"][idx]           # (b, W)
            losses["irab"] = F.cross_entropy(
                irab_logits.reshape(-1, irab_logits.size(-1)),
                irab_labels.reshape(-1),
                ignore_index=IRAB_PAD_ID,
                label_smoothing=self.config.label_smoothing,
            )

        # --- Error detection loss ---
        mask_err = batch["mask_err"]
        if mask_err.any():
            idx = mask_err.nonzero(as_tuple=True)[0]
            err_logits = out["err"][idx]                      # (b, L, C)
            err_labels = batch["err_labels"][idx]             # (b, L)
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
            + self.config.gamma_err * losses["err"]
        )
        losses["total"] = total
        return losses
