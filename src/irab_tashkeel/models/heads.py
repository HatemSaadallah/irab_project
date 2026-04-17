"""Task-specific output heads: diacritization, i'rab role, error detection.

Each head is a small module on top of the shared encoder. They share NO
parameters among themselves; only the encoder is shared.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class DiacHead(nn.Module):
    """Per-character diacritic classifier."""

    def __init__(self, hidden: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # encoder_output: (B, L, H)  →  (B, L, n_classes)
        return self.proj(encoder_output)


class IrabHead(nn.Module):
    """Per-word i'rab classifier.

    Takes encoder output (B, L, H) and word offsets (list of list of (start, end)),
    pools each word's character embeddings (mean-pool over its span), then classifies.
    """

    def __init__(self, hidden: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        word_offsets: List[List[Tuple[int, int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (word_logits, word_mask).

        word_logits: (B, W_max, n_classes) — W_max = max word count in batch
        word_mask:   (B, W_max) — 1 for real words, 0 for padding
        """
        B, L, H = encoder_output.shape
        w_max = max((len(wo) for wo in word_offsets), default=1)

        pooled = torch.zeros(B, w_max, H, device=encoder_output.device, dtype=encoder_output.dtype)
        mask = torch.zeros(B, w_max, device=encoder_output.device)

        for b, offsets in enumerate(word_offsets):
            for wi, (s, e) in enumerate(offsets):
                if s < L and e <= L and s < e:
                    pooled[b, wi] = encoder_output[b, s:e].mean(dim=0)
                    mask[b, wi] = 1.0

        logits = self.proj(pooled)
        return logits, mask


class ErrorHead(nn.Module):
    """Per-character BIO error classifier."""

    def __init__(self, hidden: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.proj(encoder_output)
