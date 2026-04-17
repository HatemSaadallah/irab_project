"""Character-level Transformer encoder.

The shared backbone for all three task heads. Config-driven so we can scale
from 5M (dev) to 300M (production) without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    vocab_size: int
    hidden: int = 768
    n_heads: int = 12
    n_layers: int = 12
    max_len: int = 512
    dropout: float = 0.1
    activation: str = "gelu"

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


class CharTransformer(nn.Module):
    """A straightforward Transformer encoder with learned positional embeddings.

    We deliberately do NOT use rotary or ALiBi — the max_len is short (512 chars)
    and learned embeddings are simpler to reason about.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        self.char_emb = nn.Embedding(config.vocab_size, config.hidden)
        self.pos_emb = nn.Embedding(config.max_len, config.hidden)
        self.emb_dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.n_heads,
            dim_feedforward=config.hidden * 4,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,  # pre-norm is more stable
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm_out = nn.LayerNorm(config.hidden)

        # Init — small std for stable training
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.char_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        char_ids: torch.Tensor,          # (B, L)
        attention_mask: torch.Tensor,    # (B, L), 1 = real, 0 = pad
    ) -> torch.Tensor:
        """Return contextualized char embeddings of shape (B, L, hidden)."""
        B, L = char_ids.shape
        pos = torch.arange(L, device=char_ids.device).unsqueeze(0).expand(B, -1)
        x = self.char_emb(char_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        # PyTorch's src_key_padding_mask: True = to IGNORE (opposite of our attn mask)
        pad_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm_out(x)
        return x

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
