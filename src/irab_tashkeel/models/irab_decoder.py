"""Per-word transformer decoder for detailed Arabic i'rab generation.

For each word in the input sentence, the decoder produces a free-form Arabic
i'rab string (e.g. "فعل مضارع مرفوع وعلامة رفعه الضمة الظاهرة") via BPE
autoregressive decoding.

The encoder's character-level output is mean-pooled over each word's character
span to produce a single per-word vector. That vector is the cross-attention
memory (a length-1 sequence) for that word's decoder. Each word decodes
independently — we flatten across (batch, word) for efficient batched decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .irab_tokenizer import EOS_ID, PAD_ID, SOS_ID


@dataclass
class IrabDecoderConfig:
    encoder_hidden: int
    vocab_size: int
    hidden: int = 256
    n_heads: int = 4
    n_layers: int = 3
    max_target_len: int = 64
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "IrabDecoderConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        return {
            "encoder_hidden": self.encoder_hidden,
            "vocab_size": self.vocab_size,
            "hidden": self.hidden,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_target_len": self.max_target_len,
            "dropout": self.dropout,
        }


class IrabDecoder(nn.Module):
    """Per-word autoregressive decoder over BPE i'rab targets."""

    def __init__(self, config: IrabDecoderConfig):
        super().__init__()
        self.config = config

        # Project encoder hidden into decoder hidden if dims differ.
        self.memory_proj = nn.Linear(config.encoder_hidden, config.hidden)

        # Target token + positional embeddings.
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(config.max_target_len, config.hidden)
        self.emb_dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden,
            nhead=config.n_heads,
            dim_feedforward=config.hidden * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.n_layers)
        self.norm_out = nn.LayerNorm(config.hidden)
        self.out_proj = nn.Linear(config.hidden, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[PAD_ID].zero_()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pool_words(
        encoder_output: torch.Tensor,
        word_offsets: List[List[Tuple[int, int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool encoder hidden states over each word's char span.

        Returns (pooled, mask):
            pooled: (B, W_max, H_enc)
            mask:   (B, W_max) — 1 for real words, 0 for padding
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
        return pooled, mask

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        """Standard causal mask: True = block attention to future positions."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        encoder_output: torch.Tensor,
        word_offsets: List[List[Tuple[int, int]]],
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass.

        Args:
            encoder_output: (B, L, H_enc)
            word_offsets:   per-batch list of (start, end) char spans per word
            target_ids:     (B, W_max, T_max), int64, padded with PAD_ID,
                            includes leading <sos> and trailing <eos>

        Returns:
            logits: (B, W_max, T_max - 1, vocab) — predictions for positions 1..T-1
            labels: (B, W_max, T_max - 1)        — corresponding gold labels
            word_mask: (B, W_max)                — 1 for real words
        """
        B, W_max, T = target_ids.shape

        pooled, word_mask = self._pool_words(encoder_output, word_offsets)
        # If batch's actual W_max from offsets differs from target_ids' W_max,
        # truncate / pad word_mask & pooled to match target_ids shape.
        if pooled.size(1) < W_max:
            pad = W_max - pooled.size(1)
            pooled = torch.nn.functional.pad(pooled, (0, 0, 0, pad))
            word_mask = torch.nn.functional.pad(word_mask, (0, pad))
        elif pooled.size(1) > W_max:
            pooled = pooled[:, :W_max]
            word_mask = word_mask[:, :W_max]

        # Project per-word memory: (B, W_max, H_dec), used as length-1 memory.
        memory = self.memory_proj(pooled)                        # (B, W, H_dec)
        memory = memory.reshape(B * W_max, 1, self.config.hidden)  # (N, 1, H_dec)

        # Decoder input: shift right (drop last, keep <sos>...<last-1>).
        dec_in = target_ids[:, :, :-1].reshape(B * W_max, T - 1)  # (N, T-1)
        dec_labels = target_ids[:, :, 1:]                          # (B, W, T-1)

        # Embed + position.
        positions = torch.arange(T - 1, device=target_ids.device).unsqueeze(0)
        x = self.tok_emb(dec_in) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Self-attn causal + key-padding masks.
        causal = self._causal_mask(T - 1, target_ids.device)
        # Pad positions in the input (everything that is PAD) — mask them out.
        tgt_pad = (dec_in == PAD_ID)

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
        )
        out = self.norm_out(out)
        logits = self.out_proj(out)                                # (N, T-1, V)
        logits = logits.reshape(B, W_max, T - 1, self.config.vocab_size)

        return logits, dec_labels, word_mask

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        word_offsets: List[List[Tuple[int, int]]],
        max_len: Optional[int] = None,
    ) -> List[List[List[int]]]:
        """Greedy decode for every real word in the batch.

        Returns nested list[B][W_b] of token-id lists (without <sos>/<eos>).
        """
        max_len = max_len or self.config.max_target_len
        B = encoder_output.size(0)
        device = encoder_output.device

        pooled, word_mask = self._pool_words(encoder_output, word_offsets)
        W_max = pooled.size(1)
        memory = self.memory_proj(pooled).reshape(B * W_max, 1, self.config.hidden)

        # All word slots start with <sos>. Pad slots will be ignored later.
        cur = torch.full((B * W_max, 1), SOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B * W_max, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            T_cur = cur.size(1)
            positions = torch.arange(T_cur, device=device).unsqueeze(0)
            x = self.tok_emb(cur) + self.pos_emb(positions)
            x = self.emb_dropout(x)
            causal = self._causal_mask(T_cur, device)
            out = self.decoder(tgt=x, memory=memory, tgt_mask=causal)
            out = self.norm_out(out)
            logits = self.out_proj(out[:, -1])         # (N, V)
            next_id = logits.argmax(dim=-1)            # (N,)
            # Once finished, freeze on PAD so subsequent decoded text is clean.
            next_id = torch.where(finished, torch.full_like(next_id, PAD_ID), next_id)
            cur = torch.cat([cur, next_id.unsqueeze(1)], dim=1)
            finished = finished | (next_id == EOS_ID)
            if bool(finished.all()):
                break

        # Unflatten and strip per-word according to word_mask.
        all_ids = cur.reshape(B, W_max, -1).cpu().tolist()
        wmask = word_mask.cpu().tolist()
        out_nested: List[List[List[int]]] = []
        for b in range(B):
            row = []
            for w in range(W_max):
                if wmask[b][w] < 0.5:
                    continue
                row.append(all_ids[b][w])
            out_nested.append(row)
        return out_nested

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
