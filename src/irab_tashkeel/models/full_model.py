"""Full multi-task model: encoder + heads + per-word i'rab decoder.

Heads:
    - DiacHead: per-character diacritization
    - IrabHead: per-word coarse i'rab classification (auxiliary)
    - IrabDecoder: per-word seq2seq Arabic i'rab string generation (main)
    - ErrorHead: per-character BIO error detection

Load from a config dict so everything is serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import CharTransformer, EncoderConfig
from .heads import DiacHead, ErrorHead, IrabHead
from .irab_decoder import IrabDecoder, IrabDecoderConfig


@dataclass
class ModelConfig:
    encoder: EncoderConfig
    n_diac: int
    n_irab: int
    n_err: int
    head_dropout: float = 0.1
    irab_decoder: Optional[IrabDecoderConfig] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        enc_dict = d.get("encoder", {}) or {"vocab_size": d["vocab_size"], "hidden": d["hidden"]}
        encoder = EncoderConfig.from_dict(enc_dict)
        irab_dec = None
        if d.get("irab_decoder"):
            dec_dict = dict(d["irab_decoder"])
            dec_dict.setdefault("encoder_hidden", encoder.hidden)
            irab_dec = IrabDecoderConfig.from_dict(dec_dict)
        return cls(
            encoder=encoder,
            n_diac=d["n_diac"],
            n_irab=d["n_irab"],
            n_err=d["n_err"],
            head_dropout=d.get("head_dropout", 0.1),
            irab_decoder=irab_dec,
        )

    def to_dict(self) -> dict:
        out = {
            "encoder": asdict(self.encoder),
            "n_diac": self.n_diac,
            "n_irab": self.n_irab,
            "n_err": self.n_err,
            "head_dropout": self.head_dropout,
        }
        if self.irab_decoder is not None:
            out["irab_decoder"] = self.irab_decoder.to_dict()
        return out


class FullModel(nn.Module):
    """Multi-task model: shared encoder + classification heads + per-word i'rab decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = CharTransformer(config.encoder)
        self.diac_head = DiacHead(config.encoder.hidden, config.n_diac, config.head_dropout)
        self.irab_head = IrabHead(config.encoder.hidden, config.n_irab, config.head_dropout)
        self.err_head = ErrorHead(config.encoder.hidden, config.n_err, config.head_dropout)
        self.irab_decoder: Optional[IrabDecoder] = (
            IrabDecoder(config.irab_decoder) if config.irab_decoder is not None else None
        )

    def forward(
        self,
        char_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_offsets: List[List[Tuple[int, int]]],
        irab_target_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encoder(char_ids, attention_mask)                # (B, L, H)
        diac_logits = self.diac_head(h)                            # (B, L, n_diac)
        err_logits = self.err_head(h)                              # (B, L, n_err)
        irab_logits, word_mask = self.irab_head(h, word_offsets)   # (B, W, n_irab), (B, W)

        out = {
            "diac": diac_logits,
            "irab": irab_logits,
            "err": err_logits,
            "word_mask": word_mask,
            "encoder_hidden": h,
        }

        if self.irab_decoder is not None and irab_target_ids is not None:
            seq_logits, seq_labels, seq_word_mask = self.irab_decoder(
                h, word_offsets, irab_target_ids
            )
            out["irab_seq_logits"] = seq_logits        # (B, W, T-1, V)
            out["irab_seq_labels"] = seq_labels        # (B, W, T-1)
            out["irab_seq_word_mask"] = seq_word_mask  # (B, W)

        return out

    @torch.no_grad()
    def generate_irab(
        self,
        char_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_offsets: List[List[Tuple[int, int]]],
        max_len: Optional[int] = None,
    ) -> List[List[List[int]]]:
        """Greedy-decode the i'rab string for every real word in the batch."""
        if self.irab_decoder is None:
            raise RuntimeError("Model was built without an irab_decoder")
        h = self.encoder(char_ids, attention_mask)
        return self.irab_decoder.generate(h, word_offsets, max_len=max_len)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "config": self.config.to_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "FullModel":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        config = ModelConfig.from_dict(ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model
