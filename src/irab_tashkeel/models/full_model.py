"""Full multi-task model: encoder + three heads.

Load from a config dict so everything is serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .encoder import CharTransformer, EncoderConfig
from .heads import DiacHead, ErrorHead, IrabHead


@dataclass
class ModelConfig:
    encoder: EncoderConfig
    n_diac: int
    n_irab: int
    n_err: int
    head_dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        enc_dict = d.get("encoder", {}) or {"vocab_size": d["vocab_size"], "hidden": d["hidden"]}
        return cls(
            encoder=EncoderConfig.from_dict(enc_dict),
            n_diac=d["n_diac"],
            n_irab=d["n_irab"],
            n_err=d["n_err"],
            head_dropout=d.get("head_dropout", 0.1),
        )

    def to_dict(self) -> dict:
        return {"encoder": asdict(self.encoder), "n_diac": self.n_diac,
                "n_irab": self.n_irab, "n_err": self.n_err,
                "head_dropout": self.head_dropout}


class FullModel(nn.Module):
    """Multi-task model: one shared encoder, three independent task heads."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = CharTransformer(config.encoder)
        self.diac_head = DiacHead(config.encoder.hidden, config.n_diac, config.head_dropout)
        self.irab_head = IrabHead(config.encoder.hidden, config.n_irab, config.head_dropout)
        self.err_head = ErrorHead(config.encoder.hidden, config.n_err, config.head_dropout)

    def forward(
        self,
        char_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_offsets: List[List[Tuple[int, int]]],
    ) -> Dict[str, torch.Tensor]:
        h = self.encoder(char_ids, attention_mask)           # (B, L, H)
        diac_logits = self.diac_head(h)                      # (B, L, n_diac)
        err_logits = self.err_head(h)                        # (B, L, n_err)
        irab_logits, word_mask = self.irab_head(h, word_offsets)  # (B, W, n_irab), (B, W)

        return {
            "diac": diac_logits,
            "irab": irab_logits,
            "err": err_logits,
            "word_mask": word_mask,
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path):
        """Save model weights + config dict to a single .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "config": self.config.to_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "FullModel":
        """Load model + config from a .pt file."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        config = ModelConfig.from_dict(ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["state_dict"])
        return model
