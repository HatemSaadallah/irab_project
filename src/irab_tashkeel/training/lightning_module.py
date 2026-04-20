"""PyTorch Lightning wrapper for multi-GPU training.

Wraps FullModel + MultiTaskLoss in a LightningModule so we can use
Lightning's Trainer with strategy="ddp" for true multi-GPU parallelism.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch

from ..models.full_model import FullModel, ModelConfig
from .losses import LossConfig, MultiTaskLoss


class IrabLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: Dict,
        loss_config: Dict,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = FullModel(ModelConfig.from_dict(model_config))
        self.loss_fn = MultiTaskLoss(LossConfig(**loss_config))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    def forward(self, char_ids, attention_mask, word_offsets):
        return self.model(char_ids, attention_mask, word_offsets)

    def _step(self, batch, stage: str):
        out = self.model(batch["char_ids"], batch["attention_mask"], batch["word_offsets"])
        losses = self.loss_fn(out, batch)
        for k, v in losses.items():
            self.log(
                f"{stage}/{k}", v,
                on_step=(stage == "train"), on_epoch=True,
                prog_bar=(k == "total"), sync_dist=True,
            )
        return losses["total"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup = self.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class LegacyCheckpointCallback(pl.Callback):
    """Saves the best/final model in the legacy `.pt` format used by Predictor."""

    def __init__(self, output_dir: str | Path):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.metrics_history: List[Dict] = []

    def _save(self, pl_module: IrabLightningModule, name: str, val_loss: float):
        inner = pl_module.model
        path = self.output_dir / f"{name}.pt"
        torch.save({
            "state_dict": inner.state_dict(),
            "config": inner.config.to_dict(),
            "step": pl_module.trainer.global_step,
            "best_val_loss": val_loss,
        }, path)
        return path

    def _save_metrics_json(self):
        path = self.output_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        train = {k.split("/", 1)[1]: v for k, v in metrics.items() if k.startswith("train/") and k.endswith("_epoch")}
        if not train:
            train = {k.split("/", 1)[1]: v for k, v in metrics.items() if k.startswith("train/")}
        val = {k.split("/", 1)[1]: v for k, v in metrics.items() if k.startswith("val/")}

        epoch_record = {
            "epoch": trainer.current_epoch + 1,
            "step": trainer.global_step,
            "train": train,
            "val": val,
        }
        self.metrics_history.append(epoch_record)
        self._save_metrics_json()

        val_total = val.get("total", float("inf"))
        if val_total < self.best_val_loss and trainer.is_global_zero:
            self.best_val_loss = val_total
            self._save(pl_module, "best", val_total)
            print(f"  ✓ saved best checkpoint (val_total={val_total:.3f})")

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._save(pl_module, "final", self.best_val_loss)
