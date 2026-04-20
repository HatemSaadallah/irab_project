"""Training loop — mixed-precision, gradient clipping, checkpointing, TB logging."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..models.full_model import FullModel
from .dataset import MTLDataset, collate_fn
from .losses import MultiTaskLoss, LossConfig


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    n_epochs: int = 15
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "none"
    num_workers: int = 4
    seed: int = 42
    eval_every_n_steps: int = 2000
    log_every_n_steps: int = 50
    early_stopping_patience: int = 3
    keep_all_checkpoints: bool = False


def cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    """LR scheduler: linear warmup then cosine decay."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        model: FullModel,
        loss_fn: MultiTaskLoss,
        config: TrainingConfig,
        output_dir: Path | str,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.device.type == "cuda" and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # AMP setup
        self.use_amp = config.mixed_precision in {"fp16", "bf16"} and self.device.type == "cuda"
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            config.mixed_precision, torch.float32
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        self.step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.metrics_history: List[Dict] = []

    def _move_to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            out = self.model(batch["char_ids"], batch["attention_mask"], batch["word_offsets"])
            losses = self.loss_fn(out, batch)

        self.scaler.scale(losses["total"]).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if hasattr(self, "scheduler"):
            self.scheduler.step()

        return {k: float(v) if torch.is_tensor(v) else v for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals = defaultdict(list)
        for batch in val_loader:
            batch = self._move_to_device(batch)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(batch["char_ids"], batch["attention_mask"], batch["word_offsets"])
                losses = self.loss_fn(out, batch)
            for k, v in losses.items():
                totals[k].append(float(v))
        return {k: sum(vs) / max(1, len(vs)) for k, vs in totals.items()}

    def save_checkpoint(self, name: str, extras: Optional[Dict] = None) -> Path:
        """Save model + optimizer + metadata."""
        path = self.output_dir / f"{name}.pt"
        payload = {
            "state_dict": self.model.state_dict(),
            "config": self.model.config.to_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
        }
        if extras:
            payload.update(extras)
        torch.save(payload, path)
        return path

    def train(self, train_ds: MTLDataset, val_ds: MTLDataset) -> Dict:
        torch.manual_seed(self.config.seed)
        cfg = self.config

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=True,
        )

        total_steps = len(train_loader) * cfg.n_epochs
        self.scheduler = cosine_warmup_schedule(
            self.optimizer, cfg.warmup_steps, total_steps
        )

        for epoch in range(cfg.n_epochs):
            running = defaultdict(list)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.n_epochs}")
            for batch in pbar:
                batch = self._move_to_device(batch)
                step_losses = self._train_step(batch)
                self.step += 1

                for k, v in step_losses.items():
                    running[k].append(v)

                if self.step % cfg.log_every_n_steps == 0:
                    pbar.set_postfix({k: f"{sum(running[k][-50:]) / max(1, min(50, len(running[k]))):.3f}"
                                     for k in running})

            # End-of-epoch eval
            val_metrics = self.evaluate(val_loader)
            epoch_record = {
                "epoch": epoch + 1,
                "step": self.step,
                "train": {k: sum(vs) / max(1, len(vs)) for k, vs in running.items()},
                "val": val_metrics,
            }
            self.metrics_history.append(epoch_record)
            self._save_metrics_json()

            train_str = " ".join(f"{k}={epoch_record['train'][k]:.3f}" for k in ("total", "diac", "irab", "err") if k in epoch_record["train"])
            val_str = " ".join(f"{k}={val_metrics[k]:.3f}" for k in ("total", "diac", "irab", "err") if k in val_metrics)
            print(f"[Epoch {epoch+1}]  train: {train_str}  |  val: {val_str}")

            # Best checkpoint
            val_total = val_metrics.get("total", float("inf"))
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.save_checkpoint("best")
                print(f"  ✓ saved best checkpoint (val_total={val_total:.3f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.early_stopping_patience:
                    print(f"Early stopping — no improvement for {cfg.early_stopping_patience} epochs")
                    break

            if cfg.keep_all_checkpoints:
                self.save_checkpoint(f"epoch_{epoch+1}")

        # Final checkpoint always saved
        self.save_checkpoint("final")
        return {"metrics_history": self.metrics_history, "best_val_loss": self.best_val_loss}

    def _save_metrics_json(self):
        path = self.output_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
