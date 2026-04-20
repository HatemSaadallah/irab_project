"""Training CLI: `python -m irab_tashkeel.training.cli --config configs/model_medium.yaml`."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from ..data.build_dataset import build_combined_dataset, load_examples, report, save_examples
from ..models.labels import DIAC_LABELS, ERR_LABELS, IRAB_LABELS, VOCAB_SIZE
from .dataset import MTLDataset, collate_fn
from .lightning_module import IrabLightningModule, LegacyCheckpointCallback


def load_yaml_config(path: Path | str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config_dict(cfg: Dict) -> Dict:
    return {
        "encoder": {
            "vocab_size": VOCAB_SIZE,
            "hidden": cfg["model"].get("hidden", 768),
            "n_heads": cfg["model"].get("n_heads", 12),
            "n_layers": cfg["model"].get("n_layers", 12),
            "max_len": cfg["model"].get("max_len", 512),
            "dropout": cfg["model"].get("dropout", 0.1),
        },
        "n_diac": cfg["heads"].get("n_diac", len(DIAC_LABELS)),
        "n_irab": cfg["heads"].get("n_irab", len(IRAB_LABELS)),
        "n_err": cfg["heads"].get("n_err", len(ERR_LABELS)),
        "head_dropout": cfg["heads"].get("dropout", 0.1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the i'rab + tashkeel model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dataset-cache", type=str, default="data/cache/combined.pkl")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    seed = cfg.get("training", {}).get("seed", 42)
    pl.seed_everything(seed, workers=True)

    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / Path(args.config).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    cache_path = Path(args.dataset_cache)
    if not args.force_rebuild and cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        all_examples = load_examples(cache_path)
    else:
        data_cfg = cfg.get("data", {})
        print("Building combined dataset …")
        all_examples = build_combined_dataset(
            tashkeela_n=data_cfg.get("tashkeela_n", 30000),
            qac_max_verses=data_cfg.get("qac_max_verses"),
            i3rab_path=Path(data_cfg["i3rab_path"]) if data_cfg.get("i3rab_path") else None,
            synthetic_per_type=data_cfg.get("synthetic_per_type", 2000),
            seed=seed,
            data_dir=Path(data_cfg.get("data_dir", "data")),
        )
        save_examples(all_examples, cache_path)
        print(f"Cached dataset to {cache_path}")

    report(all_examples)

    val_split = cfg.get("evaluation", {}).get("val_split", 0.1)
    n_val = max(1, int(len(all_examples) * val_split))
    train_examples = all_examples[:-n_val]
    val_examples = all_examples[-n_val:]
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")

    max_len = cfg["model"].get("max_len", 512)
    train_ds = MTLDataset(train_examples, max_len=max_len)
    val_ds = MTLDataset(val_examples, max_len=max_len)

    tr_cfg = cfg.get("training", {})
    train_loader = DataLoader(
        train_ds, batch_size=tr_cfg.get("batch_size", 32), shuffle=True,
        collate_fn=collate_fn, num_workers=tr_cfg.get("num_workers", 4),
        pin_memory=True, persistent_workers=tr_cfg.get("num_workers", 4) > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr_cfg.get("batch_size", 32), shuffle=False,
        collate_fn=collate_fn, num_workers=tr_cfg.get("num_workers", 4),
        pin_memory=True, persistent_workers=tr_cfg.get("num_workers", 4) > 0,
    )

    # --- Lightning module ---
    model_config = build_model_config_dict(cfg)
    loss_config = {k: v for k, v in cfg.get("loss", {}).items()
                   if k in {"alpha_diac", "beta_irab", "gamma_err", "label_smoothing"}}

    pl_module = IrabLightningModule(
        model_config=model_config,
        loss_config=loss_config,
        learning_rate=tr_cfg.get("learning_rate", 3.0e-4),
        weight_decay=tr_cfg.get("weight_decay", 0.01),
        warmup_steps=tr_cfg.get("warmup_steps", 1000),
    )
    print(f"Model: {pl_module.model.n_params() / 1e6:.2f}M params")
    print(f"  encoder: hidden={pl_module.model.config.encoder.hidden}, "
          f"layers={pl_module.model.config.encoder.n_layers}")

    if args.resume:
        print(f"Loading weights from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        pl_module.model.load_state_dict(ckpt["state_dict"])

    # --- Callbacks ---
    eval_cfg = cfg.get("evaluation", {})
    callbacks = [
        LegacyCheckpointCallback(output_dir),
        EarlyStopping(
            monitor="val/total",
            patience=eval_cfg.get("early_stopping_patience", 3),
            mode="min",
        ),
    ]

    # --- Trainer ---
    precision_map = {"fp16": "16-mixed", "bf16": "bf16-mixed", "none": "32-true"}
    precision = precision_map.get(tr_cfg.get("mixed_precision", "fp16"), "16-mixed")

    n_gpus = torch.cuda.device_count()
    accelerator = "gpu" if n_gpus > 0 else "cpu"
    devices = n_gpus if n_gpus > 0 else 1
    strategy = "ddp_find_unused_parameters_true" if n_gpus > 1 else "auto"

    print(f"Trainer: accelerator={accelerator}, devices={devices}, strategy={strategy}, precision={precision}")

    trainer = pl.Trainer(
        max_epochs=tr_cfg.get("n_epochs", 15),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        gradient_clip_val=tr_cfg.get("gradient_clip", 1.0),
        callbacks=callbacks,
        log_every_n_steps=tr_cfg.get("log_every_n_steps", 50),
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
    )

    # Snapshot config
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.is_global_zero:
        print("\nTraining complete.")
        print(f"Checkpoints at: {output_dir}")


if __name__ == "__main__":
    main()
