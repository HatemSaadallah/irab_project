"""Training CLI: `python -m irab_tashkeel.training.cli --config configs/model_medium.yaml`."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

from ..data.build_dataset import build_combined_dataset, load_examples, report, save_examples
from ..models.full_model import FullModel, ModelConfig
from ..models.labels import DIAC_LABELS, ERR_LABELS, IRAB_LABELS, VOCAB_SIZE
from .dataset import MTLDataset
from .losses import LossConfig, MultiTaskLoss
from .trainer import Trainer, TrainingConfig


def load_yaml_config(path: Path | str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_from_config(cfg: Dict) -> FullModel:
    """Build a FullModel from a config dict (usually loaded from YAML)."""
    model_cfg_dict = {
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
    return FullModel(ModelConfig.from_dict(model_cfg_dict))


def main():
    parser = argparse.ArgumentParser(description="Train the i'rab + tashkeel model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Checkpoint dir (default: runs/<config-name>)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to a checkpoint to resume from")
    parser.add_argument("--dataset-cache", type=str, default="data/cache/combined.pkl",
                       help="Where to cache the combined dataset")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Re-build the dataset from scratch")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seeds(cfg.get("training", {}).get("seed", 42))

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
            seed=cfg.get("training", {}).get("seed", 42),
            data_dir=Path(data_cfg.get("data_dir", "data")),
        )
        save_examples(all_examples, cache_path)
        print(f"Cached dataset to {cache_path}")

    report(all_examples)

    # Split train/val
    val_split = cfg.get("evaluation", {}).get("val_split", 0.1)
    n_val = max(1, int(len(all_examples) * val_split))
    train_examples = all_examples[:-n_val]
    val_examples = all_examples[-n_val:]
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")

    max_len = cfg["model"].get("max_len", 512)
    train_ds = MTLDataset(train_examples, max_len=max_len)
    val_ds = MTLDataset(val_examples, max_len=max_len)

    # --- Model ---
    model = build_model_from_config(cfg)
    print(f"Model: {model.n_params() / 1e6:.2f}M params")
    print(f"  encoder: hidden={model.config.encoder.hidden}, layers={model.config.encoder.n_layers}")

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    # --- Loss + trainer ---
    loss_cfg = LossConfig(**{k: v for k, v in cfg.get("loss", {}).items() if k in LossConfig.__annotations__})
    loss_fn = MultiTaskLoss(loss_cfg)

    tr_cfg_dict = cfg.get("training", {})
    training_config = TrainingConfig(
        **{k: v for k, v in tr_cfg_dict.items() if k in TrainingConfig.__annotations__}
    )
    eval_cfg = cfg.get("evaluation", {})
    training_config.eval_every_n_steps = eval_cfg.get("eval_every_n_steps", training_config.eval_every_n_steps)
    training_config.early_stopping_patience = eval_cfg.get("early_stopping_patience", training_config.early_stopping_patience)

    trainer = Trainer(model=model, loss_fn=loss_fn, config=training_config, output_dir=output_dir)

    # Snapshot config
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # Train
    result = trainer.train(train_ds, val_ds)
    print("\nTraining complete.")
    print(f"Best val loss: {result['best_val_loss']:.4f}")
    print(f"Checkpoints at: {output_dir}")


if __name__ == "__main__":
    main()
