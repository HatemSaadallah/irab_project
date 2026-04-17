# Training guide

## Quick paths

```bash
# Dev: 5M params, CPU ok, ~30min, good for pipeline testing
python -m irab_tashkeel.training.cli --config configs/model_small.yaml

# Production: 60M params, GPU required, ~4-6hr on T4, ~2hr on A100
python -m irab_tashkeel.training.cli --config configs/model_medium.yaml

# Large: 300M params, big GPU only, ~8-12hr on A100
python -m irab_tashkeel.training.cli --config configs/model_large.yaml
```

Checkpoints go to `runs/<config-name>/`:
```
runs/medium/
  best.pt            # best val-loss checkpoint
  epoch_N.pt         # per-epoch (optional; set keep_all_checkpoints: true)
  config.yaml        # snapshot of config used
  metrics.json       # per-epoch val metrics
  tensorboard/       # TB event files
```

## Configs — what the knobs mean

`configs/model_medium.yaml`:

```yaml
model:
  vocab_size: 62              # = len(CHAR_VOCAB); set by tokenizer
  hidden: 768                 # encoder width
  n_heads: 12                 # attention heads
  n_layers: 12                # encoder depth
  max_len: 512                # max sequence length
  dropout: 0.1

heads:
  n_diac: 15                  # from DIAC_LABELS
  n_irab: 11                  # from IRAB_LABELS
  n_err: 7                    # from ERR_LABELS (O + 3×BIO)

loss:
  alpha_diac: 1.0             # weight on diacritization loss
  beta_irab: 0.5              # weight on i'rāb loss
  gamma_err: 0.3              # weight on error detection loss
  label_smoothing: 0.1

data:
  tashkeela_n: 30000
  qac_n: 6236                 # use all QAC verses
  i3rab_path: null            # set to data/i3rab/i3rab.conllu if you have it
  synthetic_per_type: 2000    # × 3 error types = 6K synthetic

training:
  batch_size: 32
  learning_rate: 3.0e-4
  weight_decay: 0.01
  n_epochs: 15
  warmup_steps: 1000
  gradient_clip: 1.0
  mixed_precision: fp16
  num_workers: 4

evaluation:
  val_split: 0.1
  eval_every_n_steps: 2000
  early_stopping_patience: 3

logging:
  tensorboard: true
  wandb_project: null         # set to enable W&B
  log_every_n_steps: 50
```

## Hardware sizing

| Config | GPU memory (fp16) | GPU type | Batch size | Wall time for 15 epochs |
|---|---|---|---|---|
| small  | ~3 GB  | CPU or any GPU | 32 | ~30 min CPU, ~10 min GPU |
| medium | ~12 GB | T4 (16GB) | 32 | ~5 hr |
| medium | ~12 GB | A100 (40GB) | 64 | ~2 hr |
| large  | ~35 GB | A100 (40GB) or A6000 | 16 | ~10 hr |

For a single-GPU T4: use `medium`, batch_size=32, accumulation_steps=2 if OOM.
For multi-GPU: `training.cli --multi-gpu` enables DDP (still experimental — prefer single-GPU for now).

## The training loop (mechanics)

One step:
1. Sample batch from `MTLDataset` via `DataLoader` with `collate_fn`.
2. Forward pass: `encoder(chars) → {diac, irab, err}` per batch.
3. For each head, mask out samples where `mask_<head>=False`, compute loss on remainder.
4. Total loss = α·L_diac + β·L_irab + γ·L_err.
5. Backward + clip gradients + AMP scaling.
6. Optimizer step + scheduler step.

Every N steps:
- Log losses to tensorboard/W&B.
- If val interval reached: evaluate on val split, log metrics, save best checkpoint.

## Monitoring training

### Expected loss curves

At start:
- `L_diac ≈ log(15) ≈ 2.7` (uniform over 15 classes)
- `L_irab ≈ log(11) ≈ 2.4`
- `L_err ≈ log(7) ≈ 1.95`

After 1 epoch:
- `L_diac ≈ 0.8-1.2` (if learning)
- `L_irab ≈ 0.6-1.0`
- `L_err ≈ 0.1-0.3` (this one learns fast — simple task)

After 10 epochs:
- `L_diac ≈ 0.15-0.30`
- `L_irab ≈ 0.25-0.40`
- `L_err ≈ 0.03-0.08`

**Red flags**:
- Loss stays flat at initialization values → learning rate too low, or data pipeline bug.
- `L_err` drops to 0 instantly → error head is trivial (check that synthetic examples have variety).
- `L_diac` diverges → learning rate too high; reduce by 10×.
- `L_irab` stuck above 1.5 → class imbalance (check `IRAB_LABELS` distribution).

### What "good" looks like at end of training (medium config)

On the val split:
- Diacritization DER: 3-7%
- I'rāb role accuracy: 85-92%
- Error detection F1: 0.85-0.95

On SadeedDiac-25 (MSA+CA, held out from training): DER 5-8%. This is the number to report.

## Resuming training

```bash
python -m irab_tashkeel.training.cli \
    --config configs/model_medium.yaml \
    --resume runs/medium/epoch_7.pt
```

Preserves optimizer state, scheduler state, RNG state. Training continues as if uninterrupted.

## Curriculum tricks worth trying

1. **Warmup with Tashkeela-only** (5 epochs diacritization head only), then unfreeze all heads and train jointly. Can help the encoder find good representations before the irab head pulls in a conflicting direction.

2. **Progressive loss weighting**: start with β=0.1, γ=0.0 and anneal up. Prevents the smaller heads from overfitting to small data.

3. **Noisy-student** (CATT-style): after the first training run, use the model to pseudo-label a larger unlabeled Tashkeela subset, then retrain with noise augmentation on the pseudo-labels.

None of these are in v1 — future experiments.

## Common failure modes

**"Model's diacritics are blank"**: the diac head is predicting class 0 for everything (i.e., "no diacritic"). Usually because training examples have mostly padding (class 0) in their diac_labels. Check that `text_to_diac_labels()` is working on real diacritized input.

**"I'rāb predictions are always `other`"**: the irab head is not getting gradient because all batches hit `mask_irab=False`. Check your dataset mix — make sure QAC samples are actually in the training set.

**"Loss goes down but eval DER doesn't improve"**: probably label smoothing masking real signal. Try `label_smoothing: 0.0` and see if DER drops.

**"OOM on seemingly fine batch size"**: gradient checkpointing isn't on; add `training.gradient_checkpointing: true` to config.

## Reproducibility

Set seeds explicitly in config:
```yaml
training:
  seed: 42
```

But true reproducibility across GPUs is hard (cuDNN nondeterminism). Expect ±0.2% DER variation across runs of the same config.

## Distributed training (experimental)

```bash
torchrun --nproc_per_node=4 -m irab_tashkeel.training.cli \
    --config configs/model_large.yaml --multi-gpu
```

Uses DDP. Tested on 4×A100 — works but not heavily optimized. Gradient sync adds ~10% overhead.

## What to save for publication

When your best run is done, grab:
- `runs/<n>/best.pt` — the weights
- `runs/<n>/config.yaml` — the config used
- `runs/<n>/metrics.json` — per-epoch val metrics
- Final eval JSON from `python -m irab_tashkeel.evaluation.eval_cli --checkpoint runs/<n>/best.pt --output results.json`

These, plus the README and citation, are what goes to the paper + HF Hub.
