# Development guide (read this first if you're using Claude Code)

This doc is the map for the project. When Claude Code is editing this repo, it should read this file first.

## Mental model

The project has **three kinds of code**, separated deliberately:

1. **Rule-based code** (`src/irab_tashkeel/rules/`) — deterministic, no learned parameters, fully explainable. Used for (a) orthographic pre-processing before the model sees the text, and (b) a fallback path when the neural model has low confidence.

2. **Neural code** (`src/irab_tashkeel/models/`, `training/`) — the learned components. A shared char-transformer encoder with three task heads.

3. **Glue code** (`src/irab_tashkeel/inference/`, `evaluation/`) — connects rule + neural outputs into a unified API, generates explanations, computes metrics.

The Streamlit app in `app/` and the notebooks in `notebooks/` are downstream consumers of the package — they import from `irab_tashkeel.*` and don't contain research logic.

## Which file does what

### Data (`src/irab_tashkeel/data/`)

| File | What it does | When to edit |
|---|---|---|
| `schema.py` | Defines `MTLExample`, `WordAnnotation` — the unified data format | When adding a new annotation type |
| `qac.py` | Parses the Quranic Arabic Corpus `.txt` file, derives i'rāb roles from (tag + case + context) | When improving i'rāb role inference from QAC features |
| `tashkeela.py` | Streams Tashkeela sentences from HuggingFace or local files | When adding new Tashkeela sources |
| `i3rab.py` | Parses the I3rab Dependency Treebank CoNLL-like format | When the I3rab format changes |
| `synthetic.py` | Injects synthetic errors (hamza drop, tāʾ marbūṭa swap, case swap) into gold text | When adding new error types |
| `build_dataset.py` | Combines all sources into a single `list[MTLExample]` | When changing the training mix ratio |

### Models (`src/irab_tashkeel/models/`)

| File | What it does | When to edit |
|---|---|---|
| `labels.py` | `DIAC_LABELS`, `IRAB_LABELS`, `ERR_LABELS` + mappings | When adding a new role or error type |
| `encoder.py` | `CharTransformer` — the shared backbone | When changing encoder architecture (depth, width, attention type) |
| `heads.py` | `DiacHead`, `IrabHead`, `ErrorHead` — task-specific classifiers | When adding a new task or changing head capacity |
| `full_model.py` | `FullModel` — glues encoder + three heads together | When changing how heads pool / consume encoder output |
| `tokenizer.py` | `CharTokenizer` — the character vocabulary | Rarely. Only if adding new character classes. |

### Training (`src/irab_tashkeel/training/`)

| File | What it does |
|---|---|
| `dataset.py` | `MTLDataset`, `collate_fn` — PyTorch wrappers for `MTLExample` lists |
| `losses.py` | `MultiTaskLoss` — weighted sum with per-sample masking |
| `trainer.py` | Training loop with mixed-precision, gradient clipping, checkpointing |
| `cli.py` | `python -m irab_tashkeel.training.cli --config <path>` |

### Inference (`src/irab_tashkeel/inference/`)

| File | What it does |
|---|---|
| `predictor.py` | `Predictor.predict(text: str) -> PredictionResult` — the main inference API |
| `explanations.py` | Generates bilingual (Arabic + English) explanations per word |
| `hybrid.py` | Merges rule-based and neural outputs based on confidence threshold |

### Evaluation (`src/irab_tashkeel/evaluation/`)

| File | What it does |
|---|---|
| `metrics.py` | DER, WER, i'rāb role accuracy, error detection F1 |
| `benchmark.py` | Runs on Tashkeela test set, QAC held-out verses, I3rab held-out |
| `eval_cli.py` | `python -m irab_tashkeel.evaluation.eval_cli --checkpoint X` |

### Rules (`src/irab_tashkeel/rules/`)

| File | What it does |
|---|---|
| `orthographic.py` | Hamza qaṭʿ, tāʾ marbūṭa, tatweel corrections |
| `grammar.py` | `GrammarEngine` — the Tier 1/2/3 rule system |
| `case_mapping.py` | Maps `(case, state, number)` → final diacritic |
| `tiers.py` | Sentence classifier — detects Tier 2/3 constructions (kāna, inna, relatives) |

## Common tasks — recipes

### "Add a new error type (e.g., sukūn missing)"

1. Add class to `src/irab_tashkeel/models/labels.py`:
   ```python
   ERR_LABELS = [..., "B-sukun", "I-sukun"]
   ```
2. Add injector to `src/irab_tashkeel/data/synthetic.py`:
   ```python
   def inject_sukun_missing(text): ...
   ```
3. Wire into `build_dataset.py` mix.
4. Re-train: `python -m irab_tashkeel.training.cli --config configs/model_medium.yaml`.

### "Add a new i'rāb role"

1. `src/irab_tashkeel/models/labels.py` — add to `IRAB_LABELS`.
2. `src/irab_tashkeel/data/qac.py` — update `qac_word_to_irab()` to produce the new role.
3. `src/irab_tashkeel/inference/explanations.py` — add Arabic + English labels.
4. `src/irab_tashkeel/rules/grammar.py` — add rule that triggers the role.
5. Re-train.

### "Change the confidence threshold for hybrid fallback"

Edit `configs/*.yaml`:
```yaml
inference:
  hybrid:
    confidence_threshold: 0.7  # was 0.6
```

No code change needed; `predictor.py` reads the config.

### "Debug: model is not learning"

In order:
1. `pytest tests/test_data.py` — is the data pipeline producing sensible examples?
2. `pytest tests/test_models.py` — does the forward pass produce right shapes?
3. Check `runs/<name>/tensorboard/` for loss curves. If loss is flat: learning rate too high/low, or head not getting gradients because all samples have `mask=False`.
4. Sample 5 examples from `MTLDataset[0..5]` and print them — sanity-check labels are correct.

### "Deploy a new version to HF Spaces"

```bash
# 1. Upload model weights to HF Hub
python scripts/upload_to_hub.py --checkpoint runs/medium/best.pt --repo <your-handle>/irab-tashkeel-model

# 2. Push app/ directory to the Space repo
cd app/
git remote add space https://huggingface.co/spaces/<your-handle>/irab-demo
git push space main
```

## Conventions

- **All Arabic strings in code use literal Arabic** (not Buckwalter). Use raw strings (`r"..."`) to avoid Python escape weirdness.
- **File-level docstring + type hints** on every public function.
- **No notebooks in the package** — notebooks are demo/exploration only, not importable.
- **Configs drive everything** — no hardcoded hyperparameters outside `configs/`.
- **Models load from config dicts**, never from positional args. Easier to serialize.

## Testing

```bash
pip install -e ".[dev]"
pytest                     # run all tests
pytest tests/test_data.py  # just data tests
pytest -k "metric"         # tests matching a keyword
pytest --cov=irab_tashkeel # with coverage
```

Tests aim to be fast (<10s total) — no network calls, no model loading. Mock where needed.

## Things NOT to do

- Don't put Streamlit imports in `src/irab_tashkeel/`. Streamlit is an app-layer dependency.
- Don't mix rule outputs and neural outputs in `models/`. That belongs in `inference/hybrid.py`.
- Don't add new datasets without updating `schema.py`. The unified `MTLExample` is the contract.
- Don't train with `mask_*=True` for labels you don't actually have. Spurious supervision hurts the other heads.

## If you're Claude Code and asked to make changes

1. **Read the relevant doc first.** If it's a data change → `docs/DATA.md`. Model change → `docs/ARCHITECTURE.md`. Training → `docs/TRAINING.md`.
2. **Locate the file using the tables above.** Don't grep — the structure is deliberate.
3. **Write a test before the change**, if possible. Tests are in `tests/test_<module>.py`.
4. **Run tests after the change.** `pytest tests/test_<module>.py -v`.
5. **Update docs if the change affects the public API** (`Predictor.predict()` signature, CLI flags, config keys).
