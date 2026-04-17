# Evaluation

## Metrics

### Diacritization
- **DER** (Diacritic Error Rate): % of characters with wrong diacritic (standard metric, Fadel et al. 2019 definition)
- **DER-no-CE**: DER excluding the final character of each word (strips case-ending errors — isolates morphological vs syntactic errors)
- **WER-D**: % of words with at least one diacritic error

### I'rāb
- **Role accuracy**: % of words with correct i'rāb label
- **Per-role F1**: precision/recall/F1 for each of the 11 classes
- **Confusion matrix**: rows=gold, cols=predicted

### Error detection
- **Span F1**: precision/recall/F1 for exact span match (start, end, type all correct)
- **Per-type F1**: F1 for each error type (hamza, taa, case)
- **Character-level F1**: more lenient than span F1 — just token classification

All implemented in `src/irab_tashkeel/evaluation/metrics.py`.

## Benchmarks

Run the full benchmark suite:
```bash
python -m irab_tashkeel.evaluation.eval_cli --checkpoint runs/medium/best.pt \
    --benchmarks wikinews sadeed_diac_25 fadel_tashkeela i3rab_holdout \
    --output eval_results.json
```

### WikiNews
- **Size**: 2,500 MSA sentences
- **Source**: CATT repo ([dataset.zip](https://github.com/abjadai/catt/releases/download/v2/dataset.zip))
- **Expected numbers** (for comparison):
  - Darwish FRRNN 2020: 3.7% DER, 6.0% WER
  - CATT: 5.96% DER, 20.06% WER
  - Sadeed: 5.25% DER, 14.64% WER
  - Our medium: aim for 5-7% DER

### SadeedDiac-25
- **Size**: Held-out MSA + Classical, curated by Sadeed authors
- **Source**: [Sadeed paper](https://arxiv.org/abs/2504.21635), HF Hub
- **Expected**: 6-9% DER (mixed difficulty)

### Fadel Tashkeela test
- **Size**: ~107K words (Fadel 2019 refined test split)
- **Source**: HuggingFace `Misraj/Sadeed_Tashkeela` test split
- **Expected**: 3-6% DER (easier, Classical-heavy)

### I3rab held-out
- **Size**: Last 60 sentences of I3rab (never in training)
- **Use**: Primary i'rāb accuracy benchmark (since I3rab is MSA + manually labeled)

## What to report

For the paper, at minimum:
- DER + WER on WikiNews (MSA standard)
- DER on SadeedDiac-25 (cross-genre)
- I'rāb accuracy on I3rab held-out
- Error detection F1 on a synthetic held-out set

With a baseline comparison row: CATT, CAMeL-BERT, and (aspirationally) Sadeed.

## Significance testing

With the held-out splits size (60-2500 sentences), 0.5% DER differences may not be significant. For a publication, use paired bootstrap resampling:

```python
from irab_tashkeel.evaluation.metrics import paired_bootstrap
p_value = paired_bootstrap(our_preds, catt_preds, gold, metric="der", n_bootstrap=10000)
```

## Error analysis

After running eval, use the notebook:
```bash
jupyter notebook notebooks/error_analysis.ipynb
```

This loads the model's per-word predictions and groups errors by:
- I'rāb role (do we fail more on khabar than fāʿil?)
- Construction tier (simple vs kāna vs relative)
- Source corpus (does QAC-trained model overfit to Classical?)
- Word frequency (rare words harder?)
