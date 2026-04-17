---
title: Irab Tashkeel Demo
emoji: 📖
colorFrom: purple
colorTo: teal
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---

# I'rāb-Guided Arabic Diacritization

An explainable Arabic NLP demo that jointly predicts:
- **Tashkīl** (diacritics)
- **I'rāb** (grammatical role per word)
- **Errors** (orthographic + grammatical)

## Deployment

This Space expects either:
- An `$MODEL_CKPT` env var pointing to a local `.pt` file, or
- An `$HF_MODEL_REPO` env var pointing to a model repo containing `model.pt`

Set these in the Space settings → Variables & Secrets.

## How it works

A character-level Transformer encoder shared across three task heads. Trained on:
- Quranic Arabic Corpus (diacritics + i'rāb)
- Tashkeela (diacritics at scale)
- I3rab Treebank (MSA i'rāb)
- Synthetic error corruptions

See the [GitHub repo](https://github.com/) for full details.

## Limitations

- I'rāb labels are coarse (11 classes) — fine distinctions like fāʿil vs mubtadaʾ are collapsed
- Predominantly Classical Arabic training data (~98% Shamela library)
- No guarantee of linguistic correctness; use judgement for any consequential task
