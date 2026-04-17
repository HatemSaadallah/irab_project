# Architecture

## The question this project is built to answer

**Can we make Arabic diacritization explainable without sacrificing accuracy?**

Existing SOTA systems (CATT, Sadeed, CAMeL-BERT) are pure neural seq2seq or char-classifiers. They hit ~2-5% DER on WikiNews. But ask them *why* `كتاب` becomes `الكتابُ` (nominative) instead of `الكتابَ` (accusative) — they can't tell you. For pedagogical uses (teaching Arabic learners) and linguistic research, this opacity is a deal-breaker.

Our answer: **add an i'rāb head as an auxiliary task**. The i'rāb label encodes the grammatical case assignment that determines the final diacritic. If the model learns to predict i'rāb well, its diacritization is grounded in recoverable grammatical reasoning.

## The model in one picture

```
  undiacritized Arabic text
          │
          ▼
  ┌─────────────────────┐
  │  orthographic       │   (rule-based, no learning)
  │  pre-processor      │   fixes hamza, tāʾ marbūṭa, tatweel
  └─────────────────────┘
          │
          ▼
  ┌─────────────────────┐
  │  char tokenizer     │
  └─────────────────────┘
          │
          ▼
  ┌─────────────────────┐
  │  shared encoder     │   CharTransformer, N_layers × hidden_dim
  │  (60M params)       │   pretrained via MLM on Tashkeela
  └─────────────────────┘
       /   │    \
      /    │     \
     ▼     ▼      ▼
  ┌───┐ ┌────┐ ┌────┐
  │diac│ │irab│ │err │
  └───┘ └────┘ └────┘
    │     │      │
    ▼     ▼      ▼
  diacritized + per-word i'rāb + error spans
    │     │      │
    ▼     ▼      ▼
  ┌─────────────────────┐
  │ hybrid merger       │   if neural_confidence < τ:
  │                     │      fall back to rule-based engine
  └─────────────────────┘
          │
          ▼
  ┌─────────────────────┐
  │ explanation         │   bilingual templates
  │ generator           │
  └─────────────────────┘
          │
          ▼
  final output: diacritized text + per-word grammatical breakdown + errors
```

## Three task heads — why these three

### Head A: Diacritization (per-character)

Per-character softmax over 15 classes:
- 8 basic: fatḥa, ḍamma, kasra, sukūn, fatḥatān, ḍammatān, kasratān, no-diacritic
- 7 compounds: each of the 7 above + shadda (for geminated consonants)

This matches CATT's formulation exactly. We reuse their pretrained weights where possible.

### Head B: I'rāb role (per-word)

Per-word softmax over **11 classes**:
- `fiil` — verb
- `harf_jarr` — preposition
- `harf_atf` — coordinator
- `harf_nafy` — negation particle
- `mabni_noun` — indeclinable noun/pronoun (demonstratives, relatives, some proper nouns)
- `N_marfu` — noun in nominative case (fāʿil, mubtadaʾ, khabar, naʿt-marfūʿ collapsed)
- `N_mansub` — noun in accusative case (mafʿūl bih, ḥāl, khabar-kāna collapsed)
- `ism_majrur` — noun in genitive after a preposition
- `mudaf_ilayh` — noun in genitive as an iḍāfa dependent
- `other` — fallback (currently catches punctuation, unknowns)
- `<pad>` — padding token

**Why only 11 classes and not more?** Fine-grained distinctions like fāʿil vs mubtadaʾ vs khabar-all-marfūʿ require *dependency parsing*, not just POS + case. QAC doesn't label these distinctions directly; deriving them requires rules that often fail. We collapse to the more tractable (POS × case) product. The rule-based grammar engine can separate fāʿil from mubtadaʾ post-hoc using the word's position and the sentence's leading POS.

**The payoff**: the i'rāb head has a dense learning signal (every word gets a label from QAC), and the collapsed labels correlate tightly with the diacritization decision (the case determines the final diacritic).

### Head C: Error detection (per-character BIO)

Per-character softmax over 7 BIO classes:
- `O` — no error
- `B-hamza`, `I-hamza` — hamza qaṭʿ missing
- `B-taa`, `I-taa` — tāʾ marbūṭa written as hāʾ
- `B-case`, `I-case` — case ending inconsistent with grammatical role

Trained purely on **synthetic corruptions** of gold-diacritized text. This works because the corruption rules are exactly the errors a learner would make.

## Multi-task training — the actual hard part

We have three corpora with different label coverage:

| Corpus | `mask_diac` | `mask_irab` | `mask_err` |
|---|---|---|---|
| Tashkeela | ✓ | ✗ | ✗ |
| QAC | ✓ | ✓ | ✗ |
| I3rab | ✗ (derived via CAMeL) | ✓ | ✗ |
| Synthetic | ✗ (alignment breaks) | ✗ | ✓ |

Each `MTLExample` carries three boolean masks. The loss is:

```
L_total = α·L_diac·mask_diac + β·L_irab·mask_irab + γ·L_err·mask_err
```

with batch-level reduction over only the samples where each mask is True. This way:
- Tashkeela samples train only the diac head (large data, basic signal)
- QAC samples train both diac + irab heads (small data, rich signal)
- Synthetic samples train only the err head

The ratios (α, β, γ) = (1.0, 0.5, 0.3) by default. The irab weight is high despite the smaller data because its signal is semantically richer — the encoder learns grammatical structure.

## Encoder design

A character-level Transformer:
- **Vocab size**: ~60 (Arabic letters + diacritics + space + pad/unk)
- **Hidden dim**: 768 (production), 256 (dev)
- **Layers**: 12 (production), 6 (dev)
- **Heads**: 12 (production), 8 (dev)
- **Max seq len**: 512 characters

Pretraining (optional): MLM on Tashkeela, following CATT. The Noisy-Student boost from CATT's paper (finetune on pseudo-labeled additional data) is future work.

**Why character-level, not subword?** Arabic morphology is fusional — a single word like `فسيكتبونها` ("and they will write it") encodes 6 morphemes. Subword tokenizers (BPE) create arbitrary splits that don't align with morphological structure. Character-level sidesteps this.

**Why not an LLM?** Sadeed (Kuwain-1.5B fine-tune) does this and hits 5.25% DER — competitive but not the best. CATT at 60M params hits 5.96%. The extra 1.4B parameters of Kuwain buy you ~0.7% DER. For a research project with an i'rāb angle, the param efficiency matters more than the last 1% DER.

## Inference — the hybrid path

The rule-based engine runs **in parallel** to the neural model, not before or after. For each word:

```python
neural_pred = model.predict(sentence)
rule_pred = grammar_engine.analyze(sentence)

for word_idx in sentence:
    if neural_pred.confidence[word_idx] >= threshold:
        use neural_pred.diacritics[word_idx]
        use neural_pred.irab[word_idx]
        explanation = "neural (conf={})"
    elif rule_pred.tier[word_idx] == 1:
        # high-confidence rule match
        use rule_pred.diacritics[word_idx]
        use rule_pred.irab[word_idx]
        explanation = rule_pred.rule_fired[word_idx]  # e.g., "R3b: N after prep → majrūr"
    else:
        # neural + rule disagree, rule has no opinion
        use neural_pred.diacritics[word_idx]
        explanation = "neural low confidence, no rule"
```

The explanation string is what makes this system pedagogically useful. A learner gets:

> *الكتابَ* — direct object (mafʿūl bih), accusative.
> Rule R1-maful: "N after fāʿil, not in PP → mafʿūl bih (manṣūb)"

instead of just `الكتابَ`.

## What's not in the model (and why)

**No syntactic dependency parsing.** Full dependency parsing would give us proper fāʿil vs mubtadaʾ distinction, but it requires either CamelParser2.0 as a separate upstream component (slows inference) or joint training on dependency labels (requires dep-labeled data we don't have at scale). We chose the collapsed-i'rāb approach instead.

**No kāna/inna handling.** Detected by the rule engine, flagged for neural fallback. A proper handler needs per-word rewriting of case assignments, which is hard in a single-pass model.

**No semantic disambiguation.** The system can tell `قلم` is a noun, but can't tell whether it means "pen" or "I trimmed" (both are valid depending on context). This is the fundamental limit of surface-form diacritization without meaning models.

## References

- Alasmary et al. 2024 — [CATT paper](https://arxiv.org/abs/2407.03236) — the architectural starting point
- Aldallal et al. 2025 — [Sadeed paper](https://arxiv.org/abs/2504.21635) — benchmark comparison
- Dukes & Habash 2010 — [QAC morphology paper](https://corpus.quran.com/) — data source
- Halabi et al. 2021 — [I3rab paper](https://nlp.psut.edu.jo/malaac.html) — i'rāb treebank
