# Data sources and schema

## The three corpora

### 1. Tashkeela (diacritization scale)

- **Source**: [Zerrouki & Balla 2017](https://sourceforge.net/projects/tashkeela/) — 75M fully-diacritized Arabic words
- **Cleaned version we use**: [Misraj/Sadeed_Tashkeela](https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela) on HuggingFace
- **Genre**: ~98% Classical Arabic (Shamela Library), ~1.15% MSA
- **Labels**: Diacritics only (no i'rāb, no errors)
- **Role in our pipeline**: Main supervision for the diacritization head. Large scale.

**How we load it:**
```python
from datasets import load_dataset
ds = load_dataset("Misraj/Sadeed_Tashkeela", split="train", streaming=True)
```

**Caveat**: Heavy Classical Arabic skew. The model learns Classical patterns well but may over-apply them to MSA text. Mitigation: we upsample QAC + I3rab (MSA-leaning) at training time.

### 2. Quranic Arabic Corpus (the i'rāb gold)

- **Source**: [corpus.quran.com](https://corpus.quran.com/) — ~78K segments across 6236 verses
- **Mirror with Arabic script**: [mustafa0x/quran-morphology](https://github.com/mustafa0x/quran-morphology)
- **Format**: tab-separated, location + form + tag + pipe-separated features
- **Labels**: Morphology (44 POS tags) + dependency structure + case/state/mood features
- **Role in our pipeline**: The richest training signal. Every word has both diacritics AND i'rāb features.

**Format example** (one line per segment, words have multiple segments):
```
(1:1:1:1)	بِسْمِ	P	PREFIX|+
(1:1:1:2)	سْمِ	N	STEM|POS:N|LEM:ٱسْم|ROOT:سمو|M|GEN
(1:1:2:1)	ٱللَّهِ	PN	STEM|POS:PN|LEM:ٱللَّه|ROOT:أله|GEN
```

**How we derive i'rāb roles from QAC tags** (in `src/irab_tashkeel/data/qac.py`):

| QAC tag | Feature filter | Our i'rāb role |
|---|---|---|
| V | — | `fiil` |
| P | — | `harf_jarr` |
| CONJ | — | `harf_atf` |
| NEG | — | `harf_nafy` |
| REL, DEM, PRON | — | `mabni_noun` |
| N, PN, ADJ | CASE=NOM | `N_marfu` |
| N, PN, ADJ | CASE=ACC | `N_mansub` |
| N, PN, ADJ | CASE=GEN, prev POS=P | `ism_majrur` |
| N, PN, ADJ | CASE=GEN, otherwise | `mudaf_ilayh` |

**Known issue**: This heuristic merges fāʿil, mubtadaʾ, khabar (all CASE=NOM). Fine-grained role separation is out of scope for v1.

### 3. I3rab Dependency Treebank (MSA i'rāb)

- **Source**: [Halabi, Fayyoumi & Awajan 2021](https://nlp.psut.edu.jo/malaac.html) — 600 MSA sentences, hand-annotated
- **Format**: CoNLL-like, 7 columns per token (idx, word, lemma, POS, irab_role, head, case)
- **Labels**: I'rāb only (no diacritics — we derive them via CAMeL morphology)
- **Availability**: **Not openly downloadable** — requires contacting the authors
- **Role in our pipeline**: MSA balance for the i'rāb head. QAC is Classical; I3rab is MSA.

**Access instructions** for researchers:
1. Email the authors citing the TALLIP 2021 paper
2. Typical response time ~1-2 weeks
3. Place the received file at `data/i3rab/i3rab.conllu`

Until access is secured, we ship a 3-sentence demo file in the parser tests.

### 4. Synthetic errors (our own generation)

- **Source**: Programmatic corruption of QAC gold text
- **Error types**:
  - `hamza_drop`: أ/إ/آ → ا
  - `taa_marbuta_swap`: ة → ه
  - `case_swap`: swap final diacritic to wrong case
- **Labels**: BIO spans marking the error character range + error type
- **Role**: Sole supervision for the error detection head

**Generator is in**: `src/irab_tashkeel/data/synthetic.py`

**Expansion ideas** (future work):
- Agreement errors (masculinize a feminine adjective)
- Missing sukūn on final consonants
- Wrong hamza direction (أ vs إ — orthographic rather than diacritic)
- Lām al-qamariyya vs lām al-shamsiyya mix-ups

## Unified schema

All four sources map to `MTLExample` (defined in `src/irab_tashkeel/data/schema.py`):

```python
@dataclass
class MTLExample:
    bare_text: str                        # undiacritized input
    diac_labels: list[int]                # per-char diacritic class
    mask_diac: bool                       # train diac head on this?
    word_offsets: list[tuple[int, int]]   # (start, end) per word in bare_text
    irab_labels: list[int]                # per-word i'rāb class
    mask_irab: bool                       # train irab head?
    err_labels: list[int]                 # per-char BIO error class
    mask_err: bool                        # train err head?
    source: str                           # "tashkeela" / "qac" / "i3rab" / "synth_hamza" / ...
    sent_id: str | None = None
```

The three masks are crucial — they ensure each head only gets supervision from samples with real labels. Spurious zero-labels would poison training.

## Dataset mixing — the ratio question

With `build_dataset.py` defaults:

| Source | Count (default) | % of total |
|---|---|---|
| Tashkeela | 30,000 | 58% |
| QAC (each verse = 1 sentence) | 6,236 | 12% |
| I3rab (when available) | 600 | 1% |
| Synthetic (3 error types × 2000 QAC seeds) | 6,000 | 11% |
| QAC reused with error injection | 9,000 | 17% |

Total: ~51,800 examples. Randomly shuffled; 90/10 train/val split.

**Tashkeela is undersampled** deliberately. With 75M words available, we could dominate training, but it would bias the model toward Classical at the expense of the smaller MSA-relevant corpora. The 30K sample is a starting point; tune based on downstream metrics.

## Preprocessing pipeline

```
raw text
  ↓
Unicode NFC normalization
  ↓
strip tatweel (U+0640)
  ↓
(for training-only) split into sentences on . ؟ !
  ↓
filter by length (30 ≤ len(chars) ≤ 500)
  ↓
character tokenize
  ↓
pad/truncate to max_len (256 or 512)
```

All of this is in `src/irab_tashkeel/data/preprocess.py`.

## Dataset caching

Parsing QAC from scratch takes ~30 seconds. Loading from HuggingFace streaming takes ~2 minutes for 30K samples. Both get cached to `data/cache/`:

- `data/cache/qac_sentences.pkl` — parsed QAC
- `data/cache/tashkeela_30k.txt` — downloaded Tashkeela text (one per line)
- `data/cache/combined_examples.pkl` — the final `list[MTLExample]` ready for training

Run `python scripts/download_data.py --all` once; cache makes subsequent training runs start in seconds.

## Test sets (the eval side)

We **never** train on these:

1. **WikiNews** (MSA benchmark, 2,500 sentences) — downloaded from CATT repo
2. **Fadel Tashkeela test set** (refined by Sadeed authors) — via HuggingFace
3. **I3rab held-out** (last 60 sentences of the 600)
4. **SadeedDiac-25** — Sadeed's new MSA+CA benchmark

All wired into `src/irab_tashkeel/evaluation/benchmark.py`.

## Edge cases you should know

- **Dagger alif (U+0670)** (the small superscript alif in قَـٰل) — we treat it as a separate character class, not a diacritic. It's structural in Classical Arabic.
- **Quranic extended characters** (U+06D6 through U+06ED — tajwīd markers) — stripped during Tashkeela-style preprocessing, kept when training on QAC directly.
- **Alef wasla (U+0671)** — treated as a first-class character (not normalized to alif).
- **Shadda without vowel** — rare but valid (e.g. at word end). Handled as its own diacritic class.
