"""Benchmark runner — evaluate a trained model on held-out data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from ..data.i3rab import load_i3rab_examples
from ..data.qac import load_qac_examples
from ..data.tashkeela import load_tashkeela_sentences
from ..inference.predictor import Predictor
from ..models.labels import IRAB_LABELS
from ..models.tokenizer import decode_diacritized, strip_diacritics
from .metrics import der, irab_accuracy, irab_per_class_f1, summary, wer_diac


@dataclass
class BenchmarkResult:
    name: str
    n_samples: int
    metrics: Dict
    examples_shown: List[Dict]


def benchmark_diacritization_on_sentences(
    predictor: Predictor,
    gold_sentences: List[str],
    name: str = "diac",
    n_examples_to_show: int = 3,
) -> BenchmarkResult:
    """Benchmark on a list of already-diacritized gold sentences."""
    der_list: List[float] = []
    wer_list: List[float] = []
    examples = []

    for sent in tqdm(gold_sentences, desc=f"Evaluating {name}"):
        bare = strip_diacritics(sent)
        try:
            pred = predictor.predict(bare)
            d = der(pred.diacritized, sent)
            w = wer_diac(pred.diacritized, sent)
            der_list.append(d)
            wer_list.append(w)
            if len(examples) < n_examples_to_show:
                examples.append({
                    "gold": sent,
                    "predicted": pred.diacritized,
                    "der": d,
                    "wer": w,
                })
        except Exception as e:
            der_list.append(1.0)
            wer_list.append(1.0)
            continue

    return BenchmarkResult(
        name=name,
        n_samples=len(der_list),
        metrics=summary(der_list, wer_list),
        examples_shown=examples,
    )


def benchmark_irab_on_i3rab(
    predictor: Predictor,
    i3rab_path: Optional[Path] = None,
    holdout_fraction: float = 0.1,
) -> BenchmarkResult:
    """Benchmark i'rab role accuracy on the I3rab Treebank held-out split."""
    examples = load_i3rab_examples(i3rab_path)
    n_holdout = max(1, int(len(examples) * holdout_fraction))
    holdout = examples[-n_holdout:]

    all_pred: List[str] = []
    all_gold: List[str] = []
    shown = []

    for ex in tqdm(holdout, desc="Evaluating i'rab"):
        try:
            pred = predictor.predict(ex.bare_text)
            gold_roles = [IRAB_LABELS[i] for i in ex.irab_labels]
            pred_roles = [w["role"] for w in pred.words]
            # Only consider aligned positions
            n = min(len(gold_roles), len(pred_roles))
            all_pred.extend(pred_roles[:n])
            all_gold.extend(gold_roles[:n])
            if len(shown) < 3:
                shown.append({
                    "text": ex.bare_text,
                    "gold_roles": gold_roles,
                    "pred_roles": pred_roles,
                })
        except Exception:
            continue

    metrics = {
        "accuracy": irab_accuracy(all_pred, all_gold),
        "n_words": len(all_gold),
        "per_class_f1": irab_per_class_f1(all_pred, all_gold),
    }
    return BenchmarkResult(
        name="i3rab_irab", n_samples=len(holdout), metrics=metrics, examples_shown=shown,
    )


def run_full_benchmark(
    predictor: Predictor,
    tashkeela_n: int = 500,
    qac_holdout: int = 200,
    i3rab_path: Optional[Path] = None,
) -> Dict[str, BenchmarkResult]:
    """Run all benchmarks we have data for."""
    results = {}

    # Diacritization on Tashkeela sample
    try:
        tk = load_tashkeela_sentences(max_sentences=tashkeela_n)
        if tk:
            results["tashkeela"] = benchmark_diacritization_on_sentences(
                predictor, tk, name="tashkeela"
            )
    except Exception as e:
        print(f"Skipped Tashkeela: {e}")

    # Diacritization on QAC held-out verses
    try:
        qac_examples = load_qac_examples(Path("data/quran-morphology.txt"))
        # Use the last N verses as holdout
        qac_holdout_sents = []
        for ex in qac_examples[-qac_holdout:]:
            if ex.mask_diac:
                diac = decode_diacritized(ex.bare_text, ex.diac_labels)
                qac_holdout_sents.append(diac)
        if qac_holdout_sents:
            results["qac"] = benchmark_diacritization_on_sentences(
                predictor, qac_holdout_sents, name="qac_holdout"
            )
    except Exception as e:
        print(f"Skipped QAC: {e}")

    # I'rab accuracy on I3rab
    try:
        results["i3rab"] = benchmark_irab_on_i3rab(predictor, i3rab_path=i3rab_path)
    except Exception as e:
        print(f"Skipped I3rab: {e}")

    return results
