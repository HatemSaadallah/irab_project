"""Evaluation CLI: `python -m irab_tashkeel.evaluation.eval_cli --checkpoint runs/medium/best.pt`."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..inference.predictor import Predictor
from .benchmark import run_full_benchmark


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained i'rab+tashkeel model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained model checkpoint")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Where to write results JSON")
    parser.add_argument("--tashkeela-n", type=int, default=500)
    parser.add_argument("--qac-holdout", type=int, default=200)
    parser.add_argument("--i3rab-path", type=str, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} …")
    predictor = Predictor.from_checkpoint(
        args.checkpoint, confidence_threshold=args.confidence_threshold
    )
    print(f"Model: {predictor.model.n_params() / 1e6:.1f}M params")

    print("\nRunning benchmark suite …")
    results = run_full_benchmark(
        predictor,
        tashkeela_n=args.tashkeela_n,
        qac_holdout=args.qac_holdout,
        i3rab_path=Path(args.i3rab_path) if args.i3rab_path else None,
    )

    # Pretty print
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    summary = {}
    for name, r in results.items():
        print(f"\n[{name}]  n={r.n_samples}")
        if "mean_der" in r.metrics:
            print(f"  mean DER:  {r.metrics['mean_der']*100:.2f}%")
            print(f"  median:    {r.metrics['median_der']*100:.2f}%")
            print(f"  mean WER:  {r.metrics['mean_wer']*100:.2f}%")
            print(f"  catastrophic (DER>0.5): {r.metrics['catastrophic_rate']*100:.1f}%")
        if "accuracy" in r.metrics:
            print(f"  i'rab accuracy: {r.metrics['accuracy']*100:.2f}%")
            print(f"  n_words: {r.metrics['n_words']}")
        summary[name] = {
            "n_samples": r.n_samples,
            "metrics": r.metrics,
            "examples": r.examples_shown[:3],
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
