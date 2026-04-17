"""Download all corpora needed for training.

Usage:
    python scripts/download_data.py --all
    python scripts/download_data.py --qac
    python scripts/download_data.py --tashkeela --max-sentences 10000
"""

import argparse
from pathlib import Path

from irab_tashkeel.data.qac import download_qac
from irab_tashkeel.data.tashkeela import load_tashkeela_sentences


def cmd_qac(args):
    path = Path(args.data_dir) / "quran-morphology.txt"
    if path.exists() and not args.force:
        print(f"QAC already present at {path} ({path.stat().st_size / 1024:.1f} KB)")
        return
    print(f"Downloading QAC to {path} …")
    download_qac(path)
    print(f"✓ {path} ({path.stat().st_size / 1024:.1f} KB)")


def cmd_tashkeela(args):
    out_path = Path(args.data_dir) / f"tashkeela_{args.max_sentences}.txt"
    if out_path.exists() and not args.force:
        with open(out_path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"Tashkeela already present at {out_path} ({n} sentences)")
        return

    print(f"Loading Tashkeela (up to {args.max_sentences} sentences) …")
    sentences = load_tashkeela_sentences(max_sentences=args.max_sentences)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"✓ {out_path} ({len(sentences)} sentences)")


def cmd_i3rab(args):
    print("I3rab Treebank is not publicly downloadable.")
    print("Contact the authors: https://nlp.psut.edu.jo/malaac.html")
    print(f"Once you receive it, place it at {args.data_dir}/i3rab/i3rab.conllu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--qac", action="store_true")
    parser.add_argument("--tashkeela", action="store_true")
    parser.add_argument("--i3rab", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-sentences", type=int, default=30000)
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if args.all:
        args.qac = args.tashkeela = args.i3rab = True

    if not any([args.qac, args.tashkeela, args.i3rab]):
        parser.print_help()
        return

    if args.qac:
        cmd_qac(args)
    if args.tashkeela:
        cmd_tashkeela(args)
    if args.i3rab:
        cmd_i3rab(args)


if __name__ == "__main__":
    main()
