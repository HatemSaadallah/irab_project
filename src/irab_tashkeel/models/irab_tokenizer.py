"""SentencePiece BPE tokenizer for the Arabic i'rab decoder targets.

The i'rab vocabulary is small and highly repetitive (a few hundred terms
combined into stock phrases), so BPE compresses it well at ~5k pieces.

Reserved IDs (set via SP user_defined_symbols / control tokens):
    0 -> <pad>     (used for padding decoder targets)
    1 -> <unk>
    2 -> <sos>     (decoder start token)
    3 -> <eos>     (decoder end-of-sequence)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List, Optional

import sentencepiece as spm


PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

_PAD = "<pad>"
_UNK = "<unk>"
_SOS = "<sos>"
_EOS = "<eos>"


class IrabTokenizer:
    """Wraps a trained SentencePieceProcessor with sos/eos handling."""

    def __init__(self, sp: spm.SentencePieceProcessor):
        self.sp = sp
        # Sanity-check special token IDs match our convention.
        assert sp.piece_to_id(_PAD) == PAD_ID, f"pad id mismatch: {sp.piece_to_id(_PAD)}"
        assert sp.piece_to_id(_UNK) == UNK_ID, f"unk id mismatch: {sp.piece_to_id(_UNK)}"
        assert sp.piece_to_id(_SOS) == SOS_ID, f"sos id mismatch: {sp.piece_to_id(_SOS)}"
        assert sp.piece_to_id(_EOS) == EOS_ID, f"eos id mismatch: {sp.piece_to_id(_EOS)}"

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode a string to token IDs. With add_special, prepends <sos> and appends <eos>."""
        ids = self.sp.encode(text, out_type=int)
        if add_special:
            ids = [SOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a sequence, stripping special tokens (sos, eos, pad)."""
        cleaned = [i for i in ids if i not in (PAD_ID, SOS_ID, EOS_ID)]
        return self.sp.decode(cleaned)

    @classmethod
    def load(cls, model_path: str | Path) -> "IrabTokenizer":
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        return cls(sp)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        model_path: str | Path,
        vocab_size: int = 5000,
        character_coverage: float = 1.0,
    ) -> "IrabTokenizer":
        """Train a new BPE model on the given iterable of strings, save to model_path."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Drop empty strings — SP rejects them.
        cleaned = [t for t in texts if t and t.strip()]
        if not cleaned:
            raise ValueError("Cannot train tokenizer on empty corpus")

        # Train into an in-memory model so we can write atomically.
        # `hard_vocab_limit=False` makes vocab_size a soft cap — i'rab targets
        # are highly repetitive so the natural BPE vocab is often smaller than
        # the requested ceiling.
        model_buf = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(cleaned),
            model_writer=model_buf,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=character_coverage,
            pad_id=PAD_ID,
            unk_id=UNK_ID,
            bos_id=SOS_ID,
            eos_id=EOS_ID,
            pad_piece=_PAD,
            unk_piece=_UNK,
            bos_piece=_SOS,
            eos_piece=_EOS,
            hard_vocab_limit=False,
            normalization_rule_name="identity",  # keep Arabic exactly as-is
        )
        model_path.write_bytes(model_buf.getvalue())

        return cls.load(model_path)


def train_from_examples(
    examples,
    model_path: str | Path = "data/irab_spm.model",
    vocab_size: int = 5000,
) -> IrabTokenizer:
    """Convenience: train a tokenizer from a list of MTLExample, using all non-empty irab_targets."""
    targets: List[str] = []
    for ex in examples:
        for t in (ex.irab_targets or []):
            if t:
                targets.append(t)
    return IrabTokenizer.train(targets, model_path=model_path, vocab_size=vocab_size)


def main():
    """CLI: build the dataset, train the tokenizer, save it."""
    import argparse

    from ..data.build_dataset import build_combined_dataset, load_examples, save_examples

    parser = argparse.ArgumentParser(description="Train the i'rab BPE tokenizer")
    parser.add_argument("--dataset-cache", default="data/cache/combined.pkl",
                        help="Use this cached dataset if it exists, else rebuild")
    parser.add_argument("--out", default="data/irab_spm.model")
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    cache = Path(args.dataset_cache)
    if not args.force_rebuild and cache.exists():
        print(f"Loading cached dataset from {cache}")
        examples = load_examples(cache)
    else:
        print("Building combined dataset …")
        examples = build_combined_dataset()
        save_examples(examples, cache)
        print(f"Cached dataset to {cache}")

    print(f"Training tokenizer (vocab={args.vocab_size}) …")
    tok = train_from_examples(examples, model_path=args.out, vocab_size=args.vocab_size)
    print(f"Saved tokenizer to {args.out}  (vocab_size={tok.vocab_size})")

    # Sanity demo
    sample = "فعل مضارع مرفوع وعلامة رفعه الضمة الظاهرة"
    ids = tok.encode(sample)
    print(f"Sample encode: {sample}")
    print(f"  ids ({len(ids)}): {ids[:20]}…")
    print(f"  decode: {tok.decode(ids)}")


if __name__ == "__main__":
    main()
