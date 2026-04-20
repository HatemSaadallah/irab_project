"""PyTorch Dataset and collate function for MTLExample lists."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..data.schema import MTLExample
from ..models.irab_tokenizer import IrabTokenizer, PAD_ID as IRAB_TOK_PAD_ID
from ..models.labels import IRAB_PAD_ID, PAD_ID
from ..models.tokenizer import encode_chars


class MTLDataset(Dataset):
    """Wraps a list of MTLExample objects for PyTorch.

    __getitem__ returns a dict with tensors and the raw word_offsets (which
    stay as a list since they're ragged).
    """

    def __init__(
        self,
        examples: List[MTLExample],
        max_len: int = 512,
        irab_tokenizer: Optional[IrabTokenizer] = None,
        max_irab_target_len: int = 64,
    ):
        self.examples = examples
        self.max_len = max_len
        self.irab_tokenizer = irab_tokenizer
        self.max_irab_target_len = max_irab_target_len

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_irab_targets(self, targets: List[str]) -> List[List[int]]:
        """Encode each word's i'rab string to BPE ids (truncated to max_irab_target_len).

        Empty targets become empty lists (no supervision); collate fills them with PAD_ID.
        """
        if self.irab_tokenizer is None:
            return [[] for _ in targets]
        out: List[List[int]] = []
        for t in targets:
            if not t:
                out.append([])
                continue
            ids = self.irab_tokenizer.encode(t, add_special=True)
            ids = ids[: self.max_irab_target_len]
            out.append(ids)
        return out

    def __getitem__(self, idx: int) -> Dict:
        e = self.examples[idx]

        char_ids, attn_mask = encode_chars(e.bare_text, max_len=self.max_len)

        # Pad/truncate diacritic labels to max_len
        diac = list(e.diac_labels[: self.max_len])
        diac.extend([0] * (self.max_len - len(diac)))

        # Pad/truncate error labels to max_len
        err = list(e.err_labels[: self.max_len])
        err.extend([0] * (self.max_len - len(err)))

        # Filter word offsets to those fully within max_len
        offsets = [(s, min(end, self.max_len)) for s, end in e.word_offsets if s < self.max_len]
        irab = list(e.irab_labels[: len(offsets)])
        irab_targets = list((e.irab_targets or [""] * len(e.word_offsets))[: len(offsets)])
        irab_target_ids = self._encode_irab_targets(irab_targets)

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "diac_labels": torch.tensor(diac, dtype=torch.long),
            "err_labels": torch.tensor(err, dtype=torch.long),
            "word_offsets": offsets,
            "irab_labels": irab,
            "irab_target_ids": irab_target_ids,
            "mask_diac": e.mask_diac,
            "mask_irab": e.mask_irab,
            "mask_err": e.mask_err,
            "source": e.source,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of dict items into a single batch dict.

    Ragged fields (word_offsets, irab_labels, irab_target_ids) are padded to the
    batch's max word count and max target length.
    """
    B = len(batch)
    max_words = max(len(b["word_offsets"]) for b in batch) if batch else 1
    max_words = max(max_words, 1)

    # Per-word coarse irab labels: (B, W_max), pad with IRAB_PAD_ID.
    irab_padded = torch.full((B, max_words), IRAB_PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        for j, lab in enumerate(b["irab_labels"]):
            irab_padded[i, j] = lab

    # Per-word irab seq2seq targets: (B, W_max, T_max), pad with PAD_ID.
    max_t = 1
    for b in batch:
        for ids in b["irab_target_ids"]:
            if len(ids) > max_t:
                max_t = len(ids)
    # Need at least 2 positions (so the decoder has a non-empty input/label split).
    max_t = max(max_t, 2)
    irab_seq = torch.full((B, max_words, max_t), IRAB_TOK_PAD_ID, dtype=torch.long)
    irab_seq_word_mask = torch.zeros(B, max_words, dtype=torch.bool)
    for i, b in enumerate(batch):
        for j, ids in enumerate(b["irab_target_ids"]):
            if not ids:
                continue
            t = len(ids)
            irab_seq[i, j, :t] = torch.tensor(ids, dtype=torch.long)
            irab_seq_word_mask[i, j] = True

    return {
        "char_ids":           torch.stack([b["char_ids"] for b in batch]),
        "attention_mask":     torch.stack([b["attention_mask"] for b in batch]),
        "diac_labels":        torch.stack([b["diac_labels"] for b in batch]),
        "err_labels":         torch.stack([b["err_labels"] for b in batch]),
        "irab_labels":        irab_padded,
        "irab_target_ids":    irab_seq,                # (B, W_max, T_max)
        "irab_seq_word_mask": irab_seq_word_mask,      # (B, W_max) — True for words with seq supervision
        "word_offsets":       [b["word_offsets"] for b in batch],
        "mask_diac":          torch.tensor([b["mask_diac"] for b in batch], dtype=torch.bool),
        "mask_irab":          torch.tensor([b["mask_irab"] for b in batch], dtype=torch.bool),
        "mask_err":           torch.tensor([b["mask_err"]  for b in batch], dtype=torch.bool),
        "sources":            [b["source"] for b in batch],
    }
