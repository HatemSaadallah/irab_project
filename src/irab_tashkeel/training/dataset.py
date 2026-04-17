"""PyTorch Dataset and collate function for MTLExample lists."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..data.schema import MTLExample
from ..models.labels import IRAB_PAD_ID, PAD_ID
from ..models.tokenizer import encode_chars


class MTLDataset(Dataset):
    """Wraps a list of MTLExample objects for PyTorch.

    __getitem__ returns a dict with tensors and the raw word_offsets (which
    stay as a list since they're ragged).
    """

    def __init__(self, examples: List[MTLExample], max_len: int = 512):
        self.examples = examples
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

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

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "diac_labels": torch.tensor(diac, dtype=torch.long),
            "err_labels": torch.tensor(err, dtype=torch.long),
            "word_offsets": offsets,
            "irab_labels": irab,
            "mask_diac": e.mask_diac,
            "mask_irab": e.mask_irab,
            "mask_err": e.mask_err,
            "source": e.source,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of dict items into a single batch dict.

    Ragged fields (word_offsets, irab_labels) are padded to the batch's max word count.
    """
    B = len(batch)
    max_words = max(len(b["word_offsets"]) for b in batch) if batch else 1
    max_words = max(max_words, 1)

    # Pad irab_labels to (B, max_words) with IRAB_PAD_ID
    irab_padded = torch.full((B, max_words), IRAB_PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["irab_labels"])
        for j in range(n):
            irab_padded[i, j] = b["irab_labels"][j]

    return {
        "char_ids":       torch.stack([b["char_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "diac_labels":    torch.stack([b["diac_labels"] for b in batch]),
        "err_labels":     torch.stack([b["err_labels"] for b in batch]),
        "irab_labels":    irab_padded,
        "word_offsets":   [b["word_offsets"] for b in batch],
        "mask_diac":      torch.tensor([b["mask_diac"] for b in batch], dtype=torch.bool),
        "mask_irab":      torch.tensor([b["mask_irab"] for b in batch], dtype=torch.bool),
        "mask_err":       torch.tensor([b["mask_err"]  for b in batch], dtype=torch.bool),
        "sources":        [b["source"] for b in batch],
    }
