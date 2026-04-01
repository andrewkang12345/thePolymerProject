"""Character-level SMILES tokenizer with chemically-aware splitting.

Fixes over PolymerSmilesTokenizer:
  - Bracket atoms kept as single tokens: [NH2], [C@@H], [nH], [At]
  - Two-digit ring closures: %10, %11, ...
  - Single-digit ring numbers as own tokens
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Chemically-aware SMILES regex
# Order matters: longer patterns first
_SMILES_PATTERN = re.compile(
    r"("
    r"\[[^\]]+\]"          # bracket atoms: [NH2], [C@@H], [nH], [At], etc.
    r"|%[0-9]{2}"          # two-digit ring closures: %10, %11, ...
    r"|Br|Cl"              # two-letter elements
    r"|[BCNOPSFIbcnops]"   # single-letter organic subset + aromatic
    r"|[0-9]"              # single-digit ring closures
    r"|[=\#\-\+\\\/@\.\(\)\*\$\^]"  # bonds, branches, wildcards
    r")"
)

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]


def tokenize_smiles(text: str) -> list[str]:
    return _SMILES_PATTERN.findall(text)


def build_vocab(smiles_iter, min_freq: int = 1) -> dict[str, int]:
    """Build vocabulary from an iterable of SMILES strings."""
    from collections import Counter
    counts: Counter[str] = Counter()
    for smi in smiles_iter:
        counts.update(tokenize_smiles(str(smi)))
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    idx = len(SPECIAL_TOKENS)
    for tok, freq in sorted(counts.items()):
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = idx
            idx += 1
    return vocab


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    Path(path).write_text(json.dumps(vocab, indent=2, ensure_ascii=False))


def load_vocab(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text())


class ChemSmilesTokenizer:
    """HuggingFace-compatible tokenizer using chemically-aware SMILES splitting."""

    def __init__(self, vocab: dict[str, int], max_len: int = 256):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self._max_len = max_len

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.vocab["<s>"]

    @property
    def eos_token_id(self) -> int:
        return self.vocab["</s>"]

    @property
    def unk_token_id(self) -> int:
        return self.vocab["<unk>"]

    @property
    def mask_token_id(self) -> int:
        return self.vocab["<mask>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int | None = None) -> list[int]:
        max_length = max_length or self._max_len
        tokens = tokenize_smiles(text)
        ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids[: max_length - 2] + [self.eos_token_id]
        else:
            ids = ids[:max_length]
        return ids

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        return_tensors: str | None = None,
        **kwargs,
    ):
        import torch

        if isinstance(text, str):
            text = [text]
        max_length = max_length or self._max_len

        all_ids = [self.encode(t, add_special_tokens=add_special_tokens, max_length=max_length) for t in text]

        if padding is True or padding == "longest" or padding == "max_length":
            if padding == "max_length":
                pad_to = max_length
            else:
                pad_to = max(len(ids) for ids in all_ids)
            attention_masks = []
            for i, ids in enumerate(all_ids):
                mask = [1] * len(ids) + [0] * (pad_to - len(ids))
                all_ids[i] = ids + [self.pad_token_id] * (pad_to - len(ids))
                attention_masks.append(mask)
        else:
            attention_masks = [[1] * len(ids) for ids in all_ids]

        result = {"input_ids": all_ids}
        if return_attention_mask:
            result["attention_mask"] = attention_masks

        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}

        return result

    def save(self, path: str | Path) -> None:
        save_vocab(self.vocab, path)

    @classmethod
    def load(cls, path: str | Path, max_len: int = 256) -> "ChemSmilesTokenizer":
        return cls(load_vocab(path), max_len=max_len)

    @classmethod
    def from_smiles(cls, smiles_iter, max_len: int = 256, min_freq: int = 1) -> "ChemSmilesTokenizer":
        vocab = build_vocab(smiles_iter, min_freq=min_freq)
        return cls(vocab, max_len=max_len)
