"""Chemistry-aware BPE tokenizer with shared vocab for pSMILES + pSELFIES.

Pre-tokenization:
  - pSMILES: chemistry-aware regex (bracket atoms, two-letter elements, bonds, ring indices)
  - pSELFIES: selfies.split_selfies() (each bracket token is one atomic unit)

BPE merges are then applied on top of these atomic units.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import selfies as sf

CACHE = Path("/mnt/data/p1m_pretrain_experiments/cache")
VOCAB_PATH = CACHE / "chem_bpe_vocab.json"
MERGES_PATH = CACHE / "chem_bpe_merges.json"
SHORT_VOCAB_PATH = CACHE / "chem_bpe_short_vocab.json"
SHORT_MERGES_PATH = CACHE / "chem_bpe_short_merges.json"
SHORT_DEFAULT_MAX_MERGES = 50

SMILES_PREFIX = "<SMILES>"
SELFIES_PREFIX = "<SELFIES>"

_SMILES_PATTERN = re.compile(
    r"("
    r"\[[^\]]+\]"
    r"|%[0-9]{2}"
    r"|Br|Cl|Si|Se|Ge|As|Te|se|si|ge"
    r"|[BCNOPSFIbcnops]"
    r"|[0-9]"
    r"|[=\#\-\+\\\/@\.\(\)\*\$\^:~]"
    r")"
)


def _pretokenize_smiles(text: str) -> list[str]:
    return _SMILES_PATTERN.findall(text)


def _pretokenize_selfies(text: str) -> list[str]:
    try:
        return list(sf.split_selfies(text))
    except Exception:
        return []


def _is_selfies(text: str) -> bool:
    return "[Branch" in text or "[Ring" in text or (text.startswith("[") and "][" in text)


class ChemBPETokenizer:
    """HuggingFace-compatible tokenizer using chemistry-aware pre-tokenization + BPE.

    Matches the interface of ChemSmilesTokenizer so it can be used as a drop-in
    replacement in the training and evaluation pipelines.
    """

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]], max_len: int = 514):
        self.vocab = vocab
        self.id_to_token_map = {v: k for k, v in vocab.items()}
        self._max_len = max_len
        self._merges = merges
        self._merge_set = {(a, b): i for i, (a, b) in enumerate(merges)}

    @classmethod
    def load(cls, vocab_path: str | Path | None = None, merges_path: str | Path | None = None,
             max_len: int = 514) -> "ChemBPETokenizer":
        vp = Path(vocab_path) if vocab_path else VOCAB_PATH
        mp = Path(merges_path) if merges_path else MERGES_PATH
        vocab = json.loads(vp.read_text())
        raw_merges = json.loads(mp.read_text())
        merges = [(a, b) for a, b in raw_merges]
        return cls(vocab, merges, max_len=max_len)

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

    def _apply_bpe(self, tokens: list[str]) -> list[str]:
        """Apply BPE merges greedily to a list of atomic tokens."""
        if len(tokens) <= 1:
            return tokens
        while True:
            best_pair = None
            best_rank = len(self._merges)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_set.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            if len(tokens) <= 1:
                break
        return tokens

    def tokenize(self, text: str) -> list[str]:
        """Pre-tokenize + BPE merge."""
        if _is_selfies(text):
            atomic = _pretokenize_selfies(text)
        else:
            atomic = _pretokenize_smiles(text)
        return self._apply_bpe(atomic)

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int | None = None) -> list[int]:
        max_length = max_length or self._max_len
        bpe_tokens = self.tokenize(text)
        ids = [self.vocab.get(t, self.unk_token_id) for t in bpe_tokens]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids[:max_length - 2] + [self.eos_token_id]
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
            pad_to = max_length if padding == "max_length" else max(len(ids) for ids in all_ids)
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


class ShortChemBPETokenizer(ChemBPETokenizer):
    """ChemBPE tokenizer with a truncated merge list for shorter, MLM-viable sequences."""

    @classmethod
    def load(
        cls,
        vocab_path: str | Path | None = None,
        merges_path: str | Path | None = None,
        max_len: int = 514,
    ) -> "ShortChemBPETokenizer":
        return super().load(
            vocab_path=vocab_path or SHORT_VOCAB_PATH,
            merges_path=merges_path or SHORT_MERGES_PATH,
            max_len=max_len,
        )


class ChemBPERepVocab:
    """RepresentationVocab-compatible wrapper for translation decoder targets.

    Uses the same BPE tokenizer for both encoder input and decoder output,
    matching the pattern of SmiTedRepVocab.
    """

    def __init__(self, tokenizer: ChemBPETokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def bos_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def unk_id(self) -> int:
        return self.tokenizer.unk_token_id

    @property
    def mask_id(self) -> int:
        return self.tokenizer.mask_token_id

    @property
    def size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str, rep_type: str, max_length: int) -> list[int]:
        if rep_type == "pselfies":
            atomic = _pretokenize_selfies(text)
        else:
            atomic = _pretokenize_smiles(text)
        bpe_tokens = self.tokenizer._apply_bpe(atomic)
        ids = [self.bos_id]
        for tok in bpe_tokens[:max_length - 2]:
            ids.append(self.vocab.get(tok, self.unk_id))
        ids.append(self.eos_id)
        if len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids[:max_length]
