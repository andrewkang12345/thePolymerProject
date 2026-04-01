"""Clean SMI-TED tokenizer without BertTokenizer inheritance issues."""
from __future__ import annotations

import re
from pathlib import Path

import torch

# SMI-TED's exact SMILES regex
PATTERN = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")


class SmiTedTokenizer:
    """Standalone tokenizer matching SMI-TED's MolTranBertTokenizer behavior."""

    def __init__(self, vocab_file: str, max_len: int = 202):
        self.max_len = max_len
        # Load vocab
        with open(vocab_file) as f:
            tokens = [line.strip() for line in f.readlines()]
        self.vocab = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = tokens

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.vocab["<eos>"]

    @property
    def mask_token_id(self) -> int:
        return self.vocab["<mask>"]

    @property
    def unk_token_id(self) -> int:
        return self.vocab.get("<unk>", self.pad_token_id)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> list[str]:
        return PATTERN.findall(text)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] if i < len(self.id_to_token) else "<unk>" for i in ids]

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int | None = None) -> list[int]:
        max_length = max_length or self.max_len
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
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
        if isinstance(text, str):
            text = [text]
        max_length = max_length or self.max_len

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

    @classmethod
    def from_pretrained(cls, vocab_path: str, max_len: int = 202) -> "SmiTedTokenizer":
        return cls(vocab_path, max_len=max_len)
