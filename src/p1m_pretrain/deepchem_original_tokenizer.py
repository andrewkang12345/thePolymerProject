from __future__ import annotations

import collections
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizer


# Vendored from DeepChem's smiles_tokenizer.py (2.8.0) with only packaging-level
# adjustments for this repo.
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


def load_vocab(vocab_file: str):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class BasicSmilesTokenizer:
    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> list[str]:
        return [token for token in self.regex.findall(text)]


class SmilesTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.txt"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str = "", **kwargs):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocab file at path '{vocab_file}'.")
        self.vocab_file = vocab_file
        self._dc_vocab = load_vocab(vocab_file)
        self.highest_unused_index = max(
            i for i, token in enumerate(self._dc_vocab.keys()) if token.startswith("[unused")
        )
        self.ids_to_tokens = collections.OrderedDict((ids, tok) for tok, ids in self._dc_vocab.items())
        self.basic_tokenizer = BasicSmilesTokenizer()
        super().__init__(
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self._dc_vocab)

    @property
    def vocab_list(self):
        return list(self._dc_vocab.keys())

    @property
    def vocab(self):
        return self._dc_vocab

    def get_vocab(self):
        return dict(self._dc_vocab, **self.added_tokens_encoder)

    def _tokenize(self, text: str, **kwargs):
        return self.basic_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token: str):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        return " ".join(tokens).replace(" ##", "").strip()

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[Optional[int]],
        token_ids_1: Optional[List[Optional[int]]] = None,
    ) -> List[Optional[int]]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[Optional[int]],
        token_ids_1: Optional[List[Optional[int]]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return [1 if token in {self.cls_token_id, self.sep_token_id} else 0 for token in token_ids_0]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[Optional[int]],
        token_ids_1: Optional[List[Optional[int]]] = None,
    ) -> List[int]:
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = "vocab.txt" if filename_prefix is None else f"{filename_prefix}-vocab.txt"
        out_path = save_dir / filename
        out_path.write_text("\n".join(self.vocab_list) + "\n", encoding="utf-8")
        return (str(out_path),)
