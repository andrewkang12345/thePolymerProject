"""SMI-TED with extended vocab for SELFIES + representation prefixes."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import selfies as sf

import selfies as sfl

from .smi_ted_tokenizer import SmiTedTokenizer, PATTERN

SMI_TED_DIR = Path("/mnt/data/p1m_pretrain_experiments/checkpoints/smi_ted")
PI1M_CACHE = Path("/mnt/data/p1m_pretrain_experiments/cache/pi1m_train_dedup.parquet")
SELFIES_TOKENS_CACHE = Path("/mnt/data/p1m_pretrain_experiments/cache/smi_ted_selfies_tokens.json")

SMILES_PREFIX = "<SMILES>"
SELFIES_PREFIX = "<SELFIES>"


def _collect_selfies_tokens() -> list[str]:
    if SELFIES_TOKENS_CACHE.exists():
        return json.loads(SELFIES_TOKENS_CACHE.read_text())
    df = pd.read_parquet(PI1M_CACHE)
    tokens = set()
    for s in df["pselfies"].dropna().tolist():
        tokens.update(sf.split_selfies(s))
    # Only keep tokens NOT already in SMI-TED vocab
    with open(SMI_TED_DIR / "bert_vocab_curated.txt") as f:
        existing = set(line.strip() for line in f)
    new_tokens = sorted(tokens - existing)
    SELFIES_TOKENS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SELFIES_TOKENS_CACHE.write_text(json.dumps(new_tokens))
    return new_tokens


def build_extended_smi_ted_tokenizer(max_len: int = 202) -> SmiTedTokenizer:
    """Build SMI-TED tokenizer extended with SELFIES tokens and prefixes."""
    vocab_path = SMI_TED_DIR / "bert_vocab_curated.txt"
    with open(vocab_path) as f:
        base_tokens = [line.strip() for line in f]

    # Add prefixes then SELFIES tokens
    new_selfies = _collect_selfies_tokens()
    all_tokens = base_tokens + [SMILES_PREFIX, SELFIES_PREFIX] + new_selfies

    # Build vocab dict
    vocab = {tok: i for i, tok in enumerate(all_tokens)}
    tok = SmiTedTokenizer.__new__(SmiTedTokenizer)
    tok.max_len = max_len
    tok.vocab = vocab
    tok.id_to_token = all_tokens
    return tok


class SmiTedRepVocab:
    """RepresentationVocab-compatible wrapper around extended SMI-TED tokenizer.

    Used for translation decoder targets. Tokenizes pSMILES with the SMILES regex
    and pSELFIES with selfies.split_selfies(), then maps to the shared vocab IDs.
    """

    def __init__(self, tokenizer: SmiTedTokenizer):
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
        return self.tokenizer.vocab.get("<eos>", 1)

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
        """Encode text to IDs using appropriate tokenization for the representation type."""
        if rep_type == "pselfies":
            tokens = list(sfl.split_selfies(text))
        else:  # psmiles
            tokens = PATTERN.findall(text)
        ids = [self.bos_id]
        for tok in tokens[:max_length - 2]:
            ids.append(self.vocab.get(tok, self.unk_id))
        ids.append(self.eos_id)
        if len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids[:max_length]


def load_smi_ted_extended():
    """Load SMI-TED model with extended vocab."""
    from .smi_ted_wrapper import load_smi_ted_for_mlm
    model = load_smi_ted_for_mlm()
    tok = build_extended_smi_ted_tokenizer()
    model.resize_token_embeddings(len(tok))
    return model, tok


def load_smi_ted_scratch_extended():
    """Build scratch SMI-TED model with extended vocab (random init, no pretrained weights)."""
    from .smi_ted_wrapper import build_smi_ted_scratch
    model = build_smi_ted_scratch()
    tok = build_extended_smi_ted_tokenizer()
    model.resize_token_embeddings(len(tok))
    return model, tok
