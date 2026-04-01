from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor
import json
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

import selfies as sf

from .pselfies import (
    RepresentationVocab,
    canonical_proxy_smiles_from_psmiles,
    proxy_pselfies_from_psmiles,
    randomize_psmiles,
)
from .paths import get_paths


PATHS = get_paths()
PI1M_CSV = PATHS.pi1m_csv
EXTERNAL_JSONL_SPECS: tuple[tuple[str, str, str], ...] = PATHS.external_jsonl_specs


@dataclass
class P1MRecord:
    psmiles: str
    pselfies: str
    bigsmiles: str | None = None
    sa_score: float | None = None
    source_name: str | None = None


def _convert_polymer_record(psmiles: str) -> tuple[str | None, str | None]:
    canonical_proxy_smiles = canonical_proxy_smiles_from_psmiles(psmiles)
    pselfies = proxy_pselfies_from_psmiles(psmiles)
    return canonical_proxy_smiles, pselfies


def _assign_split(split_key: str, seed: int, val_fraction: float) -> str:
    payload = f"{seed}:{split_key}".encode("utf-8")
    bucket = int(hashlib.sha256(payload).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


def _hash_rank(key: str, seed: int) -> int:
    payload = f"{seed}:{key}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def prepare_clean_split(
    cache_path: Path,
    *,
    seed: int,
    val_fraction: float,
    preprocess_workers: int = 8,
) -> Path:
    if cache_path.exists():
        return cache_path

    df = pd.read_csv(PI1M_CSV, usecols=["SMILES", "SA Score"])
    df = df.rename(columns={"SMILES": "psmiles", "SA Score": "sa_score"})
    psmiles_values = df["psmiles"].astype(str).tolist()
    workers = max(1, min(preprocess_workers, os.cpu_count() or 1))
    if workers == 1:
        converted = [_convert_polymer_record(value) for value in psmiles_values]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            converted = list(executor.map(_convert_polymer_record, psmiles_values, chunksize=512))
    df["canonical_proxy_smiles"] = [item[0] for item in converted]
    df["pselfies"] = [item[1] for item in converted]
    df = df.dropna(subset=["canonical_proxy_smiles", "pselfies"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["canonical_proxy_smiles"]).reset_index(drop=True)
    df["split_key"] = df["canonical_proxy_smiles"]
    df["split"] = [_assign_split(key, seed=seed, val_fraction=val_fraction) for key in df["split_key"]]
    df = df.sort_values(["split", "split_key"]).reset_index(drop=True)
    sampled = df[["psmiles", "pselfies", "sa_score", "canonical_proxy_smiles", "split_key", "split"]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(cache_path, index=False)
    return cache_path


def prepare_pi1m_train_cache(
    cache_path: Path,
    *,
    preprocess_workers: int = 8,
) -> Path:
    if cache_path.exists():
        return cache_path

    df = pd.read_csv(PI1M_CSV, usecols=["SMILES", "SA Score"])
    df = df.rename(columns={"SMILES": "psmiles", "SA Score": "sa_score"})
    psmiles_values = df["psmiles"].astype(str).tolist()
    workers = max(1, min(preprocess_workers, os.cpu_count() or 1))
    if workers == 1:
        converted = [_convert_polymer_record(value) for value in psmiles_values]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            converted = list(executor.map(_convert_polymer_record, psmiles_values, chunksize=512))
    df["canonical_proxy_smiles"] = [item[0] for item in converted]
    df["pselfies"] = [item[1] for item in converted]
    df = df.dropna(subset=["canonical_proxy_smiles", "pselfies"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["canonical_proxy_smiles"]).reset_index(drop=True)
    df["source_name"] = "pi1m"
    df = df[["psmiles", "pselfies", "sa_score", "canonical_proxy_smiles", "source_name"]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return cache_path


def _iter_external_psmiles(specs: Iterable[tuple[str, str, str]]) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for file_path, source_name, field_name in specs:
        path = Path(file_path)
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                payload = json.loads(line)
                value = payload.get(field_name)
                if value:
                    records.append((source_name, str(value)))
    return records


def prepare_external_val_cache(
    cache_path: Path,
    *,
    pi1m_train_cache_path: Path,
    seed: int,
    val_size: int,
    preprocess_workers: int = 8,
    external_specs: Iterable[tuple[str, str, str]] = EXTERNAL_JSONL_SPECS,
) -> Path:
    if cache_path.exists():
        return cache_path

    pi1m_df = pd.read_parquet(pi1m_train_cache_path, columns=["canonical_proxy_smiles"])
    pi1m_keys = set(pi1m_df["canonical_proxy_smiles"].astype(str).tolist())

    external_records = _iter_external_psmiles(external_specs)
    psmiles_values = [value for _, value in external_records]
    workers = max(1, min(preprocess_workers, os.cpu_count() or 1))
    if workers == 1:
        converted = [_convert_polymer_record(value) for value in psmiles_values]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            converted = list(executor.map(_convert_polymer_record, psmiles_values, chunksize=256))

    rows = []
    for (source_name, psmiles), (canonical_proxy_smiles, pselfies) in zip(external_records, converted):
        if canonical_proxy_smiles is None or pselfies is None:
            continue
        if canonical_proxy_smiles in pi1m_keys:
            continue
        rows.append(
            {
                "psmiles": psmiles,
                "pselfies": pselfies,
                "canonical_proxy_smiles": canonical_proxy_smiles,
                "source_name": source_name,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("External validation cache is empty after PI1M deduplication.")
    df = df.drop_duplicates(subset=["canonical_proxy_smiles"]).reset_index(drop=True)
    df["rank"] = [_hash_rank(key, seed=seed) for key in df["canonical_proxy_smiles"].astype(str)]
    df = df.sort_values("rank").head(val_size).reset_index(drop=True)
    df = df.drop(columns=["rank"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return cache_path


def build_representation_vocab(
    cache_path: Path,
    vocab_path: Path,
    *,
    include_bigsmiles: bool = False,
) -> RepresentationVocab:
    if vocab_path.exists():
        return RepresentationVocab.load(vocab_path)
    df = pd.read_parquet(cache_path)
    train_df = df[df["split"] == "train"] if "split" in df.columns else df
    bigsmiles_values = None
    if include_bigsmiles and "bigsmiles" in train_df.columns:
        bigsmiles_values = train_df["bigsmiles"].dropna().astype(str).tolist()
    vocab = RepresentationVocab.build(
        train_df["psmiles"].astype(str).tolist(),
        train_df["pselfies"].astype(str).tolist(),
        bigsmiles_values=bigsmiles_values,
    )
    vocab.save(vocab_path)
    return vocab


def load_records(cache_path: Path, split: str | None, limit: int | None = None) -> list[P1MRecord]:
    df = pd.read_parquet(cache_path)
    split_df = df[df["split"] == split] if split is not None and "split" in df.columns else df
    if limit is not None:
        split_df = split_df.head(limit)
    return [
        P1MRecord(
            psmiles=row.psmiles,
            pselfies=row.pselfies,
            bigsmiles=None if not hasattr(row, "bigsmiles") or pd.isna(row.bigsmiles) else str(row.bigsmiles),
            sa_score=None if not hasattr(row, "sa_score") or pd.isna(row.sa_score) else float(row.sa_score),
            source_name=None if not hasattr(row, "source_name") else row.source_name,
        )
        for row in split_df.itertuples(index=False)
    ]


def load_smallmol_records(csv_path: str, limit: int | None = None) -> list[P1MRecord]:
    """Load small molecule SMILES from a CSV (expects a 'smiles' column) and convert to P1MRecord."""
    df = pd.read_csv(csv_path)
    smiles_col = "smiles" if "smiles" in df.columns else "SMILES" if "SMILES" in df.columns else df.columns[0]
    smiles_values = df[smiles_col].astype(str).str.strip().tolist()
    records = []
    for smi in smiles_values:
        smi = smi.strip().replace("\n", "")
        if not smi:
            continue
        try:
            selfies = sf.encoder(smi)
            if selfies is None:
                continue
            records.append(P1MRecord(psmiles=smi, pselfies=selfies, source_name="smallmol"))
        except Exception:
            continue
        if limit is not None and len(records) >= limit:
            break
    return records


class P1MDataset(Dataset):
    def __init__(self, records: list[P1MRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> P1MRecord:
        return self.records[index]


def _apply_mlm_mask(
    input_ids: torch.Tensor,
    *,
    pad_token_id: int,
    mask_token_id: int,
    vocab_size: int,
    probability: float,
    label_unmasked: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = input_ids.clone()
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, probability)
    special_mask = (labels == pad_token_id) | (labels == 0) | (labels == 2)
    probability_matrix.masked_fill_(special_mask, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    if not label_unmasked:
        labels[~masked_indices] = -100
    replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[replace_mask] = mask_token_id
    random_mask = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~replace_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[random_mask] = random_words[random_mask]
    return inputs, labels


class ContinuationCollator:
    def __init__(
        self,
        tokenizer,
        rep_vocab: RepresentationVocab,
        *,
        psmiles_max_len: int,
        translation_source_max_len: int,
        translation_target_max_len: int,
        mlm_probability: float,
        translation_mask_probability: float,
        smiles_prefix: str = "",
        selfies_prefix: str = "",
        mlm_selfies_mix: bool = False,
        translation_target_mode: str = "paired",
    ) -> None:
        self.tokenizer = tokenizer
        self.rep_vocab = rep_vocab
        self.psmiles_max_len = psmiles_max_len
        self.translation_source_max_len = translation_source_max_len
        self.translation_target_max_len = translation_target_max_len
        self.mlm_probability = mlm_probability
        self.translation_mask_probability = translation_mask_probability
        self.smiles_prefix = smiles_prefix
        self.selfies_prefix = selfies_prefix
        self.mlm_selfies_mix = mlm_selfies_mix
        if translation_target_mode not in {"paired", "bigsmiles"}:
            raise ValueError(f"Unsupported translation_target_mode: {translation_target_mode}")
        self.translation_target_mode = translation_target_mode

    def _tokenize(self, texts: list[str], max_len: int) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def _prefix_smiles(self, texts: list[str]) -> list[str]:
        if self.smiles_prefix:
            return [f"{self.smiles_prefix} {t}" for t in texts]
        return texts

    def _prefix_selfies(self, texts: list[str]) -> list[str]:
        if self.selfies_prefix:
            return [f"{self.selfies_prefix} {t}" for t in texts]
        return texts

    def __call__(self, records: list[P1MRecord]) -> dict[str, torch.Tensor]:
        psmiles = [record.psmiles for record in records]
        pselfies = [record.pselfies for record in records]
        bigsmiles = [record.bigsmiles for record in records]

        if self.mlm_selfies_mix:
            mlm_texts = []
            for smi, sel in zip(psmiles, pselfies):
                if random.random() < 0.5:
                    mlm_texts.append(f"{self.smiles_prefix} {smi}" if self.smiles_prefix else smi)
                else:
                    mlm_texts.append(f"{self.selfies_prefix} {sel}" if self.selfies_prefix else sel)
            mlm_tokens = self._tokenize(mlm_texts, self.psmiles_max_len)
        else:
            mlm_tokens = self._tokenize(self._prefix_smiles(psmiles), self.psmiles_max_len)
        mlm_input_ids, mlm_labels = _apply_mlm_mask(
            mlm_tokens["input_ids"],
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            vocab_size=len(self.tokenizer),
            probability=self.mlm_probability,
            label_unmasked=False,
        )

        view1 = self._tokenize(self._prefix_smiles([randomize_psmiles(text) for text in psmiles]), self.psmiles_max_len)
        view2 = self._tokenize(self._prefix_smiles([randomize_psmiles(text) for text in psmiles]), self.psmiles_max_len)

        translation_sources: list[str] = []
        translation_target_ids: list[list[int]] = []
        translation_directions: list[int] = []
        for psmiles_value, pselfies_value, bigsmiles_value in zip(psmiles, pselfies, bigsmiles):
            if self.translation_target_mode == "bigsmiles":
                if not bigsmiles_value:
                    raise RuntimeError("Missing BigSMILES value in batch for translation_target_mode='bigsmiles'")
                if random.random() < 0.5:
                    translation_sources.append(f"{self.smiles_prefix} {psmiles_value}" if self.smiles_prefix else psmiles_value)
                    translation_directions.append(0)
                else:
                    translation_sources.append(f"{self.selfies_prefix} {pselfies_value}" if self.selfies_prefix else pselfies_value)
                    translation_directions.append(1)
                translation_target_ids.append(
                    self.rep_vocab.encode(bigsmiles_value, rep_type="bigsmiles", max_length=self.translation_target_max_len)
                )
                continue

            if random.random() < 0.5:
                translation_sources.append(f"{self.smiles_prefix} {psmiles_value}" if self.smiles_prefix else psmiles_value)
                translation_target_ids.append(
                    self.rep_vocab.encode(pselfies_value, rep_type="pselfies", max_length=self.translation_target_max_len)
                )
                translation_directions.append(0)
            else:
                translation_sources.append(f"{self.selfies_prefix} {pselfies_value}" if self.selfies_prefix else pselfies_value)
                translation_target_ids.append(
                    self.rep_vocab.encode(psmiles_value, rep_type="psmiles", max_length=self.translation_target_max_len)
                )
                translation_directions.append(1)

        translation_source_tokens = self._tokenize(translation_sources, self.translation_source_max_len)
        translation_input_ids, _ = _apply_mlm_mask(
            translation_source_tokens["input_ids"],
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            vocab_size=len(self.tokenizer),
            probability=self.translation_mask_probability,
            label_unmasked=True,
        )

        batch = {
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_tokens["attention_mask"],
            "mlm_labels": mlm_labels,
            "view1_input_ids": view1["input_ids"],
            "view1_attention_mask": view1["attention_mask"],
            "view2_input_ids": view2["input_ids"],
            "view2_attention_mask": view2["attention_mask"],
            "translation_input_ids": translation_input_ids,
            "translation_attention_mask": translation_source_tokens["attention_mask"],
            "translation_target_ids": torch.tensor(translation_target_ids, dtype=torch.long),
            "translation_direction": torch.tensor(translation_directions, dtype=torch.long),
        }
        return batch
