from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import pandas as pd
import selfies as sf
import torch
from transformers import AutoTokenizer

from .deepchem_original_tokenizer import BasicSmilesTokenizer as DeepChemBasicSmilesTokenizer
from .deepchem_original_tokenizer import SmilesTokenizer as DeepChemSmilesTokenizer
from .pselfies import RepresentationVocab
from .pselfies import randomize_psmiles


CACHE = Path("/mnt/data/p1m_pretrain_experiments/cache")
PI1M_TRAIN_CACHE = CACHE / "pi1m_train_dedup.parquet"
DEEPCHEM_SMILES_DIR = CACHE / "deepchem_smiles_tokenizer"
DEEPCHEM_SMILES_VOCAB_TXT = DEEPCHEM_SMILES_DIR / "vocab.txt"
PSELFIES_TOKENS_PATH = CACHE / "dual_pselfies_tokens.json"

PSMILES_LANGUAGE_ID = 0
PSELFIES_LANGUAGE_ID = 1

PSELFIES_SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]


def ensure_deepchem_smiles_tokenizer(local_dir: Path | None = None) -> Path:
    local_dir = local_dir or DEEPCHEM_SMILES_DIR
    required_files = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    }
    if required_files.issubset({path.name for path in local_dir.iterdir()}) if local_dir.exists() else False:
        return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download("DeepChem/SmilesTokenizer_PubChem_1M", local_dir=str(local_dir))
    return local_dir


def ensure_deepchem_vocab_txt(local_dir: Path | None = None) -> Path:
    local_dir = ensure_deepchem_smiles_tokenizer(local_dir)
    vocab_txt = local_dir / "vocab.txt"
    if vocab_txt.exists():
        return vocab_txt
    vocab_json = local_dir / "vocab.json"
    payload = json.loads(vocab_json.read_text())
    ordered = [None] * (max(payload.values()) + 1)
    for token, idx in payload.items():
        ordered[idx] = token
    vocab_txt.write_text("\n".join(token if token is not None else "[UNK]" for token in ordered) + "\n")
    return vocab_txt


def load_deepchem_smiles_tokenizer(max_len: int = 256):
    local_dir = ensure_deepchem_smiles_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
    tokenizer.model_max_length = max_len
    return tokenizer


def load_original_deepchem_smiles_tokenizer(max_len: int = 256) -> DeepChemSmilesTokenizer:
    vocab_txt = ensure_deepchem_vocab_txt()
    tokenizer = DeepChemSmilesTokenizer(
        str(vocab_txt),
        do_lower_case=False,
        do_basic_tokenize=False,
        model_max_length=max_len,
    )
    tokenizer.model_max_length = max_len
    return tokenizer


def load_original_deepchem_basic_tokenizer() -> DeepChemBasicSmilesTokenizer:
    return DeepChemBasicSmilesTokenizer()


def _build_pselfies_tokens(cache_path: Path) -> list[str]:
    df = pd.read_parquet(cache_path, columns=["pselfies"])
    tokens = set()
    for value in df["pselfies"].astype(str).tolist():
        try:
            tokens.update(sf.split_selfies(value))
        except Exception:
            continue
    ordered = [tok for tok in PSELFIES_SPECIAL_TOKENS]
    ordered.extend(sorted(tokens - set(PSELFIES_SPECIAL_TOKENS)))
    return ordered


class PSelfiesTokenizer:
    def __init__(self, tokens: list[str], max_len: int = 256):
        self.id_to_token = list(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self._max_len = max_len

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["[PAD]"]

    @property
    def cls_token_id(self) -> int:
        return self.token_to_id["[CLS]"]

    @property
    def sep_token_id(self) -> int:
        return self.token_to_id["[SEP]"]

    @property
    def bos_token_id(self) -> int:
        return self.cls_token_id

    @property
    def eos_token_id(self) -> int:
        return self.sep_token_id

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["[UNK]"]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id["[MASK]"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def __len__(self) -> int:
        return len(self.id_to_token)

    def tokenize(self, text: str) -> list[str]:
        try:
            return list(sf.split_selfies(text))
        except Exception:
            return []

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int | None = None) -> list[int]:
        max_length = max_length or self._max_len
        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids[: max_length - 2] + [self.sep_token_id]
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
        max_length = max_length or self._max_len
        all_ids = [self.encode(item, add_special_tokens=add_special_tokens, max_length=max_length) for item in text]

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
            result = {key: torch.tensor(value) for key, value in result.items()}
        return result

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"tokens": self.id_to_token}, indent=2))

    @classmethod
    def load(cls, path: Path, max_len: int = 256) -> "PSelfiesTokenizer":
        payload = json.loads(path.read_text())
        tokens = payload["tokens"] if isinstance(payload, dict) else payload
        return cls(tokens, max_len=max_len)


def load_pselfies_tokenizer(max_len: int = 256, vocab_path: Path | None = None) -> PSelfiesTokenizer:
    vocab_path = vocab_path or PSELFIES_TOKENS_PATH
    if not vocab_path.exists():
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer = PSelfiesTokenizer(_build_pselfies_tokens(PI1M_TRAIN_CACHE), max_len=max_len)
        tokenizer.save(vocab_path)
        return tokenizer
    return PSelfiesTokenizer.load(vocab_path, max_len=max_len)


@dataclass
class DualTokenizerBundle:
    psmiles_tokenizer: Any
    pselfies_tokenizer: PSelfiesTokenizer

    @classmethod
    def load(cls, max_len: int = 256, use_original_deepchem: bool = False) -> "DualTokenizerBundle":
        smiles_tokenizer = (
            load_original_deepchem_smiles_tokenizer(max_len=max_len)
            if use_original_deepchem
            else load_deepchem_smiles_tokenizer(max_len=max_len)
        )
        return cls(
            psmiles_tokenizer=smiles_tokenizer,
            pselfies_tokenizer=load_pselfies_tokenizer(max_len=max_len),
        )


def _pad_sequences(sequences: list[list[int]], pad_ids: list[int], pad_to: int) -> tuple[list[list[int]], list[list[int]]]:
    padded: list[list[int]] = []
    attention_masks: list[list[int]] = []
    for ids, pad_id in zip(sequences, pad_ids):
        truncated = ids[:pad_to]
        padded_ids = truncated + [pad_id] * (pad_to - len(truncated))
        mask = [1] * len(truncated) + [0] * (pad_to - len(truncated))
        padded.append(padded_ids)
        attention_masks.append(mask)
    return padded, attention_masks


def _language_specific_ids(bundle: DualTokenizerBundle, language_id: int) -> tuple[int, int, int, int]:
    tokenizer = bundle.psmiles_tokenizer if language_id == PSMILES_LANGUAGE_ID else bundle.pselfies_tokenizer
    return tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id


def _apply_dual_mlm_mask(
    input_ids: torch.Tensor,
    language_ids: torch.Tensor,
    *,
    bundle: DualTokenizerBundle,
    probability: float,
    label_unmasked: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = input_ids.clone()
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, probability, dtype=torch.float32)
    for language_id in (PSMILES_LANGUAGE_ID, PSELFIES_LANGUAGE_ID):
        row_mask = language_ids.eq(language_id)
        if not row_mask.any():
            continue
        pad_id, cls_id, sep_id, mask_id = _language_specific_ids(bundle, language_id)
        vocab_size = len(bundle.psmiles_tokenizer) if language_id == PSMILES_LANGUAGE_ID else len(bundle.pselfies_tokenizer)
        special_mask = row_mask.unsqueeze(1) & (
            labels.eq(pad_id) | labels.eq(cls_id) | labels.eq(sep_id)
        )
        probability_matrix.masked_fill_(special_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix[row_mask]).bool()
        replace_mask = torch.bernoulli(torch.full(masked_indices.shape, 0.8)).bool() & masked_indices
        random_mask = torch.bernoulli(torch.full(masked_indices.shape, 0.5)).bool() & masked_indices & ~replace_mask
        lang_inputs = inputs[row_mask]
        lang_inputs[replace_mask] = mask_id
        if random_mask.any():
            random_words = torch.randint(vocab_size, masked_indices.shape, dtype=torch.long, device=input_ids.device)
            lang_inputs[random_mask] = random_words[random_mask]
        inputs[row_mask] = lang_inputs
        if not label_unmasked:
            lang_labels = labels[row_mask]
            lang_labels[~masked_indices] = -100
            labels[row_mask] = lang_labels
    return inputs, labels


class DualTokenizerContinuationCollator:
    def __init__(
        self,
        bundle: DualTokenizerBundle,
        *,
        psmiles_max_len: int,
        translation_source_max_len: int,
        translation_target_max_len: int,
        mlm_probability: float,
        translation_mask_probability: float,
        mlm_selfies_mix: bool = False,
        translation_target_mode: str = "paired",
        translation_vocab: RepresentationVocab | None = None,
    ) -> None:
        self.bundle = bundle
        self.psmiles_max_len = psmiles_max_len
        self.translation_source_max_len = translation_source_max_len
        self.translation_target_max_len = translation_target_max_len
        self.mlm_probability = mlm_probability
        self.translation_mask_probability = translation_mask_probability
        self.mlm_selfies_mix = mlm_selfies_mix
        if translation_target_mode not in {"paired", "bigsmiles"}:
            raise ValueError(f"Unsupported translation_target_mode: {translation_target_mode}")
        if translation_target_mode == "bigsmiles" and translation_vocab is None:
            raise ValueError("translation_vocab is required for translation_target_mode='bigsmiles'")
        self.translation_target_mode = translation_target_mode
        self.translation_vocab = translation_vocab

    def _encode_batch(
        self,
        texts: list[str],
        language_ids: list[int],
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = []
        pad_ids = []
        for text, language_id in zip(texts, language_ids):
            tokenizer = self.bundle.psmiles_tokenizer if language_id == PSMILES_LANGUAGE_ID else self.bundle.pselfies_tokenizer
            encoded.append(tokenizer.encode(text, add_special_tokens=True, max_length=max_len))
            pad_ids.append(tokenizer.pad_token_id)
        padded, attention_masks = _pad_sequences(encoded, pad_ids, max_len)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
            torch.tensor(language_ids, dtype=torch.long),
        )

    def __call__(self, records) -> dict[str, torch.Tensor]:
        psmiles = [record.psmiles for record in records]
        pselfies = [record.pselfies for record in records]
        bigsmiles = [record.bigsmiles for record in records]

        if self.mlm_selfies_mix:
            mlm_texts = []
            mlm_languages = []
            for psmiles_value, pselfies_value in zip(psmiles, pselfies):
                if torch.rand(1).item() < 0.5:
                    mlm_texts.append(psmiles_value)
                    mlm_languages.append(PSMILES_LANGUAGE_ID)
                else:
                    mlm_texts.append(pselfies_value)
                    mlm_languages.append(PSELFIES_LANGUAGE_ID)
        else:
            mlm_texts = psmiles
            mlm_languages = [PSMILES_LANGUAGE_ID] * len(psmiles)
        mlm_ids, mlm_attention, mlm_language_ids = self._encode_batch(mlm_texts, mlm_languages, self.psmiles_max_len)
        mlm_input_ids, mlm_labels = _apply_dual_mlm_mask(
            mlm_ids,
            mlm_language_ids,
            bundle=self.bundle,
            probability=self.mlm_probability,
            label_unmasked=False,
        )

        view1_ids, view1_attention, view1_language_ids = self._encode_batch(
            [randomize_psmiles(text) for text in psmiles],
            [PSMILES_LANGUAGE_ID] * len(psmiles),
            self.psmiles_max_len,
        )
        view2_ids, view2_attention, view2_language_ids = self._encode_batch(
            [randomize_psmiles(text) for text in psmiles],
            [PSMILES_LANGUAGE_ID] * len(psmiles),
            self.psmiles_max_len,
        )

        translation_sources: list[str] = []
        translation_source_languages: list[int] = []
        translation_targets: list[list[int]] = []
        translation_target_languages: list[int] = []
        translation_directions: list[int] = []

        for psmiles_value, pselfies_value, bigsmiles_value in zip(psmiles, pselfies, bigsmiles):
            if self.translation_target_mode == "bigsmiles":
                if not bigsmiles_value:
                    raise RuntimeError("Missing BigSMILES value in batch for translation_target_mode='bigsmiles'")
                if torch.rand(1).item() < 0.5:
                    translation_sources.append(psmiles_value)
                    translation_source_languages.append(PSMILES_LANGUAGE_ID)
                    translation_directions.append(0)
                else:
                    translation_sources.append(pselfies_value)
                    translation_source_languages.append(PSELFIES_LANGUAGE_ID)
                    translation_directions.append(1)
                translation_targets.append(
                    self.translation_vocab.encode(
                        bigsmiles_value,
                        rep_type="bigsmiles",
                        max_length=self.translation_target_max_len,
                    )
                )
                translation_target_languages.append(PSMILES_LANGUAGE_ID)
                continue

            if torch.rand(1).item() < 0.5:
                translation_sources.append(psmiles_value)
                translation_source_languages.append(PSMILES_LANGUAGE_ID)
                translation_targets.append(
                    self.bundle.pselfies_tokenizer.encode(
                        pselfies_value,
                        add_special_tokens=True,
                        max_length=self.translation_target_max_len,
                    )
                )
                translation_target_languages.append(PSELFIES_LANGUAGE_ID)
                translation_directions.append(0)
            else:
                translation_sources.append(pselfies_value)
                translation_source_languages.append(PSELFIES_LANGUAGE_ID)
                translation_targets.append(
                    self.bundle.psmiles_tokenizer.encode(
                        psmiles_value,
                        add_special_tokens=True,
                        max_length=self.translation_target_max_len,
                    )
                )
                translation_target_languages.append(PSMILES_LANGUAGE_ID)
                translation_directions.append(1)

        translation_ids, translation_attention, translation_source_language_ids = self._encode_batch(
            translation_sources,
            translation_source_languages,
            self.translation_source_max_len,
        )
        translation_input_ids, _ = _apply_dual_mlm_mask(
            translation_ids,
            translation_source_language_ids,
            bundle=self.bundle,
            probability=self.translation_mask_probability,
            label_unmasked=True,
        )
        if self.translation_target_mode == "bigsmiles":
            target_pad_ids = [self.translation_vocab.pad_id] * len(translation_target_languages)
        else:
            target_pad_ids = [
                self.bundle.psmiles_tokenizer.pad_token_id if lang == PSMILES_LANGUAGE_ID else self.bundle.pselfies_tokenizer.pad_token_id
                for lang in translation_target_languages
            ]
        translation_target_padded, _ = _pad_sequences(translation_targets, target_pad_ids, self.translation_target_max_len)

        return {
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention,
            "mlm_labels": mlm_labels,
            "mlm_language_ids": mlm_language_ids,
            "view1_input_ids": view1_ids,
            "view1_attention_mask": view1_attention,
            "view1_language_ids": view1_language_ids,
            "view2_input_ids": view2_ids,
            "view2_attention_mask": view2_attention,
            "view2_language_ids": view2_language_ids,
            "translation_input_ids": translation_input_ids,
            "translation_attention_mask": translation_attention,
            "translation_source_language_ids": translation_source_language_ids,
            "translation_target_ids": torch.tensor(translation_target_padded, dtype=torch.long),
            "translation_target_language_ids": torch.tensor(translation_target_languages, dtype=torch.long),
            "translation_direction": torch.tensor(translation_directions, dtype=torch.long),
        }
