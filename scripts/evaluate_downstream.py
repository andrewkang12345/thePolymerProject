#!/usr/bin/env python3
"""Evaluate pretrained backbone checkpoints on downstream polymer property prediction tasks.

Usage:
    # Evaluate a single model on a single task:
    python scripts/evaluate_downstream.py \
        --checkpoint outputs/externalval_long_checkpoint_transpolymer_both/best.pt \
        --task Egc --folds 5

    # Evaluate a single model on all tasks:
    python scripts/evaluate_downstream.py \
        --checkpoint outputs/externalval_long_checkpoint_transpolymer_both/best.pt \
        --task all --folds 5

    # Evaluate the original TransPolymer baseline:
    python scripts/evaluate_downstream.py \
        --baseline transpolymer --task all --folds 5

    # Evaluate the original MMPolymer baseline:
    python scripts/evaluate_downstream.py \
        --baseline mmpolymer --task all --folds 5

    # Evaluate random init baseline:
    python scripts/evaluate_downstream.py \
        --baseline random --task all --folds 5

    # Run all models on all tasks:
    python scripts/evaluate_downstream.py --run-all --folds 5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaModel,
    get_linear_schedule_with_warmup,
)

import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1m_pretrain.paths import get_paths

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATHS = get_paths()
EXPERIMENTS = PATHS.project_root
OUTPUTS = PATHS.outputs_dir
TRANSPOLYMER_REPO = PATHS.transpolymer_repo
MMPOLYMER_REPO = PATHS.mmpolymer_repo
MMPOLYMER_DATA = PATHS.mmpolymer_data_root
TRANSPOLYMER_CKPT = PATHS.checkpoints_dir / "transpolymer" / "pytorch_model.bin"
MMPOLYMER_CKPT = PATHS.checkpoints_dir / "mmpolymer" / "pretrain.pt"
RESULTS_DIR = PATHS.downstream_results_dir
INPUT_MODE_SINGLE = "single"
INPUT_MODE_DUAL = "dual_input"
INPUT_MODE_CONCAT_ENCODER = "concat_encoder"

# Downstream tasks available (all from MMPolymer data dir, psmiles+value CSVs)
ALL_TASKS = ["Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "Eat"]

# Models to skip: short externalval (superseded by long), smoke test, bad vw2p0
SKIP_MODELS = {
    "externalval_checkpoint_mmpolymer_both",
    "externalval_checkpoint_mmpolymer_original",
    "externalval_checkpoint_mmpolymer_translation_only",
    "externalval_checkpoint_mmpolymer_view_only",
    "externalval_checkpoint_transpolymer_both",
    "externalval_checkpoint_transpolymer_original",
    "externalval_checkpoint_transpolymer_translation_only",
    "externalval_checkpoint_transpolymer_view_only",
    "wandb_smoke_checkpoint",
}

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_TOKENIZER_CACHE = None


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_tokenizer(max_len: int = 256):
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is not None:
        return _TOKENIZER_CACHE
    module = _load_module(TRANSPOLYMER_REPO / "PolymerSmilesTokenization.py", "transpolymer_tokenizer")
    if not getattr(module.PolymerSmilesTokenizer, "_codex_patched", False):
        original_init = module.PolymerSmilesTokenizer.__init__

        def patched_init(self, vocab_file, merges_file, *args, **kwargs):
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                self.encoder = json.load(vocab_handle)
            self.decoder = {value: key for key, value in self.encoder.items()}
            return original_init(self, vocab_file, merges_file, *args, **kwargs)

        module.PolymerSmilesTokenizer.__init__ = patched_init
        module.PolymerSmilesTokenizer._codex_patched = True
    tokenizer = module.PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=max_len)
    _TOKENIZER_CACHE = tokenizer
    return tokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PolymerPropertyDataset(Dataset):
    def __init__(self, smiles: list[str], values: list[float], tokenizer, max_len: int = 256):
        self.smiles = smiles
        self.values = values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.smiles[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "prop": self.values[idx],
        }


class DualPolymerPropertyDataset(Dataset):
    def __init__(self, smiles: list[str], pselfies: list[str], values: list[float], smiles_tokenizer, selfies_tokenizer, max_len: int = 256):
        self.smiles = smiles
        self.pselfies = pselfies
        self.values = values
        self.smiles_tokenizer = smiles_tokenizer
        self.selfies_tokenizer = selfies_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_encoding = self.smiles_tokenizer(
            str(self.smiles[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        selfies_encoding = self.selfies_tokenizer(
            str(self.pselfies[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "smiles_input_ids": smiles_encoding["input_ids"].flatten(),
            "smiles_attention_mask": smiles_encoding["attention_mask"].flatten(),
            "selfies_input_ids": selfies_encoding["input_ids"].flatten(),
            "selfies_attention_mask": selfies_encoding["attention_mask"].flatten(),
            "prop": self.values[idx],
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DownstreamRegression(nn.Module):
    """CLS-token pooling + 2-layer MLP regressor (matches TransPolymer)."""

    def __init__(self, backbone: RobertaModel, hidden_size: int, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls)


class DualInputDownstreamRegression(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def _encode_branch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, language_id: int) -> torch.Tensor:
        language_ids = torch.full((input_ids.size(0),), language_id, device=input_ids.device, dtype=torch.long)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, language_ids=language_ids)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, smiles_input_ids, smiles_attention_mask, selfies_input_ids, selfies_attention_mask):
        smiles_cls = self._encode_branch(smiles_input_ids, smiles_attention_mask, 0)
        selfies_cls = self._encode_branch(selfies_input_ids, selfies_attention_mask, 1)
        combined = torch.cat([smiles_cls, selfies_cls], dim=-1)
        return self.regressor(combined)


class ConcatEncoderDownstreamRegression(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def _embed_segment(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        embedding: nn.Embedding,
        position_ids: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        hidden = embedding(input_ids) + self.backbone.position_embeddings(position_ids)
        hidden = self.backbone.embedding_dropout(self.backbone.embedding_norm(hidden))
        for block in blocks:
            hidden = block(hidden, attention_mask)
        return hidden

    def _concat_hidden(
        self,
        smiles_hidden: torch.Tensor,
        smiles_attention_mask: torch.Tensor,
        selfies_hidden: torch.Tensor,
        selfies_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = smiles_hidden.size(0)
        hidden_size = smiles_hidden.size(-1)
        smiles_lengths = smiles_attention_mask.sum(dim=1)
        selfies_lengths = selfies_attention_mask.sum(dim=1)
        total_lengths = smiles_lengths + selfies_lengths
        max_total = int(total_lengths.max().item())
        combined_hidden = smiles_hidden.new_zeros((batch_size, max_total, hidden_size))
        combined_attention = smiles_attention_mask.new_zeros((batch_size, max_total))
        for idx in range(batch_size):
            s_len = int(smiles_lengths[idx].item())
            f_len = int(selfies_lengths[idx].item())
            if s_len:
                combined_hidden[idx, :s_len] = smiles_hidden[idx, :s_len]
            if f_len:
                combined_hidden[idx, s_len : s_len + f_len] = selfies_hidden[idx, :f_len]
            combined_attention[idx, : s_len + f_len] = 1
        return combined_hidden, combined_attention

    def forward(self, smiles_input_ids, smiles_attention_mask, selfies_input_ids, selfies_attention_mask):
        # Drop the second branch CLS so the concatenated sequence has one leading CLS.
        selfies_input_ids = selfies_input_ids[:, 1:]
        selfies_attention_mask = selfies_attention_mask[:, 1:]

        smiles_positions = torch.arange(smiles_input_ids.size(1), device=smiles_input_ids.device).unsqueeze(0)
        smiles_hidden = self._embed_segment(
            smiles_input_ids,
            smiles_attention_mask,
            embedding=self.backbone.smiles_embeddings,
            position_ids=smiles_positions,
            blocks=self.backbone.smiles_encoder_layers,
        )

        smiles_lengths = smiles_attention_mask.sum(dim=1, keepdim=True)
        selfies_positions = smiles_lengths + torch.arange(selfies_input_ids.size(1), device=selfies_input_ids.device).unsqueeze(0)
        if int(selfies_positions.max().item()) >= self.backbone.position_embeddings.num_embeddings:
            raise ValueError("Concatenated dual-input sequence exceeds backbone position embedding limit")
        selfies_hidden = self._embed_segment(
            selfies_input_ids,
            selfies_attention_mask,
            embedding=self.backbone.selfies_embeddings,
            position_ids=selfies_positions,
            blocks=self.backbone.selfies_encoder_layers,
        )

        combined_hidden, combined_attention = self._concat_hidden(
            smiles_hidden,
            smiles_attention_mask,
            selfies_hidden,
            selfies_attention_mask,
        )
        for layer in self.backbone.shared_encoder_layers:
            combined_hidden = layer(combined_hidden, combined_attention)
        combined_hidden = self.backbone.final_norm(combined_hidden)
        cls = combined_hidden[:, 0, :]
        return self.regressor(cls)


def _forward_regression_model(model, batch, device):
    if "smiles_input_ids" in batch:
        return model(
            batch["smiles_input_ids"].to(device),
            batch["smiles_attention_mask"].to(device),
            batch["selfies_input_ids"].to(device),
            batch["selfies_attention_mask"].to(device),
        )
    return model(batch["input_ids"].to(device), batch["attention_mask"].to(device))


# ---------------------------------------------------------------------------
# Backbone loading utilities
# ---------------------------------------------------------------------------

def load_roberta_model_from_continuation_ckpt(ckpt_path: str, device: str = "cpu") -> tuple[RobertaModel, dict]:
    """Load a ContinuationModel checkpoint and extract the RobertaModel backbone."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    backbone_name = config["backbone"]
    backbone_family = config.get("backbone_family", "upstream_roberta")

    if backbone_family == "experimental":
        raise ValueError(f"Experimental backbones not supported for downstream eval: {ckpt_path}")

    if backbone_name == "transpolymer":
        roberta_config = RobertaConfig.from_pretrained(str(TRANSPOLYMER_REPO / "ckpt" / "pretrain.pt"))
    elif backbone_name == "mmpolymer":
        roberta_config = RobertaConfig.from_pretrained(str(MMPOLYMER_REPO / "MMPolymer" / "models" / "config"))
    elif backbone_name == "smi_ted":
        # SMI-TED uses custom architecture — load encoder directly
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from p1m_pretrain.smi_ted_wrapper import SmiTedForMLM
        smi_config = {"n_embd": 768, "n_layer": 12, "n_head": 12, "max_len": 202,
                      "d_dropout": 0.1, "num_feats": 32, "dropout": 0.1}
        # Infer vocab size from checkpoint embedding
        emb_key = "backbone.encoder.tok_emb.weight"
        n_vocab = state_dict[emb_key].shape[0] if emb_key in state_dict else 2393
        model = SmiTedForMLM(smi_config, n_vocab)
        # Load all backbone weights, skipping size mismatches in encoder's internal lang_model
        encoder_state = {}
        lang_state = {}
        for key, value in state_dict.items():
            if key.startswith("backbone.encoder."):
                new_key = key[len("backbone.encoder."):]
                # Skip encoder's internal lang_model.head if size doesn't match
                if "lang_model.head" in new_key and value.shape[0] != n_vocab:
                    continue
                encoder_state[new_key] = value
            elif key.startswith("backbone.lang_model."):
                lang_state[key[len("backbone.lang_model."):]] = value
        model.encoder.load_state_dict(encoder_state, strict=False)
        if lang_state:
            model.lang_model.load_state_dict(lang_state, strict=False)
        return model, config
    elif backbone_name in {"dual_deepchem_pselfies_shared", "dual_correctdeepchem_pselfies_shared"}:
        from p1m_pretrain.upstream import load_backbone_model

        model = load_backbone_model(
            backbone_name,
            init_mode="scratch",
            scratch_variant=config.get("scratch_variant", "base"),
        )
        model_state = {
            key[len("backbone."):]: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"  Warning: missing keys in dual backbone: {sorted(missing)}")
        if unexpected:
            print(f"  Warning: unexpected keys in dual backbone: {sorted(unexpected)}")
        return model, config
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Extract backbone.roberta.* -> strip to RobertaModel keys
    model_state = {}
    prefix = "backbone.roberta."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            model_state[key[len(prefix):]] = value

    model = RobertaModel(roberta_config)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    ignored = {"embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"}
    real_missing = set(missing) - ignored
    real_unexpected = set(unexpected) - ignored
    if real_missing:
        print(f"  Warning: missing keys in backbone: {sorted(real_missing)}")
    if real_unexpected:
        print(f"  Warning: unexpected keys in backbone: {sorted(real_unexpected)}")

    return model, config


def load_original_transpolymer(device: str = "cpu") -> RobertaModel:
    """Load the original TransPolymer pretrained checkpoint."""
    roberta_config = RobertaConfig.from_pretrained(str(TRANSPOLYMER_REPO / "ckpt" / "pretrain.pt"))
    model = RobertaModel(roberta_config)
    state_dict = torch.load(TRANSPOLYMER_CKPT, map_location=device, weights_only=False)
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("roberta."):
            model_state[key[len("roberta."):]] = value
        else:
            model_state[key] = value
    model.load_state_dict(model_state, strict=False)
    return model


def load_original_mmpolymer(device: str = "cpu") -> RobertaModel:
    """Load the original MMPolymer pretrained checkpoint (1D RoBERTa part only)."""
    roberta_config = RobertaConfig.from_pretrained(str(MMPOLYMER_REPO / "MMPolymer" / "models" / "config"))
    model = RobertaModel(roberta_config)
    raw_state = torch.load(MMPOLYMER_CKPT, map_location=device, weights_only=False)["model"]
    model_state = {}
    for key, value in raw_state.items():
        if key.startswith("PretrainedModel."):
            model_state[key[len("PretrainedModel."):]] = value
    model.load_state_dict(model_state, strict=False)
    return model


def load_random_init(backbone_name: str = "transpolymer") -> RobertaModel:
    """Random init baseline."""
    if backbone_name == "transpolymer":
        roberta_config = RobertaConfig.from_pretrained(str(TRANSPOLYMER_REPO / "ckpt" / "pretrain.pt"))
    else:
        roberta_config = RobertaConfig.from_pretrained(str(MMPOLYMER_REPO / "MMPolymer" / "models" / "config"))
    return RobertaModel(roberta_config)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    lr_backbone: float = 5e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 50
    warmup_ratio: float = 0.05
    drop_rate: float = 0.1
    tolerance: int = 8
    max_len: int = 256
    num_workers: int = 2
    seed: int = 42
    use_amp: bool = True


def train_one_epoch(model, optimizer, scheduler, loss_fn, dataloader, device, scaler_amp):
    model.train()
    epoch_loss = 0.0
    n = 0
    for batch in dataloader:
        prop = batch["prop"].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=scaler_amp is not None):
            outputs = _forward_regression_model(model, batch, device).squeeze(-1)
            loss = loss_fn(outputs, prop)
        if scaler_amp is not None:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        epoch_loss += loss.item() * len(prop)
        n += len(prop)
    return epoch_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, scaler_target):
    model.eval()
    preds, trues = [], []
    for batch in dataloader:
        prop = batch["prop"].float()
        with autocast(device_type="cuda", enabled=False):
            outputs = _forward_regression_model(model, batch, device).squeeze(-1).float().cpu()
        pred_orig = scaler_target.inverse_transform(outputs.numpy().reshape(-1, 1)).flatten()
        true_orig = scaler_target.inverse_transform(prop.numpy().reshape(-1, 1)).flatten()
        preds.extend(pred_orig.tolist())
        trues.extend(true_orig.tolist())
    preds = np.array(preds)
    trues = np.array(trues)
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return rmse, r2


def run_fold(
    backbone: RobertaModel,
    train_smiles: list[str],
    train_values: np.ndarray,
    test_smiles: list[str],
    test_values: np.ndarray,
    tokenizer,
    cfg: FinetuneConfig,
    device: str,
    fold_idx: int = 0,
    task_name: str = "",
    model_name: str = "",
    input_mode: str = INPUT_MODE_SINGLE,
    selfies_tokenizer=None,
) -> tuple[float, float, float, float, int]:
    """Train and evaluate one fold. Returns (train_rmse, test_rmse, train_r2, test_r2, best_epoch)."""
    scaler_target = StandardScaler()
    train_y = scaler_target.fit_transform(train_values.reshape(-1, 1)).flatten().tolist()
    test_y = scaler_target.transform(test_values.reshape(-1, 1)).flatten().tolist()

    if input_mode in {INPUT_MODE_DUAL, INPUT_MODE_CONCAT_ENCODER}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from p1m_pretrain.pselfies import proxy_pselfies_from_psmiles

        train_selfies = [proxy_pselfies_from_psmiles(smiles) for smiles in train_smiles]
        test_selfies = [proxy_pselfies_from_psmiles(smiles) for smiles in test_smiles]
        if any(value is None for value in train_selfies + test_selfies):
            raise RuntimeError(f"Failed to convert some downstream pSMILES to pSELFIES for task {task_name}")
        train_ds = DualPolymerPropertyDataset(train_smiles, train_selfies, train_y, tokenizer, selfies_tokenizer, cfg.max_len)
        test_ds = DualPolymerPropertyDataset(test_smiles, test_selfies, test_y, tokenizer, selfies_tokenizer, cfg.max_len)
    else:
        train_ds = PolymerPropertyDataset(train_smiles, train_y, tokenizer, cfg.max_len)
        test_ds = PolymerPropertyDataset(test_smiles, test_y, tokenizer, cfg.max_len)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)

    model_backbone = deepcopy(backbone)
    hidden_size = model_backbone.config.hidden_size
    if input_mode == INPUT_MODE_DUAL:
        model = DualInputDownstreamRegression(model_backbone, hidden_size, cfg.drop_rate).to(device)
    elif input_mode == INPUT_MODE_CONCAT_ENCODER:
        model = ConcatEncoderDownstreamRegression(model_backbone, hidden_size, cfg.drop_rate).to(device)
    else:
        model = DownstreamRegression(model_backbone, hidden_size, cfg.drop_rate).to(device)

    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.lr_backbone, "weight_decay": 0.0},
        {"params": model.regressor.parameters(), "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
    ])
    steps_per_epoch = max(len(train_dl), 1)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler_amp = GradScaler("cuda") if cfg.use_amp and device == "cuda" else None
    loss_fn = nn.MSELoss()
    best_test_r2 = -float("inf")
    best_results = (0.0, 0.0, 0.0, 0.0, 0)
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, optimizer, scheduler, loss_fn, train_dl, device, scaler_amp)
        train_rmse, train_r2 = evaluate(model, train_dl, device, scaler_target)
        test_rmse, test_r2 = evaluate(model, test_dl, device, scaler_target)

        # Log to wandb
        wandb.log({
            f"{task_name}/fold{fold_idx}/train_loss": train_loss,
            f"{task_name}/fold{fold_idx}/train_rmse": train_rmse,
            f"{task_name}/fold{fold_idx}/train_r2": train_r2,
            f"{task_name}/fold{fold_idx}/test_rmse": test_rmse,
            f"{task_name}/fold{fold_idx}/test_r2": test_r2,
        })

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_results = (train_rmse, test_rmse, train_r2, test_r2, epoch + 1)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.tolerance:
            break

    del model, optimizer, scheduler, scaler_amp
    torch.cuda.empty_cache()

    return best_results


def evaluate_model_on_task(
    backbone: RobertaModel,
    task_name: str,
    cfg: FinetuneConfig,
    device: str,
    n_folds: int = 5,
    model_name: str = "",
    tokenizer_override=None,
    input_mode: str = INPUT_MODE_SINGLE,
    selfies_tokenizer=None,
) -> dict:
    """Run K-fold CV for a model on one task."""
    tokenizer = tokenizer_override if tokenizer_override is not None else get_tokenizer(cfg.max_len)

    csv_path = MMPOLYMER_DATA / f"{task_name}.csv"
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    smiles = df["psmiles"].values
    values = df["value"].astype(float).values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(smiles)):
        train_smiles = smiles[train_idx].tolist()
        train_values = values[train_idx]
        test_smiles = smiles[test_idx].tolist()
        test_values = values[test_idx]

        train_rmse, test_rmse, train_r2, test_r2, best_ep = run_fold(
            backbone, train_smiles, train_values, test_smiles, test_values,
            tokenizer, cfg, device, fold_idx, task_name, model_name,
            input_mode=input_mode, selfies_tokenizer=selfies_tokenizer,
        )
        fold_results.append({
            "fold": fold_idx + 1,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "best_epoch": best_ep,
        })
        print(f"    Fold {fold_idx+1}/{n_folds}: test_rmse={test_rmse:.4f}, test_r2={test_r2:.4f} (ep{best_ep})")

    test_rmses = [r["test_rmse"] for r in fold_results]
    test_r2s = [r["test_r2"] for r in fold_results]
    train_rmses = [r["train_rmse"] for r in fold_results]
    train_r2s = [r["train_r2"] for r in fold_results]

    summary = {
        "task": task_name,
        "n_folds": n_folds,
        "n_samples": len(smiles),
        "test_rmse_mean": float(np.mean(test_rmses)),
        "test_rmse_std": float(np.std(test_rmses)),
        "test_r2_mean": float(np.mean(test_r2s)),
        "test_r2_std": float(np.std(test_r2s)),
        "train_rmse_mean": float(np.mean(train_rmses)),
        "train_r2_mean": float(np.mean(train_r2s)),
        "fold_results": fold_results,
    }

    # Log task summary to wandb
    wandb.log({
        f"{task_name}/test_r2_mean": summary["test_r2_mean"],
        f"{task_name}/test_r2_std": summary["test_r2_std"],
        f"{task_name}/test_rmse_mean": summary["test_rmse_mean"],
        f"{task_name}/test_rmse_std": summary["test_rmse_std"],
    })

    return summary


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------

def get_valid_continuation_models() -> list[dict]:
    """List all valid (non-corrupt, checkpoint+upstream_roberta) continuation model checkpoints."""
    models = []
    for d in sorted(os.listdir(OUTPUTS)):
        if d in SKIP_MODELS:
            continue
        best_pt = OUTPUTS / d / "best.pt"
        if not best_pt.is_file():
            continue
        try:
            ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            backbone_family = cfg.get("backbone_family", "upstream_roberta")
            if backbone_family == "experimental":
                continue
            init_mode = cfg.get("init_mode", "?")
            if init_mode == "scratch":
                continue
            models.append({
                "name": d,
                "path": str(best_pt),
                "backbone": cfg.get("backbone", "?"),
                "init_mode": init_mode,
                "backbone_family": backbone_family,
                "view_weight": cfg.get("view_weight", 0),
                "translation_weight": cfg.get("translation_weight", 0),
                "val_mlm": ckpt.get("metrics", {}).get("val_mlm_loss", None),
            })
        except Exception:
            continue
    return models


def run_single_checkpoint(
    ckpt_path: str,
    tasks: list[str],
    cfg: FinetuneConfig,
    device: str,
    n_folds: int = 5,
    input_mode: str = INPUT_MODE_SINGLE,
    model_name_override: str | None = None,
) -> list[dict]:
    """Evaluate a continuation checkpoint on specified tasks."""
    backbone, model_config = load_roberta_model_from_continuation_ckpt(ckpt_path, device="cpu")
    backbone_name = model_config.get("backbone", "")
    selfies_tokenizer = None
    if backbone_name == "smi_ted":
        from p1m_pretrain.smi_ted_tokenizer import SmiTedTokenizer
        tokenizer = SmiTedTokenizer(str(PATHS.smi_ted_dir / "bert_vocab_curated.txt"))
    elif backbone_name == "dual_deepchem_pselfies_shared":
        from p1m_pretrain.dual_tokenizer import load_deepchem_smiles_tokenizer, load_pselfies_tokenizer
        tokenizer = load_deepchem_smiles_tokenizer(cfg.max_len)
        selfies_tokenizer = load_pselfies_tokenizer(cfg.max_len)
    elif backbone_name == "dual_correctdeepchem_pselfies_shared":
        from p1m_pretrain.dual_tokenizer import load_original_deepchem_smiles_tokenizer, load_pselfies_tokenizer
        tokenizer = load_original_deepchem_smiles_tokenizer(cfg.max_len)
        selfies_tokenizer = load_pselfies_tokenizer(cfg.max_len)
    else:
        tokenizer = get_tokenizer(cfg.max_len)
        backbone.resize_token_embeddings(len(tokenizer))
    if input_mode in {INPUT_MODE_DUAL, INPUT_MODE_CONCAT_ENCODER} and backbone_name not in {"dual_deepchem_pselfies_shared", "dual_correctdeepchem_pselfies_shared"}:
        raise ValueError("Dual downstream input modes are only supported for dual_deepchem_pselfies_shared-style checkpoints")
    model_name = model_name_override or os.path.basename(os.path.dirname(ckpt_path))
    results = []
    for task in tasks:
        print(f"  Task: {task}")
        result = evaluate_model_on_task(
            backbone,
            task,
            cfg,
            device,
            n_folds,
            model_name,
            tokenizer_override=tokenizer,
            input_mode=input_mode,
            selfies_tokenizer=selfies_tokenizer,
        )
        result["model"] = model_name
        result["model_config"] = model_config
        result["model_config"]["downstream_input_mode"] = input_mode
        results.append(result)
    return results


def run_baseline(
    baseline_name: str,
    tasks: list[str],
    cfg: FinetuneConfig,
    device: str,
    n_folds: int = 5,
) -> list[dict]:
    """Evaluate a baseline model."""
    if baseline_name == "transpolymer":
        backbone = load_original_transpolymer()
    elif baseline_name == "mmpolymer":
        backbone = load_original_mmpolymer()
    elif baseline_name == "random":
        backbone = load_random_init("transpolymer")
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    tokenizer = get_tokenizer(cfg.max_len)
    backbone.resize_token_embeddings(len(tokenizer))
    model_name = f"baseline_{baseline_name}"
    results = []
    for task in tasks:
        print(f"  Task: {task}")
        result = evaluate_model_on_task(backbone, task, cfg, device, n_folds, model_name)
        result["model"] = model_name
        result["model_config"] = {"backbone": baseline_name, "init_mode": "baseline"}
        results.append(result)
    return results


def save_results(all_results: list[dict], output_path: str):
    """Save results to JSON and a summary CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_path = output_path.replace(".json", ".csv")
    rows = []
    for r in all_results:
        rows.append({
            "model": r["model"],
            "backbone": r.get("model_config", {}).get("backbone", ""),
            "view_weight": r.get("model_config", {}).get("view_weight", ""),
            "translation_weight": r.get("model_config", {}).get("translation_weight", ""),
            "task": r["task"],
            "n_samples": r["n_samples"],
            "test_rmse_mean": f"{r['test_rmse_mean']:.4f}",
            "test_rmse_std": f"{r['test_rmse_std']:.4f}",
            "test_r2_mean": f"{r['test_r2_mean']:.4f}",
            "test_r2_std": f"{r['test_r2_std']:.4f}",
            "train_r2_mean": f"{r['train_r2_mean']:.4f}",
        })
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nResults saved to:\n  {output_path}\n  {csv_path}")


def print_summary_table(all_results: list[dict]):
    """Print a formatted summary table."""
    print("\n" + "=" * 110)
    print(f"{'Model':<55} {'Task':<6} {'R² (mean±std)':<20} {'RMSE (mean±std)':<20}")
    print("=" * 110)
    for r in all_results:
        model_short = r["model"][:54]
        print(f"{model_short:<55} {r['task']:<6} "
              f"{r['test_r2_mean']:.4f}±{r['test_r2_std']:.4f}    "
              f"{r['test_rmse_mean']:.4f}±{r['test_rmse_std']:.4f}")


def build_summary_table(results: list[dict]) -> dict[str, float]:
    """Build a flat wandb-friendly summary from per-task results."""
    summary_table = {}
    for r in results:
        summary_table[f"{r['task']}_r2"] = r["test_r2_mean"]
        summary_table[f"{r['task']}_rmse"] = r["test_rmse_mean"]
    if results:
        summary_table["num_tasks_found"] = len(results)
    # Only publish a true model-level mean when we have multiple tasks.
    # Single-task runs otherwise clutter the mean_r2 chart with per-task points.
    if len(results) > 1:
        summary_table["mean_r2"] = float(np.mean([r["test_r2_mean"] for r in results]))
    return summary_table


def _checkpoint_eval_run_id(model_name: str, tasks: list[str], input_mode: str) -> str:
    key = f"{model_name}|{','.join(sorted(tasks))}|input_mode={input_mode}"
    return "ckpt-" + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Downstream evaluation for pretrained polymer encoders")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to continuation model checkpoint")
    group.add_argument("--baseline", type=str, choices=["transpolymer", "mmpolymer", "random"],
                       help="Evaluate a baseline model")
    group.add_argument("--run-all", action="store_true", help="Run all valid models + baselines")

    parser.add_argument("--task", type=str, default="all",
                        help="Task name (Egc, Egb, ...) or 'all'")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="p1m-downstream-eval")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results, skip completed models")
    parser.add_argument("--dual-input", action="store_true", help="For dual_deepchem_pselfies_shared checkpoints, use both pSMILES and converted pSELFIES as downstream inputs")
    parser.add_argument("--concat-encoder-input", action="store_true", help="For dual_deepchem_pselfies_shared checkpoints, concatenate pSMILES and converted pSELFIES inside the encoder and keep a single-width regressor head")
    parser.add_argument("--model-alias", type=str, default=None, help="Override model/run name used for outputs and wandb")
    args = parser.parse_args()

    if args.dual_input and args.concat_encoder_input:
        raise SystemExit("--dual-input and --concat-encoder-input are mutually exclusive")
    if args.concat_encoder_input:
        input_mode = INPUT_MODE_CONCAT_ENCODER
    elif args.dual_input:
        input_mode = INPUT_MODE_DUAL
    else:
        input_mode = INPUT_MODE_SINGLE

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tasks = ALL_TASKS if args.task == "all" else [args.task]

    cfg = FinetuneConfig(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        tolerance=args.patience,
        seed=args.seed,
        use_amp=not args.no_amp,
    )

    all_results = []
    completed_models = set()
    t0 = time.time()

    # Load existing results for resume
    resume_path = str(RESULTS_DIR / "all_results.json")
    if args.resume and os.path.exists(resume_path):
        with open(resume_path) as f:
            all_results = json.load(f)
        for r in all_results:
            completed_models.add(r["model"])
        # Deduplicate: keep only one set of results per model
        seen = set()
        deduped = []
        for r in all_results:
            key = (r["model"], r["task"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        all_results = deduped
        completed_models = {r["model"] for r in all_results}
        # Check if each model has all tasks
        model_tasks = {}
        for r in all_results:
            model_tasks.setdefault(r["model"], set()).add(r["task"])
        completed_models = {m for m, ts in model_tasks.items() if ts >= set(tasks)}
        print(f"Resuming: {len(completed_models)} models already completed, {len(all_results)} total results")

    if args.run_all:
        # ---- Baselines ----
        for bl in ["transpolymer", "mmpolymer", "random"]:
            run_name = f"baseline_{bl}"
            if run_name in completed_models:
                print(f"\n=== Baseline: {bl} === SKIPPED (already completed)")
                continue
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model": run_name,
                    "backbone": bl,
                    "init_mode": "baseline",
                    "tasks": tasks,
                    "folds": args.folds,
                    **cfg.__dict__,
                },
                reinit=True,
            )
            print(f"\n=== Baseline: {bl} ===")
            results = run_baseline(bl, tasks, cfg, device, args.folds)
            all_results.extend(results)

            summary_table = build_summary_table(results)
            wandb.log(summary_table)
            wandb.finish()

            # Save incrementally
            save_results(all_results, resume_path)

        # ---- Continuation models ----
        models = get_valid_continuation_models()
        remaining = [m for m in models if m["name"] not in completed_models]
        print(f"\nFound {len(models)} valid continuation models, {len(remaining)} remaining")
        for m in remaining:
            wandb.init(
                project=args.wandb_project,
                name=m["name"],
                config={
                    "model": m["name"],
                    "backbone": m["backbone"],
                    "init_mode": m["init_mode"],
                    "backbone_family": m["backbone_family"],
                    "view_weight": m["view_weight"],
                    "translation_weight": m["translation_weight"],
                    "val_mlm": m["val_mlm"],
                    "tasks": tasks,
                    "folds": args.folds,
                    **cfg.__dict__,
                },
                reinit=True,
            )
            print(f"\n=== {m['name']} (backbone={m['backbone']}, vw={m['view_weight']}, tw={m['translation_weight']}) ===")
            results = run_single_checkpoint(m["path"], tasks, cfg, device, args.folds)
            all_results.extend(results)

            summary_table = build_summary_table(results)
            wandb.log(summary_table)
            wandb.finish()

            # Save incrementally
            save_results(all_results, resume_path)

        output = args.output or resume_path

    elif args.checkpoint:
        name = args.model_alias or os.path.basename(os.path.dirname(args.checkpoint))
        if input_mode == INPUT_MODE_DUAL and args.model_alias is None:
            name = f"{name}_dualinput"
        elif input_mode == INPUT_MODE_CONCAT_ENCODER and args.model_alias is None:
            name = f"{name}_concatenc"
        run_id = _checkpoint_eval_run_id(name, tasks, input_mode)
        wandb.init(
            project=args.wandb_project,
            id=run_id,
            resume="allow",
            name=name,
            group=name,
            job_type="task_eval" if len(tasks) == 1 else "model_eval",
            config={
                "model": name,
                "checkpoint": args.checkpoint,
                "input_mode": input_mode,
                "task": tasks[0] if len(tasks) == 1 else "all",
                "tasks": tasks,
                "folds": args.folds,
                **cfg.__dict__,
            },
        )
        print(f"\n=== Checkpoint: {args.checkpoint} ===")
        results = run_single_checkpoint(
            args.checkpoint,
            tasks,
            cfg,
            device,
            args.folds,
            input_mode=input_mode,
            model_name_override=name,
        )
        all_results.extend(results)
        wandb.log(build_summary_table(results))
        wandb.finish()
        output = args.output or str(RESULTS_DIR / f"{name}.json")

    elif args.baseline:
        run_name = f"baseline_{args.baseline}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": run_name,
                "backbone": args.baseline,
                "init_mode": "baseline",
                "tasks": tasks,
                "folds": args.folds,
                **cfg.__dict__,
            },
        )
        print(f"\n=== Baseline: {args.baseline} ===")
        results = run_baseline(args.baseline, tasks, cfg, device, args.folds)
        all_results.extend(results)
        wandb.log(build_summary_table(results))
        wandb.finish()
        output = args.output or str(RESULTS_DIR / f"baseline_{args.baseline}.json")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    save_results(all_results, output)
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
