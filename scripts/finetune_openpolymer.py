#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.model_selection import KFold
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_downstream as downstream  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from p1m_pretrain.paths import get_paths  # noqa: E402
from p1m_pretrain.pselfies import proxy_pselfies_from_psmiles  # noqa: E402

PATHS = get_paths()

PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
INPUT_MODE_DUAL = "dual"
INPUT_MODE_CONCAT_ENCODER = "concat_encoder"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class OpenPolymerConfig:
    checkpoint: str
    train_csv: str
    test_csv: str
    output_dir: str
    run_name: str
    wandb_project: str
    folds: int = 5
    batch_size: int = 16
    num_epochs: int = 25
    patience: int = 5
    lr_backbone: float = 2e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    drop_rate: float = 0.1
    max_len: int = 256
    num_workers: int = 4
    seed: int = 42
    use_amp: bool = True
    multi_gpu: bool = True
    input_mode: str = INPUT_MODE_DUAL
    tg_fahrenheit_postprocess: bool = False


class TargetNormalizer:
    def __init__(self, means: np.ndarray, stds: np.ndarray):
        self.means = means.astype(np.float32)
        self.stds = stds.astype(np.float32)

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray) -> "TargetNormalizer":
        means = np.zeros(values.shape[1], dtype=np.float32)
        stds = np.ones(values.shape[1], dtype=np.float32)
        for idx in range(values.shape[1]):
            valid = mask[:, idx]
            if not np.any(valid):
                continue
            task_values = values[valid, idx].astype(np.float32)
            means[idx] = float(task_values.mean())
            std = float(task_values.std())
            stds[idx] = std if std > 1e-6 else 1.0
        return cls(means, stds)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.means[None, :]) / self.stds[None, :]

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.stds[None, :] + self.means[None, :]


def build_competition_like_weights(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = mask.sum(axis=0).astype(np.float64)
    ranges = np.ones(values.shape[1], dtype=np.float64)
    for idx in range(values.shape[1]):
        valid = mask[:, idx]
        if not np.any(valid):
            continue
        task_values = values[valid, idx].astype(np.float64)
        task_range = float(task_values.max() - task_values.min())
        ranges[idx] = max(task_range, 1e-8)

    rarity = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    rarity *= len(PROPERTIES) / rarity.sum()
    weights = rarity / ranges
    return weights.astype(np.float32), counts.astype(np.int64), ranges.astype(np.float32)


def compute_weighted_mae(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> float:
    valid_rows = mask.any(axis=1)
    if not np.any(valid_rows):
        return float("nan")
    safe_true = np.where(mask, true_values, 0.0)
    safe_pred = np.where(mask, pred_values, 0.0)
    abs_error = np.abs(safe_pred - safe_true) * mask.astype(np.float32)
    row_score = (abs_error * weights[None, :]).sum(axis=1)
    return float(row_score[valid_rows].mean())


def compute_metrics(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    ranges: np.ndarray,
) -> dict:
    metrics = {
        "wmae": compute_weighted_mae(true_values, pred_values, mask, weights),
        "num_rows": int(mask.any(axis=1).sum()),
    }
    for idx, prop in enumerate(PROPERTIES):
        valid = mask[:, idx]
        if not np.any(valid):
            metrics[f"{prop}_count"] = 0
            continue
        task_true = true_values[valid, idx]
        task_pred = pred_values[valid, idx]
        err = np.abs(task_pred - task_true)
        rmse = np.sqrt(np.mean((task_pred - task_true) ** 2))
        metrics[f"{prop}_count"] = int(valid.sum())
        metrics[f"{prop}_mae"] = float(err.mean())
        metrics[f"{prop}_rmse"] = float(rmse)
        metrics[f"{prop}_normalized_mae"] = float(err.mean() / max(float(ranges[idx]), 1e-8))
    return metrics


def maybe_postprocess_tg(pred_values: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return pred_values
    updated = pred_values.copy()
    updated[:, 0] = (9.0 / 5.0) * updated[:, 0] + 32.0
    return updated


class OpenPolymerDualDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        smiles: list[str],
        pselfies: list[str],
        targets: np.ndarray,
        mask: np.ndarray,
        smiles_tokenizer,
        selfies_tokenizer,
        max_len: int,
    ):
        self.ids = ids
        self.smiles = smiles
        self.pselfies = pselfies
        self.targets = targets.astype(np.float32)
        self.mask = mask.astype(bool)
        self.smiles_tokenizer = smiles_tokenizer
        self.selfies_tokenizer = selfies_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
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
            "id": int(self.ids[idx]),
            "smiles_input_ids": smiles_encoding["input_ids"].flatten(),
            "smiles_attention_mask": smiles_encoding["attention_mask"].flatten(),
            "selfies_input_ids": selfies_encoding["input_ids"].flatten(),
            "selfies_attention_mask": selfies_encoding["attention_mask"].flatten(),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask[idx], dtype=torch.bool),
        }


class DualInputMultiTaskRegression(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_tasks: int, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_tasks),
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


class ConcatEncoderMultiTaskRegression(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_tasks: int, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_tasks),
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


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def forward_model(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    return model(
        batch["smiles_input_ids"].to(device),
        batch["smiles_attention_mask"].to(device),
        batch["selfies_input_ids"].to(device),
        batch["selfies_attention_mask"].to(device),
    )


def multitask_l1_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    task_losses = []
    for task_idx in range(preds.size(1)):
        task_mask = mask[:, task_idx]
        if torch.any(task_mask):
            task_losses.append(torch.abs(preds[task_mask, task_idx] - targets[task_mask, task_idx]).mean())
    if not task_losses:
        return preds.sum() * 0.0
    return torch.stack(task_losses).mean()


def predict_scaled(model: nn.Module, dataloader: DataLoader, device: str, use_amp: bool) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            with autocast(device_type="cuda", enabled=use_amp and device.startswith("cuda")):
                preds = forward_model(model, batch, device).float()
            outputs.append(preds.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: str,
    scaler_amp: GradScaler | None,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in dataloader:
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp and device.startswith("cuda")):
            preds = forward_model(model, batch, device)
            loss = multitask_l1_loss(preds, targets, mask)
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
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def build_model(base_backbone: nn.Module, cfg: OpenPolymerConfig) -> nn.Module:
    backbone = deepcopy(base_backbone)
    hidden_size = backbone.config.hidden_size
    if cfg.input_mode == INPUT_MODE_DUAL:
        return DualInputMultiTaskRegression(backbone, hidden_size, len(PROPERTIES), cfg.drop_rate)
    if cfg.input_mode == INPUT_MODE_CONCAT_ENCODER:
        return ConcatEncoderMultiTaskRegression(backbone, hidden_size, len(PROPERTIES), cfg.drop_rate)
    raise ValueError(f"Unsupported input mode: {cfg.input_mode}")


def load_openpolymer_data(train_csv: str, test_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df = train_df[["id", "SMILES", *PROPERTIES]].copy()
    test_df = test_df[["id", "SMILES"]].copy()

    train_df["pselfies"] = [proxy_pselfies_from_psmiles(smiles) for smiles in train_df["SMILES"].astype(str)]
    test_df["pselfies"] = [proxy_pselfies_from_psmiles(smiles) for smiles in test_df["SMILES"].astype(str)]

    if train_df["pselfies"].isna().any() or test_df["pselfies"].isna().any():
        raise RuntimeError("Failed to convert some openPolymer SMILES strings to pSELFIES")

    mask = train_df[PROPERTIES].notna().values
    train_df = train_df.loc[mask.any(axis=1)].reset_index(drop=True)
    return train_df, test_df


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def log_visualizations(
    run_name: str,
    train_df: pd.DataFrame,
    oof_preds: np.ndarray,
    test_df: pd.DataFrame,
    test_preds: np.ndarray,
    mask: np.ndarray,
) -> None:
    for idx, prop in enumerate(PROPERTIES):
        valid = mask[:, idx]
        if not np.any(valid):
            continue
        table_df = pd.DataFrame(
            {
                "id": train_df.loc[valid, "id"].astype(int).values,
                "true": train_df.loc[valid, prop].astype(float).values,
                "pred": oof_preds[valid, idx],
            }
        )
        table = wandb.Table(dataframe=table_df)
        wandb.log(
            {
                f"oof/{prop}_parity_table": table,
                f"oof/{prop}_parity": wandb.plot.scatter(table, "true", "pred", title=f"{run_name} {prop} OOF parity"),
            }
        )

    test_pred_df = pd.DataFrame(
        {
            "id": test_df["id"].astype(int).values,
            "SMILES": test_df["SMILES"].astype(str).values,
            **{prop: test_preds[:, idx] for idx, prop in enumerate(PROPERTIES)},
        }
    )
    wandb.log({"test/predictions": wandb.Table(dataframe=test_pred_df)})
    for idx, prop in enumerate(PROPERTIES):
        wandb.log({f"test/{prop}_hist": wandb.Histogram(test_preds[:, idx])})


def run_fold(
    fold_idx: int,
    train_index: np.ndarray,
    val_index: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_backbone: nn.Module,
    smiles_tokenizer,
    selfies_tokenizer,
    cfg: OpenPolymerConfig,
    device: str,
    global_weights: np.ndarray,
    global_ranges: np.ndarray,
) -> tuple[np.ndarray, dict, np.ndarray]:
    fold_train = train_df.iloc[train_index].reset_index(drop=True)
    fold_val = train_df.iloc[val_index].reset_index(drop=True)

    train_targets_orig = fold_train[PROPERTIES].astype(np.float32).values
    val_targets_orig = fold_val[PROPERTIES].astype(np.float32).values
    train_mask = fold_train[PROPERTIES].notna().values
    val_mask = fold_val[PROPERTIES].notna().values

    normalizer = TargetNormalizer.fit(train_targets_orig, train_mask)
    train_targets = np.nan_to_num(normalizer.transform(train_targets_orig), nan=0.0)
    val_targets = np.nan_to_num(normalizer.transform(val_targets_orig), nan=0.0)
    test_targets = np.zeros((len(test_df), len(PROPERTIES)), dtype=np.float32)
    test_mask = np.zeros((len(test_df), len(PROPERTIES)), dtype=bool)

    train_ds = OpenPolymerDualDataset(
        fold_train["id"].values,
        fold_train["SMILES"].astype(str).tolist(),
        fold_train["pselfies"].astype(str).tolist(),
        train_targets,
        train_mask,
        smiles_tokenizer,
        selfies_tokenizer,
        cfg.max_len,
    )
    val_ds = OpenPolymerDualDataset(
        fold_val["id"].values,
        fold_val["SMILES"].astype(str).tolist(),
        fold_val["pselfies"].astype(str).tolist(),
        val_targets,
        val_mask,
        smiles_tokenizer,
        selfies_tokenizer,
        cfg.max_len,
    )
    test_ds = OpenPolymerDualDataset(
        test_df["id"].values,
        test_df["SMILES"].astype(str).tolist(),
        test_df["pselfies"].astype(str).tolist(),
        test_targets,
        test_mask,
        smiles_tokenizer,
        selfies_tokenizer,
        cfg.max_len,
    )

    train_dl = build_dataloader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = build_dataloader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_dl = build_dataloader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(base_backbone, cfg).to(device)
    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.lr_backbone, "weight_decay": 0.0},
            {"params": model.regressor.parameters(), "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        ]
    )
    if cfg.multi_gpu and device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    total_steps = max(len(train_dl), 1) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler_amp = GradScaler("cuda") if cfg.use_amp and device.startswith("cuda") else None

    best_val_wmae = float("inf")
    best_summary = {}
    best_val_preds = None
    best_test_preds = None
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, device, scaler_amp, cfg.use_amp)
        val_preds_scaled = predict_scaled(model, val_dl, device, cfg.use_amp)
        val_preds = maybe_postprocess_tg(normalizer.inverse_transform(val_preds_scaled), cfg.tg_fahrenheit_postprocess)
        val_metrics = compute_metrics(val_targets_orig, val_preds, val_mask, global_weights, global_ranges)

        log_payload = {
            "fold": fold_idx + 1,
            "epoch": epoch + 1,
            f"fold_{fold_idx+1}/train_loss": train_loss,
            f"fold_{fold_idx+1}/val_wmae": val_metrics["wmae"],
        }
        for prop in PROPERTIES:
            if f"{prop}_mae" in val_metrics:
                log_payload[f"fold_{fold_idx+1}/{prop}_mae"] = val_metrics[f"{prop}_mae"]
                log_payload[f"fold_{fold_idx+1}/{prop}_normalized_mae"] = val_metrics[f"{prop}_normalized_mae"]
        wandb.log(log_payload)

        if val_metrics["wmae"] < best_val_wmae:
            best_val_wmae = val_metrics["wmae"]
            best_val_preds = val_preds.copy()
            best_test_preds = maybe_postprocess_tg(
                normalizer.inverse_transform(predict_scaled(model, test_dl, device, cfg.use_amp)),
                cfg.tg_fahrenheit_postprocess,
            )
            best_summary = {
                "fold": fold_idx + 1,
                "best_epoch": epoch + 1,
                "best_val_wmae": best_val_wmae,
                **{k: v for k, v in val_metrics.items() if k != "num_rows"},
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            break

    if best_val_preds is None or best_test_preds is None:
        raise RuntimeError(f"Fold {fold_idx + 1} did not produce any predictions")

    del model, optimizer, scheduler, scaler_amp
    torch.cuda.empty_cache()

    return best_val_preds, best_summary, best_test_preds


def load_dual_correctdeepchem_bundle(checkpoint: str, max_len: int):
    backbone, model_config = downstream.load_roberta_model_from_continuation_ckpt(checkpoint, device="cpu")
    backbone_name = model_config.get("backbone", "")
    if backbone_name != "dual_correctdeepchem_pselfies_shared":
        raise ValueError(f"openPolymer script currently supports dual_correctdeepchem checkpoints only, got: {backbone_name}")

    from p1m_pretrain.dual_tokenizer import load_original_deepchem_smiles_tokenizer, load_pselfies_tokenizer

    smiles_tokenizer = load_original_deepchem_smiles_tokenizer(max_len)
    selfies_tokenizer = load_pselfies_tokenizer(max_len)
    return backbone, model_config, smiles_tokenizer, selfies_tokenizer


def save_outputs(
    cfg: OpenPolymerConfig,
    train_df: pd.DataFrame,
    oof_preds: np.ndarray,
    oof_metrics: dict,
    fold_summaries: list[dict],
    test_df: pd.DataFrame,
    test_preds: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    ranges: np.ndarray,
) -> dict[str, str]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_df = pd.DataFrame({"id": test_df["id"].astype(int).values})
    for idx, prop in enumerate(PROPERTIES):
        submission_df[prop] = test_preds[:, idx]

    oof_df = train_df[["id", "SMILES", *PROPERTIES]].copy()
    for idx, prop in enumerate(PROPERTIES):
        oof_df[f"pred_{prop}"] = oof_preds[:, idx]
        valid = oof_df[prop].notna()
        oof_df.loc[valid, f"abs_err_{prop}"] = np.abs(oof_df.loc[valid, f"pred_{prop}"] - oof_df.loc[valid, prop])

    submission_path = output_dir / "submission.csv"
    oof_path = output_dir / "oof_predictions.csv"
    summary_path = output_dir / "summary.json"

    submission_df.to_csv(submission_path, index=False)
    oof_df.to_csv(oof_path, index=False)

    summary = {
        "config": asdict(cfg),
        "oof_metrics": oof_metrics,
        "fold_summaries": fold_summaries,
        "property_weights": {prop: float(weights[idx]) for idx, prop in enumerate(PROPERTIES)},
        "property_counts": {prop: int(counts[idx]) for idx, prop in enumerate(PROPERTIES)},
        "property_ranges": {prop: float(ranges[idx]) for idx, prop in enumerate(PROPERTIES)},
        "submission_path": str(submission_path),
        "oof_predictions_path": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "submission": str(submission_path),
        "oof_predictions": str(oof_path),
        "summary": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune the e16 dual-corrected DeepChem checkpoint on openPolymer")
    parser.add_argument(
        "--checkpoint",
        default=str(PATHS.outputs_dir / "dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_vw1_tw1_e16" / "best.pt"),
    )
    parser.add_argument("--train-csv", default=str(PATHS.openpolymer_dir / "train.csv"))
    parser.add_argument("--test-csv", default=str(PATHS.openpolymer_dir / "test.csv"))
    parser.add_argument("--output-dir", default=str(PATHS.outputs_dir / "openpolymer_dual_correctdeepchem_e16"))
    parser.add_argument("--run-name", default="dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_vw1_tw1_e16_openpolymer_cv5")
    parser.add_argument("--wandb-project", default="openpolymer-transfer")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-multi-gpu", action="store_true")
    parser.add_argument("--input-mode", choices=[INPUT_MODE_DUAL, INPUT_MODE_CONCAT_ENCODER], default=INPUT_MODE_DUAL)
    parser.add_argument("--tg-fahrenheit-postprocess", action="store_true", help='Apply pred["Tg"] = (9/5) * pred["Tg"] + 32 to saved/evaluated predictions')
    args = parser.parse_args()

    cfg = OpenPolymerConfig(
        checkpoint=args.checkpoint,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        folds=args.folds,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        drop_rate=args.drop_rate,
        max_len=args.max_len,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=not args.no_amp,
        multi_gpu=not args.no_multi_gpu,
        input_mode=args.input_mode,
        tg_fahrenheit_postprocess=args.tg_fahrenheit_postprocess,
    )

    set_seed(cfg.seed)
    train_df, test_df = load_openpolymer_data(cfg.train_csv, cfg.test_csv)
    all_true = train_df[PROPERTIES].astype(np.float32).values
    all_mask = train_df[PROPERTIES].notna().values
    weights, counts, ranges = build_competition_like_weights(all_true, all_mask)

    backbone, model_config, smiles_tokenizer, selfies_tokenizer = load_dual_correctdeepchem_bundle(cfg.checkpoint, cfg.max_len)

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config={
            **asdict(cfg),
            "backbone": model_config.get("backbone"),
            "scratch_variant": model_config.get("scratch_variant"),
            "note": "OOF train proxy wMAE is used because openPolymer/test.csv has no labels locally.",
        },
    )

    oof_preds = np.zeros((len(train_df), len(PROPERTIES)), dtype=np.float32)
    test_fold_preds = []
    fold_summaries = []

    kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    start_time = time.time()
    for fold_idx, (train_index, val_index) in enumerate(kf.split(train_df)):
        print(f"=== Fold {fold_idx + 1}/{cfg.folds} ===")
        fold_val_preds, fold_summary, fold_test_preds = run_fold(
            fold_idx,
            train_index,
            val_index,
            train_df,
            test_df,
            backbone,
            smiles_tokenizer,
            selfies_tokenizer,
            cfg,
            args.device,
            weights,
            ranges,
        )
        oof_preds[val_index] = fold_val_preds
        test_fold_preds.append(fold_test_preds)
        fold_summaries.append(fold_summary)
        wandb.log({f"fold_{fold_idx+1}/best_val_wmae": fold_summary["best_val_wmae"], f"fold_{fold_idx+1}/best_epoch": fold_summary["best_epoch"]})

    test_preds = np.mean(np.stack(test_fold_preds, axis=0), axis=0)
    oof_metrics = compute_metrics(all_true, oof_preds, all_mask, weights, ranges)
    oof_metrics["elapsed_minutes"] = (time.time() - start_time) / 60.0

    output_paths = save_outputs(
        cfg,
        train_df,
        oof_preds,
        oof_metrics,
        fold_summaries,
        test_df,
        test_preds,
        weights,
        counts,
        ranges,
    )
    log_visualizations(cfg.run_name, train_df, oof_preds, test_df, test_preds, all_mask)

    final_log = {
        "oof/wmae_train_proxy": oof_metrics["wmae"],
        "oof/num_rows": oof_metrics["num_rows"],
        "runtime_minutes": oof_metrics["elapsed_minutes"],
    }
    for prop in PROPERTIES:
        if f"{prop}_mae" in oof_metrics:
            final_log[f"oof/{prop}_mae"] = oof_metrics[f"{prop}_mae"]
            final_log[f"oof/{prop}_rmse"] = oof_metrics[f"{prop}_rmse"]
            final_log[f"oof/{prop}_normalized_mae"] = oof_metrics[f"{prop}_normalized_mae"]
    wandb.log(final_log)
    wandb.summary["submission_path"] = output_paths["submission"]
    wandb.summary["oof_predictions_path"] = output_paths["oof_predictions"]
    wandb.summary["summary_path"] = output_paths["summary"]
    wandb.summary["oof_wmae_train_proxy"] = oof_metrics["wmae"]
    wandb.save(output_paths["submission"], base_path=cfg.output_dir)
    wandb.save(output_paths["oof_predictions"], base_path=cfg.output_dir)
    wandb.save(output_paths["summary"], base_path=cfg.output_dir)
    wandb.finish()

    print(json.dumps({"oof_metrics": oof_metrics, "output_paths": output_paths}, indent=2))


if __name__ == "__main__":
    main()
