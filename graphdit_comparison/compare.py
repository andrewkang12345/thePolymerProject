#!/usr/bin/env python3
"""Compare pretrained polymer model (finetuned) vs Random Forest oracle
on Graph-DiT's O2/N2/CO2 gas permeability datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs

rdBase.DisableLog("rdApp.error")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from p1m_pretrain.paths import get_paths

PATHS = get_paths()
GRAPHDIT = PATHS.graphdit_root
CHECKPOINT = ROOT / "outputs" / "dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_vw1_tw1_e16" / "best.pt"
OUTPUT_DIR = ROOT / "graphdit_comparison"

TASKS = ["O2", "N2", "CO2"]


# ── Data loading & splitting (matches Graph-DiT's random_data_split) ──────
def load_task_data(task: str) -> pd.DataFrame:
    path = GRAPHDIT / "data" / "raw" / f"{task}.csv.gz"
    df = pd.read_csv(path)
    return df


def graphdit_random_split(n: int, seed: int = 42):
    """Reproduce Graph-DiT's random_data_split: 60/20/20."""
    full_idx = list(range(n))
    train_ratio, valid_ratio, test_ratio = 0.6, 0.2, 0.2
    train_index, test_index, _, _ = train_test_split(
        full_idx, full_idx, test_size=test_ratio, random_state=seed
    )
    train_index, val_index, _, _ = train_test_split(
        train_index, train_index,
        test_size=valid_ratio / (valid_ratio + train_ratio),
        random_state=seed,
    )
    return train_index, val_index, test_index


# ── Random Forest baseline ────────────────────────────────────────────────
def smiles_to_fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp_vec, arr)
    return arr


def train_and_eval_rf(
    train_smiles: list[str], train_y: np.ndarray,
    val_smiles: list[str], val_y: np.ndarray,
    test_smiles: list[str], test_y: np.ndarray,
) -> dict:
    """Train RF on train set, evaluate on val and test."""
    def encode(smiles_list):
        fps, valid = [], []
        for s in smiles_list:
            fp = smiles_to_fp(s)
            if fp is not None:
                fps.append(fp)
                valid.append(True)
            else:
                fps.append(np.zeros(2048))
                valid.append(False)
        return np.array(fps), np.array(valid)

    X_train, mask_train = encode(train_smiles)
    X_val, mask_val = encode(val_smiles)
    X_test, mask_test = encode(test_smiles)

    rf = RandomForestRegressor(random_state=0)
    rf.fit(X_train[mask_train], train_y[mask_train])

    val_pred = rf.predict(X_val)
    test_pred = rf.predict(X_test)

    return {
        "val_mae": mean_absolute_error(val_y[mask_val], val_pred[mask_val]),
        "val_r2": r2_score(val_y[mask_val], val_pred[mask_val]),
        "test_mae": mean_absolute_error(test_y[mask_test], test_pred[mask_test]),
        "test_r2": r2_score(test_y[mask_test], test_pred[mask_test]),
        "val_pred": val_pred,
        "test_pred": test_pred,
    }


def eval_presaved_rf(
    task: str,
    val_smiles: list[str], val_y: np.ndarray,
    test_smiles: list[str], test_y: np.ndarray,
) -> dict | None:
    """Evaluate pre-saved Graph-DiT RF (trained on all data) on val/test."""
    from joblib import load
    model_path = GRAPHDIT / "data" / "evaluator" / f"{task}.joblib"
    if not model_path.exists():
        return None
    try:
        rf = load(model_path)
    except Exception:
        return None

    def encode(smiles_list):
        fps, valid = [], []
        for s in smiles_list:
            fp = smiles_to_fp(s)
            if fp is not None:
                fps.append(fp)
                valid.append(True)
            else:
                fps.append(np.zeros(2048))
                valid.append(False)
        return np.array(fps), np.array(valid)

    X_val, mask_val = encode(val_smiles)
    X_test, mask_test = encode(test_smiles)
    val_pred = rf.predict(X_val)
    test_pred = rf.predict(X_test)

    return {
        "val_mae": mean_absolute_error(val_y[mask_val], val_pred[mask_val]),
        "val_r2": r2_score(val_y[mask_val], val_pred[mask_val]),
        "test_mae": mean_absolute_error(test_y[mask_test], test_pred[mask_test]),
        "test_r2": r2_score(test_y[mask_test], test_pred[mask_test]),
    }


# ── Pretrained model finetuning ──────────────────────────────────────────
def load_pretrained_backbone(checkpoint_path: str, device: str = "cpu"):
    """Load the dual-correctdeepchem backbone from a ContinuationModel checkpoint."""
    import evaluate_downstream as downstream
    backbone, config = downstream.load_roberta_model_from_continuation_ckpt(
        checkpoint_path, device=device
    )
    return backbone, config


class PSmilesRegressionDataset(Dataset):
    def __init__(self, smiles: list[str], targets: np.ndarray, tokenizer, max_len: int = 256):
        self.smiles = smiles
        self.targets = targets.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
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
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class PSmilesRegressor(nn.Module):
    """pSMILES-only regression model: backbone CLS -> regression head."""

    def __init__(self, backbone: nn.Module, hidden_size: int, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # language_id=0 → pSMILES branch
        language_ids = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.long)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, language_ids=language_ids)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_hidden).squeeze(-1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finetune_and_eval(
    backbone: nn.Module,
    tokenizer,
    train_smiles: list[str], train_y: np.ndarray,
    val_smiles: list[str], val_y: np.ndarray,
    test_smiles: list[str], test_y: np.ndarray,
    task_name: str,
    *,
    device: str = "cuda",
    lr_backbone: float = 2e-5,
    lr_head: float = 1e-3,
    batch_size: int = 16,
    num_epochs: int = 50,
    patience: int = 8,
    max_len: int = 256,
    seed: int = 42,
) -> dict:
    """Finetune and evaluate the pretrained model on a single regression task."""
    set_seed(seed)

    # Normalize targets
    train_mean = train_y.mean()
    train_std = train_y.std()
    if train_std < 1e-6:
        train_std = 1.0
    train_y_norm = (train_y - train_mean) / train_std
    val_y_norm = (val_y - train_mean) / train_std
    test_y_norm = (test_y - train_mean) / train_std

    train_ds = PSmilesRegressionDataset(train_smiles, train_y_norm, tokenizer, max_len)
    val_ds = PSmilesRegressionDataset(val_smiles, val_y_norm, tokenizer, max_len)
    test_ds = PSmilesRegressionDataset(test_smiles, test_y_norm, tokenizer, max_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = PSmilesRegressor(deepcopy(backbone), hidden_size=768, drop_rate=0.1).to(device)
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": lr_backbone, "weight_decay": 0.0},
        {"params": model.head.parameters(), "lr": lr_head, "weight_decay": 0.01},
    ])
    total_steps = len(train_dl) * num_epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler("cuda") if device.startswith("cuda") else None

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=scaler is not None):
                pred = model(ids, mask)
                loss = nn.functional.l1_loss(pred, target)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                with autocast(device_type="cuda", enabled=scaler is not None):
                    pred = model(ids, mask).float()
                val_preds.append(pred.cpu())
        val_preds_norm = torch.cat(val_preds).numpy()
        val_preds_orig = val_preds_norm * train_std + train_mean
        val_mae = mean_absolute_error(val_y, val_preds_orig)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{task_name}] Epoch {epoch+1:3d}  train_loss={epoch_loss/n_batches:.4f}  val_mae={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  [{task_name}] Early stopping at epoch {epoch+1}")
            break

    # Evaluate best model on test
    model.load_state_dict(best_state)
    model.to(device).eval()

    test_preds = []
    with torch.no_grad():
        for batch in test_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with autocast(device_type="cuda", enabled=scaler is not None):
                pred = model(ids, mask).float()
            test_preds.append(pred.cpu())
    test_preds_norm = torch.cat(test_preds).numpy()
    test_preds_orig = test_preds_norm * train_std + train_mean

    # Also get val predictions from best model
    val_preds = []
    with torch.no_grad():
        for batch in val_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with autocast(device_type="cuda", enabled=scaler is not None):
                pred = model(ids, mask).float()
            val_preds.append(pred.cpu())
    val_preds_orig = torch.cat(val_preds).numpy() * train_std + train_mean

    del model
    torch.cuda.empty_cache()

    return {
        "val_mae": mean_absolute_error(val_y, val_preds_orig),
        "val_r2": r2_score(val_y, val_preds_orig),
        "test_mae": mean_absolute_error(test_y, test_preds_orig),
        "test_r2": r2_score(test_y, test_preds_orig),
        "val_pred": val_preds_orig,
        "test_pred": test_preds_orig,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # Load pretrained backbone and tokenizer
    print("Loading pretrained backbone...")
    backbone, ckpt_config = load_pretrained_backbone(args.checkpoint, device="cpu")
    from p1m_pretrain.dual_tokenizer import load_original_deepchem_smiles_tokenizer
    tokenizer = load_original_deepchem_smiles_tokenizer(max_len=256)
    print(f"  Backbone: {ckpt_config['backbone']}, variant: {ckpt_config.get('scratch_variant')}")

    results = {}

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        # Load data
        df = load_task_data(task)
        smiles = df["smiles"].tolist()
        y = df[task].to_numpy(dtype=np.float64)

        # Filter NaN targets
        valid_mask = ~np.isnan(y)
        n_valid = valid_mask.sum()
        print(f"  Total: {len(df)}, valid: {n_valid}")

        # Create split (on valid data only, matching Graph-DiT)
        train_idx, val_idx, test_idx = graphdit_random_split(n_valid, seed=42)
        valid_indices = np.where(valid_mask)[0]

        train_smiles = [smiles[valid_indices[i]] for i in train_idx]
        val_smiles = [smiles[valid_indices[i]] for i in val_idx]
        test_smiles = [smiles[valid_indices[i]] for i in test_idx]
        train_y = y[valid_indices[train_idx]]
        val_y = y[valid_indices[val_idx]]
        test_y = y[valid_indices[test_idx]]

        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        print(f"  Target range: [{y[valid_mask].min():.2f}, {y[valid_mask].max():.2f}], mean={y[valid_mask].mean():.2f}")

        # ── Pre-saved RF (trained on all data) ──
        print("\n  --- Pre-saved RF (trained on ALL data, for reference) ---")
        presaved = eval_presaved_rf(task, val_smiles, val_y, test_smiles, test_y)
        if presaved:
            print(f"  Val  MAE={presaved['val_mae']:.4f}  R²={presaved['val_r2']:.4f}")
            print(f"  Test MAE={presaved['test_mae']:.4f}  R²={presaved['test_r2']:.4f}")
        else:
            print("  (could not load pre-saved model)")

        # ── RF retrained on train split ──
        print("\n  --- RF (retrained on train split) ---")
        rf_results = train_and_eval_rf(train_smiles, train_y, val_smiles, val_y, test_smiles, test_y)
        print(f"  Val  MAE={rf_results['val_mae']:.4f}  R²={rf_results['val_r2']:.4f}")
        print(f"  Test MAE={rf_results['test_mae']:.4f}  R²={rf_results['test_r2']:.4f}")

        # ── Finetuned pretrained model ──
        print(f"\n  --- Finetuned pretrained model (pSMILES only) ---")
        ft_results = finetune_and_eval(
            backbone, tokenizer,
            train_smiles, train_y,
            val_smiles, val_y,
            test_smiles, test_y,
            task_name=task,
            device=args.device,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )
        print(f"  Val  MAE={ft_results['val_mae']:.4f}  R²={ft_results['val_r2']:.4f}")
        print(f"  Test MAE={ft_results['test_mae']:.4f}  R²={ft_results['test_r2']:.4f}")

        results[task] = {
            "rf_presaved": presaved,
            "rf_retrained": {k: v for k, v in rf_results.items() if k not in ("val_pred", "test_pred")},
            "finetuned": {k: v for k, v in ft_results.items() if k not in ("val_pred", "test_pred")},
            "data_info": {
                "total": len(df), "valid": int(n_valid),
                "train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
                "target_mean": float(y[valid_mask].mean()),
                "target_std": float(y[valid_mask].std()),
                "target_min": float(y[valid_mask].min()),
                "target_max": float(y[valid_mask].max()),
            },
        }

    # ── Summary table ──
    print(f"\n\n{'='*80}")
    print("SUMMARY: Test MAE (lower is better)")
    print(f"{'='*80}")
    print(f"{'Task':<8} {'RF (all data)':<16} {'RF (train only)':<16} {'Finetuned':<16} {'Winner':<12}")
    print(f"{'-'*8} {'-'*16} {'-'*16} {'-'*16} {'-'*12}")
    for task in TASKS:
        r = results[task]
        ps_mae = r["rf_presaved"]["test_mae"] if r["rf_presaved"] else float("nan")
        rf_mae = r["rf_retrained"]["test_mae"]
        ft_mae = r["finetuned"]["test_mae"]
        winner = "Finetuned" if ft_mae < rf_mae else "RF"
        print(f"{task:<8} {ps_mae:<16.4f} {rf_mae:<16.4f} {ft_mae:<16.4f} {winner:<12}")

    print(f"\n{'='*80}")
    print("SUMMARY: Test R² (higher is better)")
    print(f"{'='*80}")
    print(f"{'Task':<8} {'RF (all data)':<16} {'RF (train only)':<16} {'Finetuned':<16} {'Winner':<12}")
    print(f"{'-'*8} {'-'*16} {'-'*16} {'-'*16} {'-'*12}")
    for task in TASKS:
        r = results[task]
        ps_r2 = r["rf_presaved"]["test_r2"] if r["rf_presaved"] else float("nan")
        rf_r2 = r["rf_retrained"]["test_r2"]
        ft_r2 = r["finetuned"]["test_r2"]
        winner = "Finetuned" if ft_r2 > rf_r2 else "RF"
        print(f"{task:<8} {ps_r2:<16.4f} {rf_r2:<16.4f} {ft_r2:<16.4f} {winner:<12}")

    print(f"\n{'='*80}")
    print("SUMMARY: Val MAE (lower is better)")
    print(f"{'='*80}")
    print(f"{'Task':<8} {'RF (train only)':<16} {'Finetuned':<16} {'Winner':<12}")
    print(f"{'-'*8} {'-'*16} {'-'*16} {'-'*12}")
    for task in TASKS:
        r = results[task]
        rf_mae = r["rf_retrained"]["val_mae"]
        ft_mae = r["finetuned"]["val_mae"]
        winner = "Finetuned" if ft_mae < rf_mae else "RF"
        print(f"{task:<8} {rf_mae:<16.4f} {ft_mae:<16.4f} {winner:<12}")

    # Save results
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
