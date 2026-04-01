#!/usr/bin/env python3
"""Additional experiments: pSELFIES finetuning + MMPolymer backbone finetuning."""

from __future__ import annotations

import json
import random
import sys
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path("/mnt/data/p1m_pretrain_experiments")
GRAPHDIT = Path("/mnt/data/Graph-DiT")
DUAL_CHECKPOINT = ROOT / "outputs" / "dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_vw1_tw1_e16" / "best.pt"
MM_CHECKPOINT = ROOT / "outputs" / "a10_targeted_mm_vw1p0_tw1p0" / "best.pt"
OUTPUT_DIR = ROOT / "graphdit_comparison"
TASKS = ["O2", "N2", "CO2"]

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graphdit_random_split(n: int, seed: int = 42):
    full_idx = list(range(n))
    train_idx, test_idx, _, _ = train_test_split(full_idx, full_idx, test_size=0.2, random_state=seed)
    train_idx, val_idx, _, _ = train_test_split(train_idx, train_idx, test_size=0.2/0.8, random_state=seed)
    return train_idx, val_idx, test_idx


# ── Datasets ──────────────────────────────────────────────────────────────
class TokenizedRegressionDataset(Dataset):
    def __init__(self, texts: list[str], targets: np.ndarray, tokenizer, max_len: int = 256):
        self.texts = texts
        self.targets = targets.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
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


# ── Model wrappers ───────────────────────────────────────────────────────
class DualBackboneRegressor(nn.Module):
    """Regression using dual backbone with a specific language_id."""
    def __init__(self, backbone, hidden_size: int, language_id: int, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.language_id = language_id
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        lang_ids = torch.full((input_ids.size(0),), self.language_id, device=input_ids.device, dtype=torch.long)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, language_ids=lang_ids)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)


class RobertaRegressor(nn.Module):
    """Regression using standard RoBERTa backbone (no language_ids)."""
    def __init__(self, backbone, hidden_size: int, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)


# ── Training loop ────────────────────────────────────────────────────────
def finetune_and_eval(
    model: nn.Module,
    tokenizer,
    train_texts: list[str], train_y: np.ndarray,
    val_texts: list[str], val_y: np.ndarray,
    test_texts: list[str], test_y: np.ndarray,
    task_name: str,
    label: str,
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
    set_seed(seed)

    train_mean, train_std = train_y.mean(), train_y.std()
    if train_std < 1e-6:
        train_std = 1.0
    train_y_n = (train_y - train_mean) / train_std

    train_ds = TokenizedRegressionDataset(train_texts, train_y_n, tokenizer, max_len)
    val_ds = TokenizedRegressionDataset(val_texts, (val_y - train_mean) / train_std, tokenizer, max_len)
    test_ds = TokenizedRegressionDataset(test_texts, (test_y - train_mean) / train_std, tokenizer, max_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = model.to(device)
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": lr_backbone, "weight_decay": 0.0},
        {"params": model.head.parameters(), "lr": lr_head, "weight_decay": 0.01},
    ])
    total_steps = len(train_dl) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)
    scaler = GradScaler("cuda") if device.startswith("cuda") else None

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
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

        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                with autocast(device_type="cuda", enabled=scaler is not None):
                    pred = model(ids, mask).float()
                val_preds.append(pred.cpu())
        val_preds_orig = torch.cat(val_preds).numpy() * train_std + train_mean
        val_mae = mean_absolute_error(val_y, val_preds_orig)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}/{task_name}] Epoch {epoch+1:3d}  loss={epoch_loss/n_batches:.4f}  val_mae={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"  [{label}/{task_name}] Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.to(device).eval()

    def predict(dl):
        preds = []
        with torch.no_grad():
            for batch in dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                with autocast(device_type="cuda", enabled=scaler is not None):
                    pred = model(ids, mask).float()
                preds.append(pred.cpu())
        return torch.cat(preds).numpy() * train_std + train_mean

    val_preds_orig = predict(val_dl)
    test_preds_orig = predict(test_dl)

    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    return {
        "val_mae": mean_absolute_error(val_y, val_preds_orig),
        "val_r2": r2_score(val_y, val_preds_orig),
        "test_mae": mean_absolute_error(test_y, test_preds_orig),
        "test_r2": r2_score(test_y, test_preds_orig),
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load previous results
    prev_path = OUTPUT_DIR / "results.json"
    results = json.loads(prev_path.read_text()) if prev_path.exists() else {}

    import evaluate_downstream as downstream
    from p1m_pretrain.dual_tokenizer import load_original_deepchem_smiles_tokenizer, load_pselfies_tokenizer
    from p1m_pretrain.pselfies import proxy_pselfies_from_psmiles
    from p1m_pretrain.upstream import load_polymer_smiles_tokenizer

    # ── Load backbones ────────────────────────────────────────────────────
    print("Loading dual backbone (for pSELFIES experiment)...")
    dual_backbone, dual_config = downstream.load_roberta_model_from_continuation_ckpt(str(DUAL_CHECKPOINT), device="cpu")
    selfies_tokenizer = load_pselfies_tokenizer(max_len=256)
    print(f"  {dual_config['backbone']}, variant={dual_config.get('scratch_variant')}")

    print("Loading MMPolymer backbone...")
    mm_backbone, mm_config = downstream.load_roberta_model_from_continuation_ckpt(str(MM_CHECKPOINT), device="cpu")
    mm_tokenizer = load_polymer_smiles_tokenizer(max_len=256)
    hidden_size_mm = mm_backbone.config.hidden_size
    print(f"  {mm_config['backbone']}, variant={mm_config.get('scratch_variant')}, hidden={hidden_size_mm}")

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        df = pd.read_csv(GRAPHDIT / "data" / "raw" / f"{task}.csv.gz")
        smiles = df["smiles"].tolist()
        y = df[task].to_numpy(dtype=np.float64)
        valid_mask = ~np.isnan(y)
        n_valid = valid_mask.sum()
        valid_indices = np.where(valid_mask)[0]

        train_idx, val_idx, test_idx = graphdit_random_split(n_valid, seed=42)

        train_smiles = [smiles[valid_indices[i]] for i in train_idx]
        val_smiles = [smiles[valid_indices[i]] for i in val_idx]
        test_smiles = [smiles[valid_indices[i]] for i in test_idx]
        train_y = y[valid_indices[train_idx]]
        val_y = y[valid_indices[val_idx]]
        test_y = y[valid_indices[test_idx]]

        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # ── Experiment 1: Dual backbone + pSELFIES ────────────────────────
        print("\n  --- Finetuned dual backbone (pSELFIES input) ---")
        # Convert SMILES → pSELFIES
        train_selfies = [proxy_pselfies_from_psmiles(s) for s in train_smiles]
        val_selfies = [proxy_pselfies_from_psmiles(s) for s in val_smiles]
        test_selfies = [proxy_pselfies_from_psmiles(s) for s in test_smiles]

        # Check conversion success
        train_ok = sum(1 for s in train_selfies if s is not None)
        val_ok = sum(1 for s in val_selfies if s is not None)
        test_ok = sum(1 for s in test_selfies if s is not None)
        print(f"  pSELFIES conversion: train={train_ok}/{len(train_selfies)}, val={val_ok}/{len(val_selfies)}, test={test_ok}/{len(test_selfies)}")

        # Replace None with empty string (will get UNK tokens)
        train_selfies = [s if s is not None else "" for s in train_selfies]
        val_selfies = [s if s is not None else "" for s in val_selfies]
        test_selfies = [s if s is not None else "" for s in test_selfies]

        model_selfies = DualBackboneRegressor(deepcopy(dual_backbone), hidden_size=768, language_id=1)
        selfies_results = finetune_and_eval(
            model_selfies, selfies_tokenizer,
            train_selfies, train_y,
            val_selfies, val_y,
            test_selfies, test_y,
            task_name=task, label="pSELFIES",
            device=device,
        )
        print(f"  Val  MAE={selfies_results['val_mae']:.4f}  R²={selfies_results['val_r2']:.4f}")
        print(f"  Test MAE={selfies_results['test_mae']:.4f}  R²={selfies_results['test_r2']:.4f}")

        # ── Experiment 2: MMPolymer backbone + pSMILES ────────────────────
        print(f"\n  --- Finetuned MMPolymer backbone (pSMILES input) ---")
        model_mm = RobertaRegressor(deepcopy(mm_backbone), hidden_size=hidden_size_mm)
        mm_results = finetune_and_eval(
            model_mm, mm_tokenizer,
            train_smiles, train_y,
            val_smiles, val_y,
            test_smiles, test_y,
            task_name=task, label="MMPolymer",
            device=device,
        )
        print(f"  Val  MAE={mm_results['val_mae']:.4f}  R²={mm_results['val_r2']:.4f}")
        print(f"  Test MAE={mm_results['test_mae']:.4f}  R²={mm_results['test_r2']:.4f}")

        # Store results
        if task not in results:
            results[task] = {}
        results[task]["finetuned_selfies"] = selfies_results
        results[task]["finetuned_mmpolymer"] = mm_results

    # ── Combined summary ──────────────────────────────────────────────────
    print(f"\n\n{'='*110}")
    print("FULL COMPARISON: Test MAE (lower is better)")
    print(f"{'='*110}")
    header = f"{'Task':<6} {'RF(ECFP4)':<12} {'RF(emb)':<12} {'FT pSMILES':<12} {'FT pSELFIES':<14} {'FT MMPoly':<12} {'Best':<20}"
    print(header)
    print("-" * 110)
    for task in TASKS:
        r = results[task]
        rf = r.get("rf_retrained", {}).get("test_mae", float("nan"))
        emb = r.get("rf_pretrained_emb", {}).get("test_mae", float("nan"))
        ft_sm = r.get("finetuned", {}).get("test_mae", float("nan"))
        ft_sf = r.get("finetuned_selfies", {}).get("test_mae", float("nan"))
        ft_mm = r.get("finetuned_mmpolymer", {}).get("test_mae", float("nan"))
        scores = {"RF(ECFP4)": rf, "RF(emb)": emb, "FT pSMILES": ft_sm, "FT pSELFIES": ft_sf, "FT MMPoly": ft_mm}
        best = min(scores, key=scores.get)
        print(f"{task:<6} {rf:<12.2f} {emb:<12.2f} {ft_sm:<12.2f} {ft_sf:<14.2f} {ft_mm:<12.2f} {best:<20}")

    print(f"\n{'='*110}")
    print("FULL COMPARISON: Test R² (higher is better)")
    print(f"{'='*110}")
    header = f"{'Task':<6} {'RF(ECFP4)':<12} {'RF(emb)':<12} {'FT pSMILES':<12} {'FT pSELFIES':<14} {'FT MMPoly':<12} {'Best':<20}"
    print(header)
    print("-" * 110)
    for task in TASKS:
        r = results[task]
        rf = r.get("rf_retrained", {}).get("test_r2", float("nan"))
        emb = r.get("rf_pretrained_emb", {}).get("test_r2", float("nan"))
        ft_sm = r.get("finetuned", {}).get("test_r2", float("nan"))
        ft_sf = r.get("finetuned_selfies", {}).get("test_r2", float("nan"))
        ft_mm = r.get("finetuned_mmpolymer", {}).get("test_r2", float("nan"))
        scores = {"RF(ECFP4)": rf, "RF(emb)": emb, "FT pSMILES": ft_sm, "FT pSELFIES": ft_sf, "FT MMPoly": ft_mm}
        best = max(scores, key=scores.get)
        print(f"{task:<6} {rf:<12.4f} {emb:<12.4f} {ft_sm:<12.4f} {ft_sf:<14.4f} {ft_mm:<12.4f} {best:<20}")

    print(f"\n{'='*110}")
    print("FULL COMPARISON: Val MAE (lower is better)")
    print(f"{'='*110}")
    header = f"{'Task':<6} {'RF(ECFP4)':<12} {'RF(emb)':<12} {'FT pSMILES':<12} {'FT pSELFIES':<14} {'FT MMPoly':<12} {'Best':<20}"
    print(header)
    print("-" * 110)
    for task in TASKS:
        r = results[task]
        rf = r.get("rf_retrained", {}).get("val_mae", float("nan"))
        emb = r.get("rf_pretrained_emb", {}).get("val_mae", float("nan"))
        ft_sm = r.get("finetuned", {}).get("val_mae", float("nan"))
        ft_sf = r.get("finetuned_selfies", {}).get("val_mae", float("nan"))
        ft_mm = r.get("finetuned_mmpolymer", {}).get("val_mae", float("nan"))
        scores = {"RF(ECFP4)": rf, "RF(emb)": emb, "FT pSMILES": ft_sm, "FT pSELFIES": ft_sf, "FT MMPoly": ft_mm}
        best = min(scores, key=scores.get)
        print(f"{task:<6} {rf:<12.2f} {emb:<12.2f} {ft_sm:<12.2f} {ft_sf:<14.2f} {ft_mm:<12.2f} {best:<20}")

    # Save
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
