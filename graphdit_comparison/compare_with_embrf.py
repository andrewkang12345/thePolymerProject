#!/usr/bin/env python3
"""Add RF-on-pretrained-embeddings comparison to Graph-DiT oracle comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from p1m_pretrain.paths import get_paths

PATHS = get_paths()
GRAPHDIT = PATHS.graphdit_root
CHECKPOINT = ROOT / "outputs" / "dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_vw1_tw1_e16" / "best.pt"
OUTPUT_DIR = ROOT / "graphdit_comparison"
TASKS = ["O2", "N2", "CO2"]


def graphdit_random_split(n: int, seed: int = 42):
    full_idx = list(range(n))
    train_idx, test_idx, _, _ = train_test_split(full_idx, full_idx, test_size=0.2, random_state=seed)
    train_idx, val_idx, _, _ = train_test_split(train_idx, train_idx, test_size=0.2/0.8, random_state=seed)
    return train_idx, val_idx, test_idx


@torch.no_grad()
def extract_embeddings(backbone, tokenizer, smiles_list, device="cuda", batch_size=64, max_len=256):
    """Extract CLS embeddings from the pretrained backbone (pSMILES branch)."""
    backbone.eval()
    backbone.to(device)
    all_embeddings = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i : i + batch_size]
        encodings = tokenizer(
            batch_smiles,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        language_ids = torch.zeros(input_ids.size(0), device=device, dtype=torch.long)

        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, language_ids=language_ids)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    return np.concatenate(all_embeddings, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load backbone & tokenizer
    print("Loading pretrained backbone...")
    import evaluate_downstream as downstream
    backbone, ckpt_config = downstream.load_roberta_model_from_continuation_ckpt(str(CHECKPOINT), device="cpu")
    from p1m_pretrain.dual_tokenizer import load_original_deepchem_smiles_tokenizer
    tokenizer = load_original_deepchem_smiles_tokenizer(max_len=256)
    print(f"  Backbone: {ckpt_config['backbone']}, variant: {ckpt_config.get('scratch_variant')}")

    # Load previous results
    prev_results_path = OUTPUT_DIR / "results.json"
    if prev_results_path.exists():
        with open(prev_results_path) as f:
            prev_results = json.load(f)
    else:
        prev_results = {}

    emb_rf_results = {}

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

        # Extract embeddings
        print("  Extracting pretrained embeddings...")
        all_smiles = train_smiles + val_smiles + test_smiles
        all_emb = extract_embeddings(backbone, tokenizer, all_smiles, device=device)
        n_train = len(train_idx)
        n_val = len(val_idx)
        X_train = all_emb[:n_train]
        X_val = all_emb[n_train : n_train + n_val]
        X_test = all_emb[n_train + n_val:]
        print(f"  Embedding shape: {X_train.shape}")

        # Train RF on embeddings
        print("  Training RF on pretrained embeddings...")
        rf = RandomForestRegressor(random_state=0, n_estimators=100)
        rf.fit(X_train, train_y)

        val_pred = rf.predict(X_val)
        test_pred = rf.predict(X_test)

        val_mae = mean_absolute_error(val_y, val_pred)
        val_r2 = r2_score(val_y, val_pred)
        test_mae = mean_absolute_error(test_y, test_pred)
        test_r2 = r2_score(test_y, test_pred)

        print(f"  Val  MAE={val_mae:.4f}  R²={val_r2:.4f}")
        print(f"  Test MAE={test_mae:.4f}  R²={test_r2:.4f}")

        emb_rf_results[task] = {
            "val_mae": val_mae, "val_r2": val_r2,
            "test_mae": test_mae, "test_r2": test_r2,
        }

    # Print combined summary table
    print(f"\n\n{'='*90}")
    print("COMBINED SUMMARY: Test MAE (lower is better)")
    print(f"{'='*90}")
    print(f"{'Task':<6} {'RF (ECFP4)':<14} {'RF (pretrained emb)':<22} {'Finetuned':<14} {'Best':<20}")
    print(f"{'-'*6} {'-'*14} {'-'*22} {'-'*14} {'-'*20}")
    for task in TASKS:
        rf_mae = prev_results.get(task, {}).get("rf_retrained", {}).get("test_mae", float("nan"))
        emb_mae = emb_rf_results[task]["test_mae"]
        ft_mae = prev_results.get(task, {}).get("finetuned", {}).get("test_mae", float("nan"))
        scores = {"RF (ECFP4)": rf_mae, "RF (pretrained emb)": emb_mae, "Finetuned": ft_mae}
        best = min(scores, key=scores.get)
        print(f"{task:<6} {rf_mae:<14.4f} {emb_mae:<22.4f} {ft_mae:<14.4f} {best:<20}")

    print(f"\n{'='*90}")
    print("COMBINED SUMMARY: Test R² (higher is better)")
    print(f"{'='*90}")
    print(f"{'Task':<6} {'RF (ECFP4)':<14} {'RF (pretrained emb)':<22} {'Finetuned':<14} {'Best':<20}")
    print(f"{'-'*6} {'-'*14} {'-'*22} {'-'*14} {'-'*20}")
    for task in TASKS:
        rf_r2 = prev_results.get(task, {}).get("rf_retrained", {}).get("test_r2", float("nan"))
        emb_r2 = emb_rf_results[task]["test_r2"]
        ft_r2 = prev_results.get(task, {}).get("finetuned", {}).get("test_r2", float("nan"))
        scores = {"RF (ECFP4)": rf_r2, "RF (pretrained emb)": emb_r2, "Finetuned": ft_r2}
        best = max(scores, key=scores.get)
        print(f"{task:<6} {rf_r2:<14.4f} {emb_r2:<22.4f} {ft_r2:<14.4f} {best:<20}")

    print(f"\n{'='*90}")
    print("COMBINED SUMMARY: Val MAE (lower is better)")
    print(f"{'='*90}")
    print(f"{'Task':<6} {'RF (ECFP4)':<14} {'RF (pretrained emb)':<22} {'Finetuned':<14} {'Best':<20}")
    print(f"{'-'*6} {'-'*14} {'-'*22} {'-'*14} {'-'*20}")
    for task in TASKS:
        rf_mae = prev_results.get(task, {}).get("rf_retrained", {}).get("val_mae", float("nan"))
        emb_mae = emb_rf_results[task]["val_mae"]
        ft_mae = prev_results.get(task, {}).get("finetuned", {}).get("val_mae", float("nan"))
        scores = {"RF (ECFP4)": rf_mae, "RF (pretrained emb)": emb_mae, "Finetuned": ft_mae}
        best = min(scores, key=scores.get)
        print(f"{task:<6} {rf_mae:<14.4f} {emb_mae:<22.4f} {ft_mae:<14.4f} {best:<20}")

    # Save updated results
    for task in TASKS:
        if task in prev_results:
            prev_results[task]["rf_pretrained_emb"] = emb_rf_results[task]
        else:
            prev_results[task] = {"rf_pretrained_emb": emb_rf_results[task]}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(prev_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
