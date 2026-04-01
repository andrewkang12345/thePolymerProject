#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1m_pretrain.paths import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="p1m_wide_search_trial_")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = get_paths().outputs_dir
    rows = []
    for run_dir in sorted(root.glob(f"{args.prefix}*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as handle:
            payload = json.load(handle)
        rows.append(
            {
                "run": run_dir.name,
                "backbone": payload["config"]["backbone"],
                "init_mode": payload["config"]["init_mode"],
                "backbone_family": payload["config"]["backbone_family"],
                "scratch_variant": payload["config"]["scratch_variant"],
                "position_embedding_type": payload["config"]["position_embedding_type"],
                "attention_variant": payload["config"]["attention_variant"],
                "translation_decoder_type": payload["config"]["translation_decoder_type"],
                "view_weight": payload["config"]["view_weight"],
                "translation_weight": payload["config"]["translation_weight"],
                "learning_rate": payload["config"]["learning_rate"],
                "batch_size": payload["config"]["batch_size"],
                "best_mlm_step": payload["best_mlm_metrics"]["step"],
                "best_mlm_loss": payload["best_mlm_metrics"]["val_mlm_loss"],
                "best_combined_step": payload["best_combined_metrics"]["step"],
                "best_combined_loss": payload["best_combined_metrics"]["val_combined_loss"],
                "best_combined_translation_acc": payload["best_combined_metrics"]["val_translation_token_accuracy"],
                "best_combined_view_top1": payload["best_combined_metrics"]["val_view_top1"],
            }
        )
    rows.sort(key=lambda row: row["best_combined_loss"])
    out_json = root / "search_report.json"
    out_csv = root / "search_report.csv"
    out_json.write_text(json.dumps(rows, indent=2))
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(out_json)
    print(out_csv)


if __name__ == "__main__":
    main()
