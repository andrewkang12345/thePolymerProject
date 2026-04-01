#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1m_pretrain.train import ExperimentConfig, run_experiment


def main() -> None:
    grid = [
        ("original", 0.0, 0.0),
        ("view_only", 0.1, 0.0),
        ("translation_only", 0.0, 0.5),
        ("both", 0.1, 0.5),
    ]
    summaries = []
    for backbone in ["transpolymer", "mmpolymer"]:
        for name, view_weight, translation_weight in grid:
            run_name = f"externalval_checkpoint_{backbone}_{name}"
            config = ExperimentConfig(
                backbone=backbone,
                run_name=run_name,
                init_mode="checkpoint",
                validation_protocol="external_polymer_mix_v1",
                train_size=4096,
                val_size=512,
                batch_size=8,
                steps=30,
                eval_every=15,
                preprocess_workers=8,
                view_weight=view_weight,
                translation_weight=translation_weight,
            )
            summaries.append(run_experiment(config))
    print(
        json.dumps(
            {
                summary["config"]["run_name"]: {
                    "best_mlm_metrics": summary["best_mlm_metrics"],
                    "best_combined_metrics": summary["best_combined_metrics"],
                }
                for summary in summaries
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
