#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1m_pretrain.paths import get_paths
from p1m_pretrain.train import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="p1m_wide_search")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--preprocess-workers", type=int, default=8)
    parser.add_argument("--wandb-project", default="p1m-pretrain-experiments")
    parser.add_argument("--wandb-mode", default="offline")
    return parser.parse_args()


def objective_builder(args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        backbone = trial.suggest_categorical("backbone", ["transpolymer", "mmpolymer"])
        init_mode = trial.suggest_categorical("init_mode", ["checkpoint", "scratch"])
        if init_mode == "checkpoint":
            backbone_family = "upstream_roberta"
            scratch_variant = "base"
            position_embedding_type = "absolute"
            attention_variant = "mha"
            num_key_value_heads = 4
        else:
            backbone_family = trial.suggest_categorical("backbone_family", ["upstream_roberta", "experimental"])
            scratch_variant = trial.suggest_categorical("scratch_variant", ["base", "deep", "small", "tiny"])
            if backbone_family == "experimental":
                position_embedding_type = trial.suggest_categorical("position_embedding_type", ["absolute", "rope"])
                attention_variant = trial.suggest_categorical("attention_variant", ["mha", "gqa"])
                num_key_value_heads = (
                    trial.suggest_categorical("num_key_value_heads", [2, 4, 6])
                    if attention_variant == "gqa"
                    else 4
                )
            else:
                position_embedding_type = "absolute"
                attention_variant = "mha"
                num_key_value_heads = 4

        translation_decoder_type = trial.suggest_categorical("translation_decoder_type", ["autoregressive", "diffusion_like"])
        config = ExperimentConfig(
            backbone=backbone,
            run_name=f"{args.study_name}_trial_{trial.number:03d}",
            init_mode=init_mode,
            backbone_family=backbone_family,
            scratch_variant=scratch_variant,
            position_embedding_type=position_embedding_type,
            attention_variant=attention_variant,
            num_key_value_heads=num_key_value_heads,
            train_size=args.train_size,
            val_size=args.val_size,
            batch_size=trial.suggest_categorical("batch_size", [8, 12, 16]),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 2e-4, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True),
            steps=args.steps,
            eval_every=args.eval_every,
            preprocess_workers=args.preprocess_workers,
            validation_protocol="external_polymer_mix_v1",
            translation_decoder_type=translation_decoder_type,
            translation_decoder_layers=trial.suggest_categorical("translation_decoder_layers", [2, 4]),
            translation_decoder_dropout=trial.suggest_categorical("translation_decoder_dropout", [0.0, 0.1, 0.2]),
            translation_num_diffusion_steps=trial.suggest_categorical("translation_num_diffusion_steps", [8, 16, 32]),
            translation_diffusion_max_corrupt_prob=trial.suggest_categorical(
                "translation_diffusion_max_corrupt_prob", [0.5, 0.8, 0.95]
            ),
            mlm_probability=trial.suggest_categorical("mlm_probability", [0.1, 0.15, 0.2]),
            translation_mask_probability=trial.suggest_categorical("translation_mask_probability", [0.1, 0.15, 0.3]),
            view_weight=trial.suggest_categorical("view_weight", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]),
            translation_weight=trial.suggest_categorical("translation_weight", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]),
            view_temperature=trial.suggest_categorical("view_temperature", [0.05, 0.07, 0.1, 0.2]),
            gradient_clip_norm=trial.suggest_categorical("gradient_clip_norm", [0.5, 1.0, 2.0]),
            wandb_project=args.wandb_project,
            wandb_group=args.study_name,
            wandb_mode=args.wandb_mode,
        )
        try:
            summary = run_experiment(config)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                trial.set_user_attr("failed", "oom")
                return float("inf")
            raise
        best_mlm = summary["best_mlm_metrics"]
        best_combined = summary["best_combined_metrics"]
        final_metrics = summary["final_metrics"]
        trial.set_user_attr("run_name", config.run_name)
        trial.set_user_attr("best_mlm_metrics", best_mlm)
        trial.set_user_attr("best_combined_metrics", best_combined)
        trial.set_user_attr("final_metrics", final_metrics)
        trial.set_user_attr("config", summary["config"])
        return best_combined["val_combined_loss"]

    return objective


def main() -> None:
    args = parse_args()
    paths = get_paths()
    storage = args.storage or f"sqlite:///{paths.outputs_dir / (args.study_name + '.db')}"
    study = optuna.create_study(study_name=args.study_name, storage=storage, direction="minimize", load_if_exists=True)
    study.optimize(objective_builder(args), n_trials=args.n_trials)
    best_by_combined = min(
        (trial for trial in study.trials if trial.value is not None),
        key=lambda trial: trial.user_attrs["best_combined_metrics"]["val_combined_loss"],
    )
    best_by_mlm = min(
        (trial for trial in study.trials if trial.value is not None),
        key=lambda trial: trial.user_attrs["best_mlm_metrics"]["val_mlm_loss"],
    )
    summary = {
        "study_name": args.study_name,
        "storage": storage,
        "n_trials": len([trial for trial in study.trials if trial.value is not None]),
        "best_by_combined": {
            "trial_number": best_by_combined.number,
            "run_name": best_by_combined.user_attrs["run_name"],
            "metrics": best_by_combined.user_attrs["best_combined_metrics"],
            "config": best_by_combined.user_attrs["config"],
        },
        "best_by_mlm": {
            "trial_number": best_by_mlm.number,
            "run_name": best_by_mlm.user_attrs["run_name"],
            "metrics": best_by_mlm.user_attrs["best_mlm_metrics"],
            "config": best_by_mlm.user_attrs["config"],
        },
    }
    summary_path = paths.outputs_dir / f"{args.study_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
