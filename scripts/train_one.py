#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1m_pretrain.train import ExperimentConfig, run_experiment


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["transpolymer", "mmpolymer", "molformer", "smi_ted", "dual_deepchem_pselfies_shared", "dual_correctdeepchem_pselfies_shared"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--init-mode", choices=["scratch", "checkpoint", "huggingface"], default="checkpoint")
    parser.add_argument("--backbone-family", choices=["upstream_roberta", "experimental"], default="upstream_roberta")
    parser.add_argument("--scratch-variant", choices=["base", "deep", "small", "tiny"], default="base")
    parser.add_argument("--position-embedding-type", choices=["absolute", "rope"], default="absolute")
    parser.add_argument("--attention-variant", choices=["mha", "gqa"], default="mha")
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--preprocess-workers", type=int, default=8)
    parser.add_argument("--validation-protocol", default="external_polymer_mix_v1")
    parser.add_argument("--translation-decoder-type", choices=["autoregressive", "diffusion_like"], default="autoregressive")
    parser.add_argument("--translation-decoder-layers", type=int, default=2)
    parser.add_argument("--translation-decoder-dropout", type=float, default=0.1)
    parser.add_argument("--translation-num-diffusion-steps", type=int, default=16)
    parser.add_argument("--translation-diffusion-max-corrupt-prob", type=float, default=0.8)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--translation-mask-probability", type=float, default=0.15)
    parser.add_argument("--view-weight", type=float, default=0.0)
    parser.add_argument("--translation-weight", type=float, default=0.0)
    parser.add_argument("--view-temperature", type=float, default=0.07)
    parser.add_argument("--combined-view-weight", type=float, default=1.0)
    parser.add_argument("--combined-translation-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--wandb-project", default="p1m-pretrain-experiments")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--smallmol-csv", default=None)
    parser.add_argument("--smallmol-fraction", type=float, default=0.0)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Train for N epochs (overrides --steps)")
    parser.add_argument("--multi-gpu", action="store_true", help="Use DataParallel across all GPUs")
    parser.add_argument("--num-workers-override", type=int, default=None, help="DataLoader num_workers")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from checkpoint path")
    parser.add_argument("--mlm-selfies-mix", action="store_true", help="Randomly use pSELFIES for 50%% of MLM inputs")
    parser.add_argument("--translation-target-mode", choices=["paired", "bigsmiles"], default="paired", help="Translation targets: paired pSMILES<->pSELFIES or BigSMILES-only")
    parser.add_argument("--force-generic-translation-decoder", action="store_true", help="Use the generic translation decoder even if the backbone provides a custom one")
    parser.add_argument("--transfer-from", type=str, default=None, help="Transfer encoder weights from this checkpoint (cross-tokenizer)")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0, help="Freeze transferred encoder for N epochs, then unfreeze")
    args = parser.parse_args()
    return ExperimentConfig(
        backbone=args.backbone,
        run_name=args.run_name,
        init_mode=args.init_mode,
        backbone_family=args.backbone_family,
        scratch_variant=args.scratch_variant,
        position_embedding_type=args.position_embedding_type,
        attention_variant=args.attention_variant,
        num_key_value_heads=args.num_key_value_heads,
        train_size=args.train_size,
        val_size=args.val_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        steps=args.steps,
        eval_every=args.eval_every,
        preprocess_workers=args.preprocess_workers,
        validation_protocol=args.validation_protocol,
        translation_decoder_type=args.translation_decoder_type,
        translation_decoder_layers=args.translation_decoder_layers,
        translation_decoder_dropout=args.translation_decoder_dropout,
        translation_num_diffusion_steps=args.translation_num_diffusion_steps,
        translation_diffusion_max_corrupt_prob=args.translation_diffusion_max_corrupt_prob,
        mlm_probability=args.mlm_probability,
        translation_mask_probability=args.translation_mask_probability,
        view_weight=args.view_weight,
        translation_weight=args.translation_weight,
        view_temperature=args.view_temperature,
        combined_view_weight=args.combined_view_weight,
        combined_translation_weight=args.combined_translation_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        smallmol_csv=args.smallmol_csv,
        smallmol_fraction=args.smallmol_fraction,
        cuda_device=args.cuda_device,
        epochs=args.epochs,
        multi_gpu=args.multi_gpu,
        num_workers_override=args.num_workers_override,
        resume_from=args.resume_from,
        mlm_selfies_mix=args.mlm_selfies_mix,
        translation_target_mode=args.translation_target_mode,
        force_generic_translation_decoder=args.force_generic_translation_decoder,
        transfer_from=args.transfer_from,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
    )


if __name__ == "__main__":
    config = parse_args()
    summary = run_experiment(config)
    print(json.dumps({"best_mlm_metrics": summary["best_mlm_metrics"], "best_combined_metrics": summary["best_combined_metrics"]}, indent=2))
