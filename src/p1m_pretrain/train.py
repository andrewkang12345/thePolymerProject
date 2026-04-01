from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
import itertools
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from .data import (
    ContinuationCollator,
    P1MDataset,
    build_representation_vocab,
    load_records,
    load_smallmol_records,
    prepare_clean_split,
    prepare_external_val_cache,
    prepare_pi1m_train_cache,
)
from .bigsmiles import augment_parquet_with_bigsmiles
from .dual_tokenizer import DualTokenizerBundle, DualTokenizerContinuationCollator, PSELFIES_TOKENS_PATH
from .modeling import ContinuationModel
from .paths import get_paths
from .upstream import load_polymer_smiles_tokenizer, load_tokenizer_for_backbone


PATHS = get_paths()


@dataclass
class ExperimentConfig:
    backbone: str
    run_name: str
    init_mode: str = "checkpoint"
    backbone_family: str = "upstream_roberta"
    scratch_variant: str = "base"
    position_embedding_type: str = "absolute"
    attention_variant: str = "mha"
    num_key_value_heads: int = 4
    train_size: int = 20000
    val_size: int = 2000
    val_fraction: float = 0.1
    seed: int = 13
    batch_size: int = 12
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    steps: int = 120
    eval_every: int = 40
    num_workers: int = 0
    psmiles_max_len: int = 192
    translation_source_max_len: int = 256
    translation_target_max_len: int = 160
    translation_decoder_type: str = "autoregressive"
    translation_decoder_layers: int = 2
    translation_decoder_dropout: float = 0.1
    translation_num_diffusion_steps: int = 16
    translation_diffusion_max_corrupt_prob: float = 0.8
    mlm_probability: float = 0.15
    translation_mask_probability: float = 0.15
    view_weight: float = 0.0
    translation_weight: float = 0.0
    view_temperature: float = 0.07
    combined_view_weight: float = 1.0
    combined_translation_weight: float = 1.0
    gradient_clip_norm: float = 1.0
    train_log_every: int = 10
    output_root: str = str(PATHS.outputs_dir)
    cache_root: str = str(PATHS.cache_dir)
    split_protocol: str = "canonical_proxy_hash_v1"
    preprocess_workers: int = 8
    validation_protocol: str = "external_polymer_mix_v1"
    wandb_project: str = "p1m-pretrain-experiments"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_mode: str | None = None
    smallmol_csv: str | None = None
    smallmol_fraction: float = 0.0
    cuda_device: int | None = None
    epochs: int | None = None
    multi_gpu: bool = False
    num_workers_override: int | None = None
    resume_from: str | None = None
    mlm_selfies_mix: bool = False
    translation_target_mode: str = "paired"
    force_generic_translation_decoder: bool = False
    transfer_from: str | None = None
    freeze_encoder_epochs: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _unpack_metrics(metrics_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unpack stacked metrics tensor from ContinuationModel.forward().
    DataParallel may mean-reduce across GPUs, so we get averaged values."""
    if metrics_tensor.dim() == 1:
        t = metrics_tensor
    else:
        t = metrics_tensor.mean(dim=0)
    return {
        "total_loss": t[0],
        "mlm_loss": t[1],
        "view_loss": t[2],
        "translation_loss": t[3],
        "view_top1": t[4],
        "translation_token_accuracy": t[5],
    }


def _augment_eval_metrics(metrics: dict[str, float], config: ExperimentConfig) -> dict[str, float]:
    metrics["val_combined_loss"] = (
        metrics["val_mlm_loss"]
        + config.combined_view_weight * metrics["val_view_loss"]
        + config.combined_translation_weight * metrics["val_translation_loss"]
    )
    metrics["val_objective_loss"] = (
        metrics["val_mlm_loss"]
        + config.view_weight * metrics["val_view_loss"]
        + config.translation_weight * metrics["val_translation_loss"]
    )
    return metrics


@torch.no_grad()
def evaluate(model: ContinuationModel, loader: DataLoader, device: torch.device, config: ExperimentConfig) -> dict[str, float]:
    model.eval()
    metrics = {
        "mlm_loss": 0.0,
        "view_loss": 0.0,
        "translation_loss": 0.0,
        "view_top1": 0.0,
        "translation_token_accuracy": 0.0,
        "count": 0,
    }
    for batch in loader:
        batch = _move_to_device(batch, device)
        out = model(batch, view_weight=config.view_weight, translation_weight=config.translation_weight)
        m = _unpack_metrics(out)
        metrics["mlm_loss"] += float(m["mlm_loss"])
        metrics["view_loss"] += float(m["view_loss"])
        metrics["translation_loss"] += float(m["translation_loss"])
        metrics["view_top1"] += float(m["view_top1"])
        metrics["translation_token_accuracy"] += float(m["translation_token_accuracy"])
        metrics["count"] += 1
    count = max(metrics.pop("count"), 1)
    reduced = {f"val_{key}": value / count for key, value in metrics.items()}
    return _augment_eval_metrics(reduced, config)


def _save_checkpoint(
    path: Path,
    *,
    config: ExperimentConfig,
    model: ContinuationModel,
    optimizer: AdamW,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def _wandb_mode(config: ExperimentConfig) -> str:
    if config.wandb_mode:
        return config.wandb_mode
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    return "offline"


def _transfer_weights(model: ContinuationModel, transfer_from: str, device: torch.device) -> dict:
    """Transfer compatible weights from a pretrained checkpoint into a ContinuationModel.

    Copies encoder layers, position embeddings, and LayerNorm weights.
    Skips word_embeddings and lm_head (vocab size mismatch for cross-tokenizer transfer).
    """
    src_state = torch.load(transfer_from, map_location=device, weights_only=False)
    if isinstance(src_state, dict) and "model_state_dict" in src_state:
        src_state = src_state["model_state_dict"]

    dst_state = model.state_dict()
    transferred, skipped = [], []

    for src_key, src_val in src_state.items():
        if "word_embeddings" in src_key or "lm_head" in src_key or "position_ids" in src_key:
            skipped.append(src_key)
            continue
        dst_key = f"backbone.{src_key}"
        if dst_key in dst_state and src_val.shape == dst_state[dst_key].shape:
            dst_state[dst_key] = src_val
            transferred.append(src_key)
        else:
            skipped.append(src_key)

    model.load_state_dict(dst_state)
    print(f"Transferred {len(transferred)} weights, skipped {len(skipped)}")
    return {"transferred": len(transferred), "skipped": len(skipped)}


def _freeze_encoder(model: ContinuationModel) -> None:
    """Freeze all transferred params; keep word_embeddings, lm_head, view_projector, translation_decoder trainable."""
    for name, param in model.named_parameters():
        if any(k in name for k in ["word_embeddings", "lm_head", "view_projector", "translation_decoder"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen encoder: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")


def _unfreeze_all(model: ContinuationModel) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfrozen all: {total:,} params trainable")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    seed_everything(config.seed)

    cache_root = Path(config.cache_root)
    output_root = Path(config.output_root)
    output_dir = output_root / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=config.wandb_group,
        name=config.run_name,
        mode=_wandb_mode(config),
        dir=str(output_dir),
        config=asdict(config),
        reinit="finish_previous",
    )

    if config.validation_protocol == "external_polymer_mix_v1":
        train_cache_base_path = cache_root / "pi1m_train_dedup.parquet"
        val_cache_base_path = cache_root / f"external_val_seed{config.seed}_n{config.val_size}.parquet"
        vocab_path = cache_root / (
            "rep_vocab_pi1m_train_bigsmiles.json"
            if config.translation_target_mode == "bigsmiles"
            else "rep_vocab_pi1m_train.json"
        )
        prepare_pi1m_train_cache(train_cache_base_path, preprocess_workers=config.preprocess_workers)
        prepare_external_val_cache(
            val_cache_base_path,
            pi1m_train_cache_path=train_cache_base_path,
            seed=config.seed,
            val_size=config.val_size,
            preprocess_workers=config.preprocess_workers,
        )
        train_cache_path = train_cache_base_path
        val_cache_path = val_cache_base_path
        if config.translation_target_mode == "bigsmiles":
            train_cache_path = cache_root / "pi1m_train_dedup_bigsmiles.parquet"
            val_cache_path = cache_root / f"external_val_seed{config.seed}_n{config.val_size}_bigsmiles.parquet"
            augment_parquet_with_bigsmiles(
                train_cache_base_path,
                train_cache_path,
                preprocess_workers=config.preprocess_workers,
                drop_missing=True,
            )
            augment_parquet_with_bigsmiles(
                val_cache_base_path,
                val_cache_path,
                preprocess_workers=config.preprocess_workers,
                drop_missing=True,
            )
        rep_vocab = build_representation_vocab(
            train_cache_path,
            vocab_path,
            include_bigsmiles=config.translation_target_mode == "bigsmiles",
        )
        train_limit = config.train_size if config.train_size > 0 else None
        val_limit = config.val_size if config.val_size > 0 else None
        train_records = load_records(train_cache_path, None, limit=train_limit)
        val_records = load_records(val_cache_path, None, limit=val_limit)
        cache_path = train_cache_path
    else:
        cache_base_path = cache_root / f"pi1m_clean_seed{config.seed}_val{str(config.val_fraction).replace('.', 'p')}.parquet"
        vocab_path = cache_root / (
            f"rep_vocab_clean_seed{config.seed}_val{str(config.val_fraction).replace('.', 'p')}_bigsmiles.json"
            if config.translation_target_mode == "bigsmiles"
            else f"rep_vocab_clean_seed{config.seed}_val{str(config.val_fraction).replace('.', 'p')}.json"
        )
        prepare_clean_split(
            cache_base_path,
            seed=config.seed,
            val_fraction=config.val_fraction,
            preprocess_workers=config.preprocess_workers,
        )
        cache_path = cache_base_path
        if config.translation_target_mode == "bigsmiles":
            cache_path = cache_root / f"pi1m_clean_seed{config.seed}_val{str(config.val_fraction).replace('.', 'p')}_bigsmiles.parquet"
            augment_parquet_with_bigsmiles(
                cache_base_path,
                cache_path,
                preprocess_workers=config.preprocess_workers,
                drop_missing=True,
            )
        rep_vocab = build_representation_vocab(
            cache_path,
            vocab_path,
            include_bigsmiles=config.translation_target_mode == "bigsmiles",
        )
        train_limit = config.train_size if config.train_size > 0 else None
        val_limit = config.val_size if config.val_size > 0 else None
        train_records = load_records(cache_path, "train", limit=train_limit)
        val_records = load_records(cache_path, "val", limit=val_limit)

    if config.smallmol_csv and config.smallmol_fraction > 0:
        sm_records = load_smallmol_records(config.smallmol_csv)
        n_sm = int(len(train_records) * config.smallmol_fraction)
        train_records = train_records + random.sample(sm_records, min(n_sm, len(sm_records)))
        random.shuffle(train_records)
        print(f"Added {min(n_sm, len(sm_records))} small molecule records (total: {len(train_records)})")

    model_translation_kwargs = {
        "translation_vocab_size": rep_vocab.size,
        "translation_pad_id": rep_vocab.pad_id,
        "translation_bos_id": rep_vocab.bos_id,
        "translation_eos_id": rep_vocab.eos_id,
        "translation_mask_id": rep_vocab.mask_id,
    }

    if config.backbone in {"dual_deepchem_pselfies_shared", "dual_correctdeepchem_pselfies_shared"}:
        dual_bundle = DualTokenizerBundle.load(
            max_len=max(config.psmiles_max_len, config.translation_source_max_len),
            use_original_deepchem=config.backbone == "dual_correctdeepchem_pselfies_shared",
        )
        tokenizer = dual_bundle.psmiles_tokenizer
        collator = DualTokenizerContinuationCollator(
            dual_bundle,
            psmiles_max_len=config.psmiles_max_len,
            translation_source_max_len=config.translation_source_max_len,
            translation_target_max_len=config.translation_target_max_len,
            mlm_probability=config.mlm_probability,
            translation_mask_probability=config.translation_mask_probability,
            mlm_selfies_mix=config.mlm_selfies_mix,
            translation_target_mode=config.translation_target_mode,
            translation_vocab=rep_vocab if config.translation_target_mode == "bigsmiles" else None,
        )
        vocab_path = PSELFIES_TOKENS_PATH if config.translation_target_mode != "bigsmiles" else vocab_path
        if config.translation_target_mode == "bigsmiles":
            model_translation_kwargs = {
                "translation_vocab_size": rep_vocab.size,
                "translation_pad_id": rep_vocab.pad_id,
                "translation_bos_id": rep_vocab.bos_id,
                "translation_eos_id": rep_vocab.eos_id,
                "translation_mask_id": rep_vocab.mask_id,
            }
        else:
            model_translation_kwargs = {
                "translation_vocab_size": 1,
                "translation_pad_id": 0,
                "translation_bos_id": 0,
                "translation_eos_id": 0,
                "translation_mask_id": 0,
            }
    else:
        tokenizer = load_tokenizer_for_backbone(
            config.backbone,
            max_len=max(config.psmiles_max_len, config.translation_source_max_len),
        )

        # Set representation prefixes for extended tokenizer models
        smiles_prefix = ""
        selfies_prefix = ""
        if config.backbone == "smi_ted":
            from .smi_ted_extended import SMILES_PREFIX as SP, SELFIES_PREFIX as SFP
            smiles_prefix = SP
            selfies_prefix = SFP
            from .smi_ted_extended import SmiTedRepVocab
            rep_vocab = SmiTedRepVocab(tokenizer)
            model_translation_kwargs = {
                "translation_vocab_size": rep_vocab.size,
                "translation_pad_id": rep_vocab.pad_id,
                "translation_bos_id": rep_vocab.bos_id,
                "translation_eos_id": rep_vocab.eos_id,
                "translation_mask_id": rep_vocab.mask_id,
            }
        collator = ContinuationCollator(
            tokenizer,
            rep_vocab,
            psmiles_max_len=config.psmiles_max_len,
            translation_source_max_len=config.translation_source_max_len,
            translation_target_max_len=config.translation_target_max_len,
            mlm_probability=config.mlm_probability,
            translation_mask_probability=config.translation_mask_probability,
            smiles_prefix=smiles_prefix,
            selfies_prefix=selfies_prefix,
            mlm_selfies_mix=config.mlm_selfies_mix,
            translation_target_mode=config.translation_target_mode,
        )

    dl_workers = config.num_workers_override if config.num_workers_override is not None else config.num_workers
    train_loader = DataLoader(
        P1MDataset(train_records),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=dl_workers,
        collate_fn=collator,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        P1MDataset(val_records),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=dl_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    if config.cuda_device is not None:
        device = torch.device(f"cuda:{config.cuda_device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuationModel(
        backbone_name=config.backbone,
        init_mode=config.init_mode,
        translation_vocab_size=model_translation_kwargs["translation_vocab_size"],
        translation_pad_id=model_translation_kwargs["translation_pad_id"],
        translation_bos_id=model_translation_kwargs["translation_bos_id"],
        translation_eos_id=model_translation_kwargs["translation_eos_id"],
        translation_max_length=config.translation_target_max_len,
        translation_mask_id=model_translation_kwargs["translation_mask_id"],
        translation_decoder_type=config.translation_decoder_type,
        translation_decoder_layers=config.translation_decoder_layers,
        translation_decoder_dropout=config.translation_decoder_dropout,
        translation_num_diffusion_steps=config.translation_num_diffusion_steps,
        translation_diffusion_max_corrupt_prob=config.translation_diffusion_max_corrupt_prob,
        view_temperature=config.view_temperature,
        force_generic_translation_decoder=(
            config.force_generic_translation_decoder
            or (
                config.backbone in {"dual_deepchem_pselfies_shared", "dual_correctdeepchem_pselfies_shared"}
                and config.translation_target_mode == "bigsmiles"
            )
        ),
        backbone_kwargs={
            "backbone_family": config.backbone_family,
            "scratch_variant": config.scratch_variant,
            "position_embedding_type": config.position_embedding_type,
            "attention_variant": config.attention_variant,
            "num_key_value_heads": config.num_key_value_heads,
        },
    ).to(device)

    # Pre-initialize lazy modules before DataParallel (e.g. SMI-TED feature maps)
    if hasattr(model.backbone, "warmup_feature_maps"):
        model.backbone.warmup_feature_maps(device)

    # Transfer weights from a different backbone checkpoint (cross-tokenizer)
    if config.transfer_from:
        _transfer_weights(model, config.transfer_from, device)
        if config.freeze_encoder_epochs > 0:
            _freeze_encoder(model)

    # Multi-GPU via DataParallel
    raw_model = model
    if config.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    start_step = 0
    if config.resume_from:
        print(f"Resuming from checkpoint: {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("metrics", {}).get("step", 0)
        print(f"  Loaded model + optimizer state, resuming from step {start_step}")

    # Disable autocast for models with incompatible attention (e.g. fast_transformers linear attention)
    use_autocast = device.type == "cuda" and torch.cuda.is_bf16_supported() and config.backbone != "smi_ted"

    # Compute total steps from epochs if specified
    if config.epochs is not None and config.epochs > 0:
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.epochs
        eval_every = max(steps_per_epoch, 1)  # eval once per epoch
        print(f"Training for {config.epochs} epochs: {steps_per_epoch} steps/epoch, {total_steps} total steps")
    else:
        total_steps = config.steps
        eval_every = config.eval_every

    # Compute unfreeze step for transfer learning
    unfreeze_step = 0
    if config.transfer_from and config.freeze_encoder_epochs > 0 and config.epochs is not None:
        steps_per_ep = len(train_loader)
        unfreeze_step = start_step + config.freeze_encoder_epochs * steps_per_ep
        print(f"Will unfreeze encoder at step {unfreeze_step} (after {config.freeze_encoder_epochs} epochs)")

    history: list[dict[str, float]] = []
    train_iter = itertools.cycle(train_loader)
    effective_total = total_steps + start_step
    progress = tqdm(range(start_step + 1, effective_total + 1), desc=config.run_name, leave=False)
    best_mlm_metrics: dict[str, float] | None = None
    best_combined_metrics: dict[str, float] | None = None

    for step in progress:
        # Unfreeze encoder at the right step
        if unfreeze_step > 0 and step == unfreeze_step + 1:
            _unfreeze_all(raw_model)
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.learning_rate, weight_decay=config.weight_decay,
            )
            run.log({"train/unfrozen": 1, "step": step}, step=step)
            print(f"Step {step}: unfroze encoder, new optimizer created")
        model.train()
        batch = _move_to_device(next(train_iter), device)
        optimizer.zero_grad(set_to_none=True)

        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
        with autocast_context:
            if isinstance(model, torch.nn.DataParallel):
                bs = batch["mlm_input_ids"].size(0)
                batch["_view_weight"] = torch.full((bs,), config.view_weight, device=device)
                batch["_translation_weight"] = torch.full((bs,), config.translation_weight, device=device)
                out = model(batch)
            else:
                out = model(batch, view_weight=config.view_weight, translation_weight=config.translation_weight)
            m = _unpack_metrics(out)
            loss = m["total_loss"]

        loss.backward()
        if config.gradient_clip_norm > 0:
            params = model.module.parameters() if isinstance(model, torch.nn.DataParallel) else model.parameters()
            torch.nn.utils.clip_grad_norm_(params, max_norm=config.gradient_clip_norm)
        optimizer.step()

        progress.set_postfix(
            mlm=f"{float(m['mlm_loss']):.3f}",
            view=f"{float(m['view_loss']):.3f}",
            trans=f"{float(m['translation_loss']):.3f}",
        )

        if step % config.train_log_every == 0 or step == 1:
            run.log(
                {
                    "train/total_loss": float(loss.detach()),
                    "train/mlm_loss": float(m["mlm_loss"]),
                    "train/view_loss": float(m["view_loss"]),
                    "train/translation_loss": float(m["translation_loss"]),
                    "train/view_top1": float(m["view_top1"]),
                    "train/translation_token_accuracy": float(m["translation_token_accuracy"]),
                    "step": step,
                },
                step=step,
            )

        if step % eval_every == 0 or step == effective_total:
            save_model = raw_model  # always save unwrapped model
            metrics = evaluate(save_model, val_loader, device, config)
            metrics["step"] = step
            history.append(metrics)
            run.log({f"eval/{key}": value for key, value in metrics.items()}, step=step)
            if best_mlm_metrics is None or metrics["val_mlm_loss"] < best_mlm_metrics["val_mlm_loss"]:
                best_mlm_metrics = metrics
                _save_checkpoint(
                    output_dir / "best_mlm.pt",
                    config=config,
                    model=save_model,
                    optimizer=optimizer,
                    metrics=metrics,
                )
                _save_checkpoint(
                    output_dir / "best.pt",
                    config=config,
                    model=save_model,
                    optimizer=optimizer,
                    metrics=metrics,
                )
            if best_combined_metrics is None or metrics["val_combined_loss"] < best_combined_metrics["val_combined_loss"]:
                best_combined_metrics = metrics
                _save_checkpoint(
                    output_dir / "best_combined.pt",
                    config=config,
                    model=save_model,
                    optimizer=optimizer,
                    metrics=metrics,
                )

    final_metrics = history[-1] if history else evaluate(model, val_loader, device, config)
    summary = {
        "config": asdict(config),
        "best_mlm_metrics": best_mlm_metrics or final_metrics,
        "best_combined_metrics": best_combined_metrics or final_metrics,
        "final_metrics": final_metrics,
        "history": history,
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "cache_path": str(cache_path),
        "vocab_path": str(vocab_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    run.summary.update(
        {
            "best_mlm_val_mlm_loss": summary["best_mlm_metrics"]["val_mlm_loss"],
            "best_combined_val_combined_loss": summary["best_combined_metrics"]["val_combined_loss"],
            "final_val_mlm_loss": summary["final_metrics"]["val_mlm_loss"],
            "final_val_combined_loss": summary["final_metrics"]["val_combined_loss"],
        }
    )
    run.finish()
    return summary
