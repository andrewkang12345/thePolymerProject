#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1m_pretrain.data import P1MRecord  # noqa: E402
from p1m_pretrain.dual_tokenizer import DualTokenizerBundle, DualTokenizerContinuationCollator  # noqa: E402
from p1m_pretrain.modeling import ContinuationModel  # noqa: E402
from p1m_pretrain.paths import get_paths  # noqa: E402
from p1m_pretrain.pselfies import proxy_pselfies_from_psmiles  # noqa: E402
from p1m_pretrain.train import ExperimentConfig, _save_checkpoint, _unpack_metrics, evaluate, seed_everything  # noqa: E402


PATHS = get_paths()


class PolyOne100MIterableDataset(IterableDataset[P1MRecord]):
    def __init__(
        self,
        parquet_dir: Path,
        *,
        seed: int,
        shuffle_files: bool = True,
        shuffle_rows: bool = True,
        source_name: str = "polyone_100m",
    ) -> None:
        self.parquet_dir = parquet_dir
        self.seed = seed
        self.shuffle_files = shuffle_files
        self.shuffle_rows = shuffle_rows
        self.source_name = source_name
        self.files = sorted(parquet_dir.glob("polyOne_*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No polyOne_*.parquet shards found in {parquet_dir}")

    def __iter__(self) -> Iterator[P1MRecord]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + worker_id)
        np_rng = np.random.default_rng(self.seed + worker_id)
        files = self.files[worker_id::num_workers]
        while True:
            pass_files = list(files)
            if self.shuffle_files:
                rng.shuffle(pass_files)
            for parquet_path in pass_files:
                table = pq.read_table(parquet_path, columns=["smiles"])
                smiles_values = table.column("smiles").to_pylist()
                indices = np.arange(len(smiles_values))
                if self.shuffle_rows:
                    np_rng.shuffle(indices)
                for idx in indices:
                    psmiles = smiles_values[int(idx)]
                    if not psmiles:
                        continue
                    pselfies = proxy_pselfies_from_psmiles(str(psmiles))
                    if not pselfies:
                        continue
                    yield P1MRecord(psmiles=str(psmiles), pselfies=pselfies, source_name=self.source_name)


def _load_polyone_val_records(dev_path: Path, *, val_size: int, seed: int) -> list[P1MRecord]:
    del seed
    if not dev_path.exists():
        raise FileNotFoundError(f"PolyOne dev SMILES file not found: {dev_path}")
    records: list[P1MRecord] = []
    with dev_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            psmiles = line.strip()
            if not psmiles:
                continue
            pselfies = proxy_pselfies_from_psmiles(psmiles)
            if not pselfies:
                continue
            record = P1MRecord(psmiles=psmiles, pselfies=pselfies, source_name="polyone_100m_dev")
            records.append(record)
            if len(records) >= val_size:
                break
    if not records:
        raise RuntimeError(f"No valid validation records could be built from {dev_path}")
    return records


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _count_polyone_rows(parquet_dir: Path) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in sorted(parquet_dir.glob("polyOne_*.parquet")))


def parse_args() -> argparse.Namespace:
    default_root = PATHS.poly_any2any_root / "data" / "raw" / "polyone_100m" / "original"
    parser = argparse.ArgumentParser()
    parser.add_argument("--polyone-root", type=Path, default=default_root)
    parser.add_argument("--run-name", default="polyone100m_dual_correctdeepchem_deep_selfiesmlm_vw1_tw1")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=None, help="Default is one pass over all parquet rows.")
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--psmiles-max-len", type=int, default=192)
    parser.add_argument("--translation-source-max-len", type=int, default=256)
    parser.add_argument("--translation-target-max-len", type=int, default=160)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--translation-mask-probability", type=float, default=0.15)
    parser.add_argument("--view-weight", type=float, default=1.0)
    parser.add_argument("--translation-weight", type=float, default=1.0)
    parser.add_argument("--view-temperature", type=float, default=0.07)
    parser.add_argument("--translation-decoder-layers", type=int, default=2)
    parser.add_argument("--translation-decoder-dropout", type=float, default=0.1)
    parser.add_argument("--scratch-variant", choices=["base", "deep", "small", "tiny"], default="deep")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--wandb-project", default="p1m-polyone100m")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--output-root", type=Path, default=PATHS.outputs_dir)
    parser.add_argument("--no-multi-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    parquet_dir = args.polyone_root
    total_rows = _count_polyone_rows(parquet_dir)
    total_steps = args.steps or max(total_rows // args.batch_size, 1)
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        backbone="dual_correctdeepchem_pselfies_shared",
        run_name=args.run_name,
        init_mode="scratch",
        scratch_variant=args.scratch_variant,
        train_size=0,
        val_size=args.val_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        steps=total_steps,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        psmiles_max_len=args.psmiles_max_len,
        translation_source_max_len=args.translation_source_max_len,
        translation_target_max_len=args.translation_target_max_len,
        mlm_probability=args.mlm_probability,
        translation_mask_probability=args.translation_mask_probability,
        view_weight=args.view_weight,
        translation_weight=args.translation_weight,
        view_temperature=args.view_temperature,
        translation_decoder_layers=args.translation_decoder_layers,
        translation_decoder_dropout=args.translation_decoder_dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        validation_protocol="polyone_100m_v1",
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        output_root=str(args.output_root),
        cache_root=str(PATHS.cache_dir),
        multi_gpu=not args.no_multi_gpu,
        num_workers_override=args.num_workers,
        resume_from=str(args.resume_from) if args.resume_from else None,
        mlm_selfies_mix=True,
        translation_target_mode="paired",
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.run_name,
        mode=args.wandb_mode,
        dir=str(output_dir),
        config={**asdict(config), "polyone_rows": total_rows, "polyone_root": str(parquet_dir)},
        reinit="finish_previous",
    )

    bundle = DualTokenizerBundle.load(max_len=max(args.psmiles_max_len, args.translation_source_max_len), use_original_deepchem=True)
    collator = DualTokenizerContinuationCollator(
        bundle,
        psmiles_max_len=args.psmiles_max_len,
        translation_source_max_len=args.translation_source_max_len,
        translation_target_max_len=args.translation_target_max_len,
        mlm_probability=args.mlm_probability,
        translation_mask_probability=args.translation_mask_probability,
        mlm_selfies_mix=True,
        translation_target_mode="paired",
    )
    train_dataset = PolyOne100MIterableDataset(parquet_dir, seed=args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
        pin_memory=True,
    )
    val_records = _load_polyone_val_records(parquet_dir / "generated_polymer_smiles_dev.txt", val_size=args.val_size, seed=args.seed)
    val_loader = DataLoader(val_records, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuationModel(
        backbone_name="dual_correctdeepchem_pselfies_shared",
        init_mode="scratch",
        translation_vocab_size=1,
        translation_pad_id=0,
        translation_bos_id=0,
        translation_eos_id=0,
        translation_max_length=args.translation_target_max_len,
        translation_mask_id=0,
        translation_decoder_layers=args.translation_decoder_layers,
        translation_decoder_dropout=args.translation_decoder_dropout,
        view_temperature=args.view_temperature,
        backbone_kwargs={"scratch_variant": args.scratch_variant},
    ).to(device)
    raw_model = model
    if not args.no_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("metrics", {}).get("step", 0))
        print(f"Resumed from {args.resume_from} at step {start_step}")

    use_autocast = device.type == "cuda" and torch.cuda.is_bf16_supported()
    train_iter = iter(train_loader)
    best_combined: dict[str, float] | None = None
    progress = tqdm(range(start_step + 1, total_steps + 1), desc=args.run_name)
    for step in progress:
        model.train()
        batch = _move_to_device(next(train_iter), device)
        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
        with autocast_context:
            if isinstance(model, torch.nn.DataParallel):
                batch["_view_weight"] = torch.full((batch["mlm_input_ids"].size(0),), args.view_weight, device=device)
                batch["_translation_weight"] = torch.full((batch["mlm_input_ids"].size(0),), args.translation_weight, device=device)
                out = model(batch)
            else:
                out = model(batch, view_weight=args.view_weight, translation_weight=args.translation_weight)
            metrics = _unpack_metrics(out)
            loss = metrics["total_loss"]
        loss.backward()
        if args.gradient_clip_norm > 0:
            params = model.module.parameters() if isinstance(model, torch.nn.DataParallel) else model.parameters()
            torch.nn.utils.clip_grad_norm_(params, args.gradient_clip_norm)
        optimizer.step()

        progress.set_postfix(
            mlm=f"{float(metrics['mlm_loss'].detach()):.3f}",
            view=f"{float(metrics['view_loss'].detach()):.3f}",
            trans=f"{float(metrics['translation_loss'].detach()):.3f}",
        )
        if step == 1 or step % 50 == 0:
            run.log(
                {
                    "train/total_loss": float(loss.detach()),
                    "train/mlm_loss": float(metrics["mlm_loss"].detach()),
                    "train/view_loss": float(metrics["view_loss"].detach()),
                    "train/translation_loss": float(metrics["translation_loss"].detach()),
                    "train/view_top1": float(metrics["view_top1"].detach()),
                    "train/translation_token_accuracy": float(
                        metrics["translation_token_accuracy"].detach()
                    ),
                    "step": step,
                },
                step=step,
            )

        should_eval = step % args.eval_every == 0 or step == total_steps
        should_save = step % args.save_every == 0 or step == total_steps
        if should_eval:
            val_metrics = evaluate(raw_model, val_loader, device, config)
            val_metrics["step"] = step
            val_log = {
                "val/mlm_loss": val_metrics["val_mlm_loss"],
                "val/view_loss": val_metrics["val_view_loss"],
                "val/translation_loss": val_metrics["val_translation_loss"],
                "val/combined_loss": val_metrics["val_combined_loss"],
                "val/objective_loss": val_metrics["val_objective_loss"],
                "val/view_top1": val_metrics["val_view_top1"],
                "val/translation_token_accuracy": val_metrics["val_translation_token_accuracy"],
                "step": step,
            }
            run.log(val_log | {f"eval/{key}": value for key, value in val_metrics.items()}, step=step)
            if best_combined is None or val_metrics["val_combined_loss"] < best_combined["val_combined_loss"]:
                best_combined = val_metrics
                _save_checkpoint(output_dir / "best.pt", config=config, model=raw_model, optimizer=optimizer, metrics=val_metrics)
                _save_checkpoint(output_dir / "best_combined.pt", config=config, model=raw_model, optimizer=optimizer, metrics=val_metrics)
        if should_save:
            latest_metrics = {"step": step}
            _save_checkpoint(output_dir / "latest.pt", config=config, model=raw_model, optimizer=optimizer, metrics=latest_metrics)

    summary = {"config": asdict(config), "best_combined_metrics": best_combined}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
