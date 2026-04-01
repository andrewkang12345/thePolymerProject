#!/usr/bin/env python3
"""Publish a model-level downstream summary run to wandb from per-task JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1m_pretrain.paths import get_paths

RESULTS_DIR = get_paths().downstream_results_dir
ALL_TASKS = ["Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "Eat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model/run name prefix used in downstream_results/<model>_<task>.json")
    parser.add_argument("--wandb-project", default="p1m-downstream-eval")
    parser.add_argument("--allow-partial", action="store_true", help="Publish even if some task JSONs are missing")
    return parser.parse_args()


def load_results(model_name: str) -> list[dict]:
    results: list[dict] = []
    for task in ALL_TASKS:
        path = RESULTS_DIR / f"{model_name}_{task}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, list) and payload:
            results.append(payload[0])
    return results


def main() -> None:
    args = parse_args()
    results = load_results(args.model)
    found_tasks = [r["task"] for r in results]
    missing_tasks = [task for task in ALL_TASKS if task not in found_tasks]

    if missing_tasks and not args.allow_partial:
        raise SystemExit(f"Missing task results for {args.model}: {missing_tasks}")
    if not results:
        raise SystemExit(f"No downstream result JSONs found for {args.model}")

    first = results[0]
    model_config = first.get("model_config", {})
    summary_table: dict[str, float | int | str] = {}
    for r in results:
        summary_table[f"{r['task']}_r2"] = r["test_r2_mean"]
        summary_table[f"{r['task']}_rmse"] = r["test_rmse_mean"]
    summary_table["mean_r2"] = float(np.mean([r["test_r2_mean"] for r in results]))
    summary_table["num_tasks_found"] = len(results)
    summary_table["num_tasks_expected"] = len(ALL_TASKS)
    summary_table["is_complete"] = int(len(results) == len(ALL_TASKS))
    summary_table["missing_tasks"] = ",".join(missing_tasks)

    run_id = "summary-" + hashlib.md5(args.model.encode("utf-8")).hexdigest()[:12]
    run = wandb.init(
        project=args.wandb_project,
        id=run_id,
        resume="allow",
        name=args.model,
        job_type="summary",
        config={
            "model": args.model,
            "backbone": model_config.get("backbone"),
            "view_weight": model_config.get("view_weight"),
            "translation_weight": model_config.get("translation_weight"),
            "tasks_found": found_tasks,
            "tasks_missing": missing_tasks,
            "source": "downstream_results_json",
        },
    )
    run.log(summary_table)
    for key, value in summary_table.items():
        run.summary[key] = value
    run.finish()

    print(f"Published summary for {args.model}")
    print(json.dumps(summary_table, indent=2))


if __name__ == "__main__":
    main()
