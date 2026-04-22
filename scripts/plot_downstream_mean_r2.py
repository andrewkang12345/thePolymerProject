#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "downstream_results"
OUTPUT_PATH = REPO_ROOT / "docs" / "downstream_mean_r2_test.png"
REQUIRED_TASKS = ("EPS", "Eat", "Eea", "Egb", "Egc", "Ei", "Nc", "Xc")
EXTRA_MODELS_FROM_ALL_RESULTS = {"baseline_mmpolymer", "baseline_transpolymer"}


def make_label(model: str, model_config: dict[str, object]) -> str:
    run_name = str(model_config.get("run_name") or model)
    downstream_input_mode = model_config.get("downstream_input_mode")
    if model != run_name:
        if downstream_input_mode == "dual_input":
            return f"{run_name} [dual_input]"
        if downstream_input_mode == "concat_encoder":
            return f"{run_name} [concat_encoder]"
        if model.endswith("_dualinput"):
            return f"{run_name} [dual_input]"
        if model.endswith("_concatenc"):
            return f"{run_name} [concat_encoder]"
    return run_name


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            if {"model", "task", "test_r2_mean"} <= row.keys():
                if path.name == "all_results.json" and str(row["model"]) not in EXTRA_MODELS_FROM_ALL_RESULTS:
                    continue
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[tuple[str, dict[str, object], float, float]]:
    by_model: dict[str, dict[str, object]] = {}
    for row in rows:
        model = str(row["model"])
        task = str(row["task"])
        score = float(row["test_r2_mean"])
        entry = by_model.setdefault(model, {"task_scores": {}, "model_config": row.get("model_config", {})})
        task_scores = entry["task_scores"]
        assert isinstance(task_scores, dict)
        task_scores[task] = score

    comparable: list[tuple[str, dict[str, object], float, float]] = []
    for model, entry in by_model.items():
        task_scores = entry["task_scores"]
        assert isinstance(task_scores, dict)
        if tuple(sorted(task_scores)) != tuple(sorted(REQUIRED_TASKS)):
            continue
        values = np.array([task_scores[task] for task in REQUIRED_TASKS], dtype=float)
        model_config = entry["model_config"]
        assert isinstance(model_config, dict)
        comparable.append((model, model_config, float(values.mean()), float(values.std(ddof=0))))
    comparable.sort(key=lambda item: item[2], reverse=True)
    return comparable


def make_plot(summary: list[tuple[str, dict[str, object], float, float]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    models = [make_label(model, model_config) for model, model_config, _, _ in summary]
    means = np.array([mean for _, _, mean, _ in summary], dtype=float)
    stds = np.array([std for _, _, _, std in summary], dtype=float)
    y = np.arange(len(models))

    fig_height = max(6.2, 0.42 * len(models) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.patch.set_facecolor("#f7f4ec")
    ax.set_facecolor("#fffdf8")

    colors = ["#2f6c5a"] + ["#b8c9c0"] * (len(models) - 1)
    edges = ["#20493d"] + ["#8aa497"] * (len(models) - 1)

    ax.barh(
        y,
        means,
        xerr=stds,
        color=colors,
        edgecolor=edges,
        linewidth=1.0,
        error_kw={"ecolor": "#6b7c73", "elinewidth": 1.2, "capsize": 3},
    )

    ax.set_yticks(y, labels=models, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.76, max(0.83, float(means.max()) + 0.006))
    ax.set_xlabel("Mean test R² across 8 downstream tasks", fontsize=11)
    ax.set_title("Downstream Evaluation Summary", fontsize=16, weight="bold", loc="left", pad=6)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.8, color="#d7ddd8")
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9aa8a0")

    for yi, mean in enumerate(means):
        ax.text(mean + 0.0015, yi, f"{mean:.3f}", va="center", ha="left", fontsize=9, color="#1f2c27")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    if not summary:
        raise SystemExit("No comparable downstream result sets with all required tasks were found.")
    make_plot(summary)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
