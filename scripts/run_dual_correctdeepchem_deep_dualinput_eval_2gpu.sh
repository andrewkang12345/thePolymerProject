#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_paths.sh"
cd "${P1M_PROJECT_ROOT}"

export WANDB_DIR="${P1M_WANDB_ROOT}"

BASE_RUN="dual_correctdeepchem_pselfies_shared_deep_vw1_tw1_e2"
RUN_NAME="${BASE_RUN}_dualinput"
CKPT="outputs/${BASE_RUN}/best.pt"
GPU2_TASKS=("Egc" "Egb" "Eea" "Ei")
GPU3_TASKS=("Xc" "EPS" "Nc" "Eat")

echo "=== Downstream eval: dual-embedding input on GPUs 2-3 ==="
echo "Checkpoint: ${CKPT}"
echo "Alias: ${RUN_NAME}"

(
  for task in "${GPU2_TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python3 scripts/evaluate_downstream.py \
      --checkpoint "${CKPT}" \
      --task "${task}" \
      --folds 5 \
      --epochs 50 \
      --patience 8 \
      --dual-input \
      --model-alias "${RUN_NAME}" \
      --output "downstream_results/${RUN_NAME}_${task}.json" \
      --wandb-project p1m-downstream-eval
  done
) &
pid2=$!

(
  for task in "${GPU3_TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluate_downstream.py \
      --checkpoint "${CKPT}" \
      --task "${task}" \
      --folds 5 \
      --epochs 50 \
      --patience 8 \
      --dual-input \
      --model-alias "${RUN_NAME}" \
      --output "downstream_results/${RUN_NAME}_${task}.json" \
      --wandb-project p1m-downstream-eval
  done
) &
pid3=$!

wait "${pid2}" "${pid3}"
python3 scripts/publish_downstream_summary.py --model "${RUN_NAME}" --wandb-project p1m-downstream-eval --allow-partial

echo "=== Dual-input downstream eval complete ==="
