#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_paths.sh"
cd "${P1M_PROJECT_ROOT}"

export WANDB_DIR="${P1M_WANDB_ROOT}"

RUN_NAME="${RUN_NAME:-dual_correctdeepchem_pselfies_shared_deep_selfiest5spaninf_vw1_tw1_e2}"
CKPT="outputs/${RUN_NAME}/best.pt"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
EVAL_GPU_A="${EVAL_GPU_A:-0}"
EVAL_GPU_B="${EVAL_GPU_B:-1}"
GPU_A_TASKS=("Egc" "Egb" "Eea" "Ei")
GPU_B_TASKS=("Xc" "EPS" "Nc" "Eat")

echo "=== Phase 1: 2-epoch deep corrected dual run with pSELFIES T5 span infilling on GPUs ${TRAIN_GPUS} ==="
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" python3 scripts/train_one.py \
  --backbone dual_correctdeepchem_pselfies_shared \
  --scratch-variant deep \
  --init-mode scratch \
  --pretrain-objective t5_span_infilling \
  --run-name "${RUN_NAME}" \
  --validation-protocol external_polymer_mix_v1 \
  --train-size 0 \
  --val-size 2000 \
  --batch-size 128 \
  --learning-rate 5e-5 \
  --weight-decay 0.01 \
  --view-weight 1.0 \
  --translation-weight 1.0 \
  --epochs 2 \
  --mlm-probability 0.15 \
  --mlm-selfies-mix \
  --multi-gpu \
  --num-workers-override 4 \
  --wandb-project p1m-fullscale

echo "=== Phase 2: downstream evaluation on GPUs ${EVAL_GPU_A}-${EVAL_GPU_B} ==="
(
  for task in "${GPU_A_TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_A}" python3 scripts/evaluate_downstream.py \
      --checkpoint "${CKPT}" \
      --task "${task}" \
      --folds 5 \
      --epochs 50 \
      --patience 8 \
      --output "downstream_results/${RUN_NAME}_${task}.json" \
      --wandb-project p1m-downstream-eval
  done
) &
pid_a=$!

(
  for task in "${GPU_B_TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_B}" python3 scripts/evaluate_downstream.py \
      --checkpoint "${CKPT}" \
      --task "${task}" \
      --folds 5 \
      --epochs 50 \
      --patience 8 \
      --output "downstream_results/${RUN_NAME}_${task}.json" \
      --wandb-project p1m-downstream-eval
  done
) &
pid_b=$!

wait "${pid_a}" "${pid_b}"
python3 scripts/publish_downstream_summary.py --model "${RUN_NAME}" --wandb-project p1m-downstream-eval --allow-partial

echo "=== pSELFIES T5 span-infilling deep corrected dual run complete ==="
