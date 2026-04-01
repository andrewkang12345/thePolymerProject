#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_paths.sh"
cd "${P1M_PROJECT_ROOT}"

export WANDB_DIR="${P1M_WANDB_ROOT}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

RUN_NAME="dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_bigsmilestrans_vw1_tw1_e2"
CKPT="outputs/${RUN_NAME}/best.pt"
GPU2_TASKS=("Egc" "Egb" "Eea" "Ei")
GPU3_TASKS=("Xc" "EPS" "Nc" "Eat")

echo "=== Ensuring BigSMILES converter path is available ==="
python3 - <<'PY'
from p1m_pretrain.bigsmiles import convert_psmiles_to_bigsmiles

sample = "*N[Si](*)(C)C"
print(sample, "->", convert_psmiles_to_bigsmiles(sample))
PY

echo "=== Phase 1: 2-epoch deep corrected dual run with pSELFIES MLM + BigSMILES translation on GPUs 2-3 ==="
CUDA_VISIBLE_DEVICES=2,3 python3 scripts/train_one.py \
  --backbone dual_correctdeepchem_pselfies_shared \
  --scratch-variant deep \
  --init-mode scratch \
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
  --mlm-selfies-mix \
  --translation-target-mode bigsmiles \
  --multi-gpu \
  --num-workers-override 4 \
  --wandb-project p1m-fullscale

echo "=== Phase 2: downstream evaluation on GPUs 2-3 ==="
(
  for task in "${GPU2_TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python3 scripts/evaluate_downstream.py \
      --checkpoint "${CKPT}" \
      --task "${task}" \
      --folds 5 \
      --epochs 50 \
      --patience 8 \
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
      --output "downstream_results/${RUN_NAME}_${task}.json" \
      --wandb-project p1m-downstream-eval
  done
) &
pid3=$!

wait "${pid2}" "${pid3}"
python3 scripts/publish_downstream_summary.py --model "${RUN_NAME}" --wandb-project p1m-downstream-eval --allow-partial

echo "=== pSELFIES-MLM + BigSMILES-translation deep corrected dual run complete ==="
