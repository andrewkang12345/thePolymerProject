#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_paths.sh"

ROOT="${P1M_PROJECT_ROOT}"
RUN_NAME="dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_vw1_tw1_e16_openpolymer_cv5"
OUT_DIR="${ROOT}/outputs/openpolymer_dual_correctdeepchem_e16"

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES=2,3 python3 scripts/finetune_openpolymer.py \
  --checkpoint "${ROOT}/outputs/dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_vw1_tw1_e16/best.pt" \
  --train-csv "${ROOT}/openPolymer/train.csv" \
  --test-csv "${ROOT}/openPolymer/test.csv" \
  --output-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --wandb-project "openpolymer-transfer" \
  --folds 5 \
  --epochs 25 \
  --patience 5 \
  --batch-size 16 \
  --lr-backbone 2e-5 \
  --lr-head 1e-4 \
  --num-workers 4 \
  --input-mode dual \
  --tg-fahrenheit-postprocess
