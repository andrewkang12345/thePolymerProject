#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_paths.sh"
cd "${P1M_PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_DIR="${P1M_WANDB_ROOT}"

RUN_NAME="${RUN_NAME:-polyone1m_dual_correctdeepchem_deep_selfiesmlm_vw1_tw1}"
POLY_ANY2ANY_ROOT="${P1M_POLY_ANY2ANY_ROOT}"
POLYONE_ROOT="${POLYONE_ROOT:-${POLY_ANY2ANY_ROOT}/data/raw/polyone_1m/original}"

python3 scripts/train_polyone_100m.py \
  --polyone-root "${POLYONE_ROOT}" \
  --run-name "${RUN_NAME}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --steps "${STEPS:-7813}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --save-every "${SAVE_EVERY:-500}" \
  --val-size "${VAL_SIZE:-4096}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --learning-rate "${LEARNING_RATE:-5e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --view-weight "${VIEW_WEIGHT:-1.0}" \
  --translation-weight "${TRANSLATION_WEIGHT:-1.0}" \
  --scratch-variant deep \
  --wandb-project "${WANDB_PROJECT:-p1m-polyone1m}" \
  "$@"
