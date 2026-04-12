#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_paths.sh"
cd "${P1M_PROJECT_ROOT}"

export WANDB_DIR="${P1M_WANDB_ROOT}"

COMMON="--validation-protocol external_polymer_mix_v1 \
  --train-size 0 --val-size 2000 --batch-size 128 \
  --multi-gpu --num-workers-override 4 \
  --learning-rate 5e-5 \
  --view-weight 1.0 --translation-weight 1.0 \
  --wandb-project p1m-fullscale"

echo "=== Run 1/2: Scratch, original tokenizer, 30 epochs ==="
python3 scripts/train_one.py \
  --backbone transpolymer --init-mode scratch \
  --run-name scratch_origtok_vw1_tw1 \
  --epochs 30 \
  $COMMON

echo "=== Run 2/2: TransPolymer checkpoint, original tokenizer, 5 epochs ==="
python3 scripts/train_one.py \
  --backbone transpolymer --init-mode checkpoint \
  --run-name transpolymer_cont_vw1_tw1 \
  --epochs 5 \
  $COMMON

echo "=== All training complete. Starting downstream eval ==="

for name in scratch_origtok_vw1_tw1 transpolymer_cont_vw1_tw1; do
  echo "Evaluating $name..."
  WANDB_DIR="${P1M_WANDB_ROOT}" python3 scripts/evaluate_downstream.py \
    --checkpoint "outputs/${name}/best.pt" \
    --task all --folds 5 --epochs 50 --patience 8 \
    --wandb-project p1m-downstream-eval \
    2>&1 | tee "downstream_results/${name}_eval.log"
done

echo "=== All done ==="
