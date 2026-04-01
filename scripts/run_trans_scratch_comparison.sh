#!/bin/bash
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_paths.sh"
cd "${P1M_PROJECT_ROOT}"

echo "=== Training both transpolymer scratch runs in parallel (2 GPUs each) ==="

# Run 1: transpolymer scratch, MLM only (vw=0, tw=0)
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_one.py \
  --backbone transpolymer \
  --init-mode scratch \
  --run-name trans_scratch_mlm_only_e2 \
  --validation-protocol external_polymer_mix_v1 \
  --train-size 0 \
  --val-size 2000 \
  --batch-size 128 \
  --learning-rate 5e-5 \
  --weight-decay 0.01 \
  --view-weight 0.0 \
  --translation-weight 0.0 \
  --epochs 2 \
  --multi-gpu \
  --num-workers-override 4 \
  --wandb-project p1m-fullscale &
PID1=$!

# Run 2: transpolymer scratch, vw=1 tw=1
CUDA_VISIBLE_DEVICES=2,3 python3 scripts/train_one.py \
  --backbone transpolymer \
  --init-mode scratch \
  --run-name trans_scratch_vw1_tw1_e2 \
  --validation-protocol external_polymer_mix_v1 \
  --train-size 0 \
  --val-size 2000 \
  --batch-size 128 \
  --learning-rate 5e-5 \
  --weight-decay 0.01 \
  --view-weight 1.0 \
  --translation-weight 1.0 \
  --epochs 2 \
  --multi-gpu \
  --num-workers-override 4 \
  --wandb-project p1m-fullscale &
PID2=$!

echo "Training PIDs: $PID1 (mlm_only), $PID2 (vw1_tw1)"
wait $PID1
echo "=== trans_scratch_mlm_only_e2 training done ==="
wait $PID2
echo "=== trans_scratch_vw1_tw1_e2 training done ==="

echo "=== Downstream eval: 8 tasks per model, 4 GPUs ==="
TASKS=("Egc" "Egb" "Eea" "Ei" "Xc" "EPS" "Nc" "Eat")

# Eval model 1: mlm_only
echo "--- Evaluating trans_scratch_mlm_only_e2 ---"
for i in 0 1 2 3; do
  t1=${TASKS[$((i*2))]}
  t2=${TASKS[$((i*2+1))]}
  (
    CUDA_VISIBLE_DEVICES=$i python3 scripts/evaluate_downstream.py \
      --checkpoint outputs/trans_scratch_mlm_only_e2/best.pt --task "$t1" --folds 5 --epochs 50 --patience 8 \
      --output downstream_results/trans_scratch_mlm_only_e2_${t1}.json \
      --wandb-project p1m-downstream-eval
    CUDA_VISIBLE_DEVICES=$i python3 scripts/evaluate_downstream.py \
      --checkpoint outputs/trans_scratch_mlm_only_e2/best.pt --task "$t2" --folds 5 --epochs 50 --patience 8 \
      --output downstream_results/trans_scratch_mlm_only_e2_${t2}.json \
      --wandb-project p1m-downstream-eval
  ) &
done
wait
echo "--- trans_scratch_mlm_only_e2 eval done ---"
python3 scripts/publish_downstream_summary.py --model trans_scratch_mlm_only_e2 --wandb-project p1m-downstream-eval --allow-partial

# Eval model 2: vw1_tw1
echo "--- Evaluating trans_scratch_vw1_tw1_e2 ---"
for i in 0 1 2 3; do
  t1=${TASKS[$((i*2))]}
  t2=${TASKS[$((i*2+1))]}
  (
    CUDA_VISIBLE_DEVICES=$i python3 scripts/evaluate_downstream.py \
      --checkpoint outputs/trans_scratch_vw1_tw1_e2/best.pt --task "$t1" --folds 5 --epochs 50 --patience 8 \
      --output downstream_results/trans_scratch_vw1_tw1_e2_${t1}.json \
      --wandb-project p1m-downstream-eval
    CUDA_VISIBLE_DEVICES=$i python3 scripts/evaluate_downstream.py \
      --checkpoint outputs/trans_scratch_vw1_tw1_e2/best.pt --task "$t2" --folds 5 --epochs 50 --patience 8 \
      --output downstream_results/trans_scratch_vw1_tw1_e2_${t2}.json \
      --wandb-project p1m-downstream-eval
  ) &
done
wait
echo "--- trans_scratch_vw1_tw1_e2 eval done ---"
python3 scripts/publish_downstream_summary.py --model trans_scratch_vw1_tw1_e2 --wandb-project p1m-downstream-eval --allow-partial

echo "=== All done ==="
