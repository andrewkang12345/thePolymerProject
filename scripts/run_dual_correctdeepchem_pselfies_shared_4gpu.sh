#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_paths.sh"
cd "${P1M_PROJECT_ROOT}"

export WANDB_DIR="${P1M_WANDB_ROOT}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

RUN_NAME="dual_correctdeepchem_pselfies_shared_scratch_vw1_tw1_e2"
CKPT="outputs/${RUN_NAME}/best.pt"
TASKS=("Egc" "Egb" "Eea" "Ei" "Xc" "EPS" "Nc" "Eat")

echo "=== Ensuring corrected DeepChem SMILES + atomic pSELFIES tokenizer artifacts ==="
python3 - <<'PY'
from p1m_pretrain.dual_tokenizer import ensure_deepchem_smiles_tokenizer, ensure_deepchem_vocab_txt, load_pselfies_tokenizer

deepchem_dir = ensure_deepchem_smiles_tokenizer()
vocab_txt = ensure_deepchem_vocab_txt()
pselfies = load_pselfies_tokenizer()
print(f"DeepChem tokenizer dir: {deepchem_dir}")
print(f"DeepChem vocab.txt: {vocab_txt}")
print(f"pSELFIES vocab size: {len(pselfies)}")
PY

echo "=== Phase 1: 2-epoch scratch dual-tokenizer training on all GPUs ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/train_one.py \
  --backbone dual_correctdeepchem_pselfies_shared \
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
  --multi-gpu \
  --num-workers-override 4 \
  --wandb-project p1m-fullscale

echo "=== Phase 2: downstream evaluation across all GPUs ==="
for i in 0 1 2 3; do
  t1=${TASKS[$((i * 2))]}
  t2=${TASKS[$((i * 2 + 1))]}
  (
    CUDA_VISIBLE_DEVICES=$i python3 scripts/evaluate_downstream.py \
      --checkpoint "${CKPT}" \
      --task "${t1}" \
      --folds 5 \
      --epochs 50 \
      --patience 8 \
      --output "downstream_results/${RUN_NAME}_${t1}.json" \
      --wandb-project p1m-downstream-eval
    CUDA_VISIBLE_DEVICES=$i python3 scripts/evaluate_downstream.py \
      --checkpoint "${CKPT}" \
      --task "${t2}" \
      --folds 5 \
      --epochs 50 \
      --patience 8 \
      --output "downstream_results/${RUN_NAME}_${t2}.json" \
      --wandb-project p1m-downstream-eval
  ) &
done
wait
python3 scripts/publish_downstream_summary.py --model "${RUN_NAME}" --wandb-project p1m-downstream-eval --allow-partial

echo "=== Corrected dual-tokenizer run complete ==="
