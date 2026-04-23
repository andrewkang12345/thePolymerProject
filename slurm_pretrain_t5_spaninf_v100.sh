#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:2
#SBATCH -J p1m-t5span-e2
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

if [[ -z "${PROJECT:-}" ]]; then
  echo "PROJECT is not set; run this on Bridges-2 after logging in through the PSC environment." >&2
  exit 1
fi

REPO_ROOT="${PROJECT}/polymer_pretrain_experiments"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/load_paths.sh"

projects
id -gn
echo "${PROJECT}"
my_quotas
pwd
ls -lah

for required_path in "${P1M_TRANSPOLYMER_REPO}" "${P1M_MMPOLYMER_REPO}" "${P1M_MMPOLYMER_DATA_ROOT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path missing for downstream workflow: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${P1M_WANDB_ROOT}" "${REPO_ROOT}/outputs" "${REPO_ROOT}/downstream_results"

if [[ -n "${P1M_ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${P1M_ENV_ACTIVATE}"
elif [[ -n "${P1M_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${P1M_CONDA_ENV}"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export WANDB_DIR="${P1M_WANDB_ROOT}"
export TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
export EVAL_GPU_A="${EVAL_GPU_A:-0}"
export EVAL_GPU_B="${EVAL_GPU_B:-1}"

bash "${REPO_ROOT}/scripts/run_dual_correctdeepchem_pselfies_shared_deep_selfiest5spaninf_2gpu.sh"
