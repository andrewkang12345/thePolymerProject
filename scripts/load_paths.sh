#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_FILE="${ROOT_DIR}/paths.txt"

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "Missing path config: ${PATHS_FILE}" >&2
  exit 1
fi

get_path_value() {
  local key="$1"
  local raw
  raw="$(grep -E "^${key}=" "${PATHS_FILE}" | tail -n 1 | cut -d= -f2-)"
  if [[ -z "${raw}" ]]; then
    echo "Missing ${key} in ${PATHS_FILE}" >&2
    exit 1
  fi
  if [[ "${raw}" = /* ]]; then
    printf '%s\n' "${raw}"
  else
    printf '%s\n' "$(cd "${ROOT_DIR}" && realpath "${raw}")"
  fi
}

export P1M_PROJECT_ROOT="$(get_path_value project_root)"
export P1M_DATA_ROOT="$(get_path_value data_root)"
export P1M_POLY_ANY2ANY_ROOT="$(get_path_value poly_any2any_root)"
export P1M_TRANSPOLYMER_REPO="$(get_path_value transpolymer_repo)"
export P1M_MMPOLYMER_REPO="$(get_path_value mmpolymer_repo)"
export P1M_MMPOLYMER_DATA_ROOT="$(get_path_value mmpolymer_data_root)"
export P1M_BIGSMILES_REPO="$(get_path_value bigsmiles_repo)"
export P1M_GRAPHDIT_ROOT="$(get_path_value graphdit_root)"
export P1M_WANDB_ROOT="$(get_path_value wandb_root)"
export PYTHONPATH="${P1M_PROJECT_ROOT}/src:${PYTHONPATH:-}"
