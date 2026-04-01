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
export P1M_WANDB_ROOT="$(get_path_value wandb_root)"
export PYTHONPATH="${P1M_PROJECT_ROOT}/src:${PYTHONPATH:-}"
