#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_paths.sh"
cd "${P1M_PROJECT_ROOT}"

echo "=== Phase 1/4: concat-through-encoder downstream eval ==="
bash scripts/run_dual_correctdeepchem_deep_concat_encoder_eval_2gpu.sh

echo "=== Phase 2/4: dual-input downstream eval ==="
bash scripts/run_dual_correctdeepchem_deep_dualinput_eval_2gpu.sh

echo "=== Phase 3/4: deep corrected dual pretraining with pSELFIES MLM ==="
bash scripts/run_dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_2gpu.sh

echo "=== Phase 4/4: deep corrected dual pretraining with pSELFIES MLM + BigSMILES translation ==="
bash scripts/run_dual_correctdeepchem_pselfies_shared_deep_selfiesmlm_bigsmiles_2gpu.sh

echo "=== All requested follow-up runs complete ==="
