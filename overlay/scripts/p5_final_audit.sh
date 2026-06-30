#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${SOKENAR_DATA_ROOT:-datasets}"
MEAN="${SOKENAR_CSL_MEAN_PATH:-$DATA_ROOT/CSL-Daily/mean.pt}"
STD="${SOKENAR_CSL_STD_PATH:-$DATA_ROOT/CSL-Daily/std.pt}"

audit_run() {
  local candidate_name="$1"
  local reference_name="$2"
  local dataset="$3"
  local candidate_dir="results/mgpt/${candidate_name}/test_rank_0"
  local reference_dir="results/mgpt/${reference_name}/test_rank_0"
  local audit_file="results/mgpt/${candidate_name}/vitality_audit.txt"

  python scripts/motion_vitality_compare.py \
    --reference "$reference_dir" \
    --candidate "$candidate_dir" \
    --candidate-name "$candidate_name" \
    --mean "$MEAN" \
    --std "$STD" \
    --diversity-samples 200 | tee "$audit_file"

  python scripts/render_inference_gifs.py \
    --run_name "$candidate_name" \
    --dataset "$dataset" \
    --mean_path "$MEAN" \
    --std_path "$STD" \
    --top_k 3
}

audit_run SOKENAR_P5_AGGRESSIVE_E19_CSL_FULL SOKENAR_P3_E19_CSL_FULL csl
audit_run SOKENAR_P5_AGGRESSIVE_E19_PHX_FULL SOKENAR_P3_E19_PHX_FULL phoenix
