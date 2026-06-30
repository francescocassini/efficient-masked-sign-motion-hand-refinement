#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SOKENAR
mkdir -p "${SOKENAR_DATA_ROOT:-/workspace/datasets}" \
  "${SOKENAR_ARTIFACT_ROOT:-/workspace/artifacts}/checkpoints"

mode="${1:-smoke}"
shift || true

case "$mode" in
  models)
    exec python scripts/download_models.py "$@"
    ;;
  data)
    exec bash scripts/download_dataset_from_hf.sh \
      "${SOKENAR_DATA_REPO:?Set SOKENAR_DATA_REPO}" \
      "${SOKENAR_DATA_ROOT:-/workspace/datasets}"
    ;;
  train-p3)
    exec python -m train --cfg configs/train/p3_csl_phoenix.yaml \
      --nodebug --use_gpus 0 --device 0 --num_nodes 1 "$@"
    ;;
  infer-p3|infer-p5)
    dataset="${1:-phoenix}"
    shift || true
    variant="${mode#infer-}"
    exec python -m test --cfg "configs/infer/${variant}_${dataset}.yaml" \
      --task t2m --nodebug --use_gpus 0 --device 0 --num_nodes 1 "$@"
    ;;
  smoke)
    exec make smoke
    ;;
  shell|bash)
    exec bash "$@"
    ;;
  *)
    exec "$mode" "$@"
    ;;
esac

