#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <SOURCE_DATA_DIR> [OUTPUT_DIR]"
  echo "Example: $0 datasets artifacts/dataset_archives"
  exit 1
fi

SRC_DIR="$1"
OUT_DIR="${2:-$SRC_DIR}"

for d in "$SRC_DIR/How2Sign" "$SRC_DIR/CSL-Daily" "$SRC_DIR/Phoenix_2014T"; do
  if [[ ! -d "$d" ]]; then
    echo "[ERROR] Missing required dataset directory: $d"
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

create_tar() {
  local folder_name="$1"
  local src_parent="$2"
  local target_tar="$3"

  echo "[INFO] Creating $target_tar from $folder_name ..."
  # Dereference symlinks so archives contain real split files (Docker-safe).
  tar --dereference -C "$src_parent" -czf "$target_tar" "$folder_name"
  du -sh "$target_tar"
}

create_tar "How2Sign" "$SRC_DIR" "$OUT_DIR/How2Sign.tar.gz"
create_tar "CSL-Daily" "$SRC_DIR" "$OUT_DIR/CSL-Daily.tar.gz"
create_tar "Phoenix_2014T" "$SRC_DIR" "$OUT_DIR/Phoenix_2014T.tar.gz"

echo "[OK] Archive build completed in: $OUT_DIR"
