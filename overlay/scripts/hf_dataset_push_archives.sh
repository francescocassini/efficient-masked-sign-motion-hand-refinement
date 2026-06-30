#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <HF_USER/DATASET_REPO> <ARCHIVES_DIR> [COMMIT_MSG]"
  echo "Example: $0 <user>/sokenar-data-assets artifacts/dataset_archives"
  exit 1
fi

REPO_ID="$1"
ARCHIVES_DIR="$2"
COMMIT_MSG="${3:-Update private SOKE dataset archives}"

if [[ ! -d "$ARCHIVES_DIR" ]]; then
  echo "[ERROR] Archive dir not found: $ARCHIVES_DIR"
  exit 1
fi

for f in "$ARCHIVES_DIR/How2Sign.tar.gz" "$ARCHIVES_DIR/CSL-Daily.tar.gz" "$ARCHIVES_DIR/Phoenix_2014T.tar.gz"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing archive: $f"
    exit 1
  fi
done

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[ERROR] huggingface-cli not found. Install with: pip install -U huggingface_hub"
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[ERROR] git-lfs not found. Install git-lfs first."
  exit 1
fi

echo "[INFO] Checking Hugging Face login..."
if ! huggingface-cli whoami >/dev/null 2>&1; then
  echo "[ERROR] Not logged in. Run: huggingface-cli login"
  exit 1
fi

WORK_DIR="$(mktemp -d /tmp/soke_hf_archives_repo.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "[INFO] Working dir: $WORK_DIR"
git lfs install
git clone "https://huggingface.co/datasets/$REPO_ID" "$WORK_DIR"

cd "$WORK_DIR"

git lfs track "*.tar.gz" "*.zip"
cp -f "$ARCHIVES_DIR/How2Sign.tar.gz" .
cp -f "$ARCHIVES_DIR/CSL-Daily.tar.gz" .
cp -f "$ARCHIVES_DIR/Phoenix_2014T.tar.gz" .

git add .gitattributes How2Sign.tar.gz CSL-Daily.tar.gz Phoenix_2014T.tar.gz

if git diff --cached --quiet; then
  echo "[INFO] No changes to push."
  exit 0
fi

git commit -m "$COMMIT_MSG"
git push --progress origin main

echo "[OK] Archive push completed: https://huggingface.co/datasets/$REPO_ID"
