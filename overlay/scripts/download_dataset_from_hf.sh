#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-${SOKE_HF_DATASET_REPO:-}}"
TARGET_DIR="${2:-${SOKENAR_DATA_ROOT:-${SOKE_DATA_ROOT:-datasets}}}"

if [[ -z "$REPO_ID" ]]; then
  echo "[ERROR] Missing repo id."
  echo "Usage: $0 <HF_USER/DATASET_REPO> [TARGET_DIR]"
  echo "or set SOKE_HF_DATASET_REPO in environment."
  exit 1
fi

echo "[INFO] Downloading private dataset: $REPO_ID"
echo "[INFO] Target dir: $TARGET_DIR"

mkdir -p "$TARGET_DIR"
export REPO_ID TARGET_DIR
export SOKE_SPLITS_ROOT="${SOKE_SPLITS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data/splits}"

python - <<'PY'
import os
import tarfile
import zipfile
import shutil
import time
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ.get("REPO_ID")
target_dir = os.environ.get("TARGET_DIR")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
splits_root = Path(os.environ.get("SOKE_SPLITS_ROOT", "")).resolve()

if not repo_id:
    raise SystemExit("REPO_ID not provided")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    token=token,
    resume_download=True,
)

root = Path(target_dir)
def ensure_csl_poses_alias():
    csl_root = root / "CSL-Daily"
    poses = csl_root / "poses"
    legacy = csl_root / "csl-daily_pose"
    if poses.exists() or not legacy.exists():
        return
    try:
        poses.symlink_to(legacy.name)
        print("[INFO] created alias: CSL-Daily/poses -> csl-daily_pose")
    except Exception:
        pass

def archive_markers(name: str):
    if name in ("How2Sign.tar.gz", "How2Sign.zip"):
        return [
            root / "How2Sign" / "train" / "re_aligned" / "how2sign_realigned_train_preprocessed_fps.csv",
            root / "How2Sign" / "val" / "re_aligned" / "how2sign_realigned_val_preprocessed_fps.csv",
            root / "How2Sign" / "test" / "re_aligned" / "how2sign_realigned_test_preprocessed_fps.csv",
            root / "How2Sign" / "train" / "poses",
            root / "How2Sign" / "val" / "poses",
            root / "How2Sign" / "test" / "poses",
        ]
    if name in ("CSL-Daily.tar.gz", "CSL-Daily.zip"):
        has_pose_root = (root / "CSL-Daily" / "poses").exists() or (root / "CSL-Daily" / "csl-daily_pose").exists()
        return [
            root / "CSL-Daily" / "csl_clean.train",
            root / "CSL-Daily" / "csl_clean.val",
            root / "CSL-Daily" / "csl_clean.test",
            root / "CSL-Daily" / "mean.pt",
            root / "CSL-Daily" / "std.pt",
            Path("/__virtual_exists__") if has_pose_root else Path("/__virtual_missing__"),
        ]
    if name in ("Phoenix_2014T.tar.gz", "Phoenix_2014T.zip"):
        has_pose_root = any((root / "Phoenix_2014T" / p).exists() for p in ("train", "dev", "test"))
        return [
            root / "Phoenix_2014T" / "phoenix14t.train",
            root / "Phoenix_2014T" / "phoenix14t.dev",
            root / "Phoenix_2014T" / "phoenix14t.test",
            Path("/__virtual_exists__") if has_pose_root else Path("/__virtual_missing__"),
        ]
    return []

def archive_stamp_path(name: str):
    return root / f".soke_extracted_{name.replace('/', '_')}.ok"

def archive_expected_roots(name: str):
    if name in ("How2Sign.tar.gz", "How2Sign.zip"):
        return [root / "How2Sign"]
    if name in ("CSL-Daily.tar.gz", "CSL-Daily.zip"):
        return [root / "CSL-Daily"]
    if name in ("Phoenix_2014T.tar.gz", "Phoenix_2014T.zip"):
        return [root / "Phoenix_2014T"]
    return []

def write_archive_stamp(name: str):
    archive_stamp_path(name).write_text(f"ok {int(time.time())}\n", encoding="utf-8")

archive_names = [
    "How2Sign.tar.gz",
    "CSL-Daily.tar.gz",
    "Phoenix_2014T.tar.gz",
    "How2Sign.zip",
    "CSL-Daily.zip",
    "Phoenix_2014T.zip",
]

for name in archive_names:
    p = root / name
    if not p.exists():
        continue
    force = str(os.environ.get("SOKE_FORCE_REEXTRACT", "")).strip().lower()
    forced = force in ("1", "true", "yes", "on")
    stamp = archive_stamp_path(name)
    expected_roots = archive_expected_roots(name)
    if (not forced) and stamp.exists() and (not expected_roots or all(r.exists() for r in expected_roots)):
        print(f"[INFO] Skipping extract for {p.name} (stamp already present)")
        continue
    markers = archive_markers(name)
    if (not forced) and markers and all(m.exists() for m in markers):
        print(f"[INFO] Skipping extract for {p.name} (all markers already present)")
        continue
    print(f"[INFO] Extracting {p.name} ...")
    if p.suffix == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(root)
    else:
        with tarfile.open(p, "r:*") as tf:
            tf.extractall(root)
    ensure_csl_poses_alias()
    write_archive_stamp(name)

mapping = [
    (
        splits_root / "how2sign" / "how2sign_realigned_train_preprocessed_fps.csv",
        root / "How2Sign" / "train" / "re_aligned" / "how2sign_realigned_train_preprocessed_fps.csv",
    ),
    (
        splits_root / "how2sign" / "how2sign_realigned_val_preprocessed_fps.csv",
        root / "How2Sign" / "val" / "re_aligned" / "how2sign_realigned_val_preprocessed_fps.csv",
    ),
    (
        splits_root / "how2sign" / "how2sign_realigned_test_preprocessed_fps.csv",
        root / "How2Sign" / "test" / "re_aligned" / "how2sign_realigned_test_preprocessed_fps.csv",
    ),
    (
        splits_root / "csl_daily" / "csl_clean.train",
        root / "CSL-Daily" / "csl_clean.train",
    ),
    (
        splits_root / "csl_daily" / "csl_clean.val",
        root / "CSL-Daily" / "csl_clean.val",
    ),
    (
        splits_root / "csl_daily" / "csl_clean.test",
        root / "CSL-Daily" / "csl_clean.test",
    ),
    (
        splits_root / "phoenix" / "phoenix14t.train",
        root / "Phoenix_2014T" / "phoenix14t.train",
    ),
    (
        splits_root / "phoenix" / "phoenix14t.dev",
        root / "Phoenix_2014T" / "phoenix14t.dev",
    ),
    (
        splits_root / "phoenix" / "phoenix14t.test",
        root / "Phoenix_2014T" / "phoenix14t.test",
    ),
]

if splits_root.exists():
    repaired = 0
    for src, dst in mapping:
        if not src.exists():
            continue
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        repaired += 1
    if repaired:
        print(f"[INFO] Repaired split files from repo: {repaired}")

print("[OK] Dataset sync completed.")
PY
