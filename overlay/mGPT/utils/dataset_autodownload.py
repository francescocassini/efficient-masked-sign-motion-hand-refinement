import os
import subprocess
from pathlib import Path
import tarfile
import time
import zipfile
import shutil


def _log(msg: str):
    print(f"[dataset-autodl] {msg}")


def _required_paths(cfg):
    h2s_root = Path(cfg.DATASET.H2S.ROOT)
    csl_root = Path(cfg.DATASET.H2S.CSL_ROOT)
    pho_root = Path(cfg.DATASET.H2S.PHOENIX_ROOT)
    mean_path = Path(cfg.DATASET.H2S.MEAN_PATH)
    std_path = Path(cfg.DATASET.H2S.STD_PATH)

    req = [
        h2s_root / "train" / "re_aligned" / "how2sign_realigned_train_preprocessed_fps.csv",
        h2s_root / "val" / "re_aligned" / "how2sign_realigned_val_preprocessed_fps.csv",
        h2s_root / "test" / "re_aligned" / "how2sign_realigned_test_preprocessed_fps.csv",
        csl_root / "csl_clean.train",
        csl_root / "csl_clean.val",
        csl_root / "csl_clean.test",
        pho_root / "phoenix14t.train",
        pho_root / "phoenix14t.dev",
        pho_root / "phoenix14t.test",
        mean_path,
        std_path,
    ]
    return req


def _all_present(paths):
    return all(p.exists() for p in paths)


def _missing_paths(paths):
    return [p for p in paths if not p.exists()]


def _resolve_data_root(cfg):
    if os.environ.get("SOKENAR_DATA_ROOT"):
        return Path(os.environ["SOKENAR_DATA_ROOT"]).expanduser().resolve()
    if os.environ.get("SOKE_DATA_ROOT"):
        return Path(os.environ["SOKE_DATA_ROOT"]).expanduser().resolve()
    # Fallback: infer as common parent for dataset roots.
    h2s_root = Path(cfg.DATASET.H2S.ROOT).resolve()
    return h2s_root.parent


def _repo_root() -> Path:
    # mGPT/utils/dataset_autodownload.py -> repo root is 2 levels up from mGPT/
    return Path(__file__).resolve().parents[2]


def _repair_split_files_from_repo(cfg):
    repo_root = _repo_root()
    split_root = repo_root / "data" / "splits"
    if not split_root.exists():
        return 0

    mapping = [
        (
            split_root / "how2sign" / "how2sign_realigned_train_preprocessed_fps.csv",
            Path(cfg.DATASET.H2S.ROOT) / "train" / "re_aligned" / "how2sign_realigned_train_preprocessed_fps.csv",
        ),
        (
            split_root / "how2sign" / "how2sign_realigned_val_preprocessed_fps.csv",
            Path(cfg.DATASET.H2S.ROOT) / "val" / "re_aligned" / "how2sign_realigned_val_preprocessed_fps.csv",
        ),
        (
            split_root / "how2sign" / "how2sign_realigned_test_preprocessed_fps.csv",
            Path(cfg.DATASET.H2S.ROOT) / "test" / "re_aligned" / "how2sign_realigned_test_preprocessed_fps.csv",
        ),
        (
            split_root / "csl_daily" / "csl_clean.train",
            Path(cfg.DATASET.H2S.CSL_ROOT) / "csl_clean.train",
        ),
        (
            split_root / "csl_daily" / "csl_clean.val",
            Path(cfg.DATASET.H2S.CSL_ROOT) / "csl_clean.val",
        ),
        (
            split_root / "csl_daily" / "csl_clean.test",
            Path(cfg.DATASET.H2S.CSL_ROOT) / "csl_clean.test",
        ),
        (
            split_root / "phoenix" / "phoenix14t.train",
            Path(cfg.DATASET.H2S.PHOENIX_ROOT) / "phoenix14t.train",
        ),
        (
            split_root / "phoenix" / "phoenix14t.dev",
            Path(cfg.DATASET.H2S.PHOENIX_ROOT) / "phoenix14t.dev",
        ),
        (
            split_root / "phoenix" / "phoenix14t.test",
            Path(cfg.DATASET.H2S.PHOENIX_ROOT) / "phoenix14t.test",
        ),
    ]

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
        _log(f"repaired split file from repo: {dst}")

    return repaired


def _run(cmd, cwd=None, check=True):
    _log("run: " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check)


def _extract_archive(archive_path: Path, target_root: Path):
    start = time.time()
    _log(f"extracting archive: {archive_path.name}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_root)
        _log(f"extracted archive: {archive_path.name} in {time.time() - start:.1f}s")
        return

    # tar, tar.gz, tgz
    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(target_root)
    _log(f"extracted archive: {archive_path.name} in {time.time() - start:.1f}s")


def _ensure_csl_poses_alias(data_root: Path):
    csl_root = data_root / "CSL-Daily"
    poses = csl_root / "poses"
    legacy = csl_root / "csl-daily_pose"
    if poses.exists():
        return False
    if not legacy.exists():
        return False
    try:
        poses.symlink_to(legacy.name)
        _log("created alias: CSL-Daily/poses -> csl-daily_pose")
    except Exception:
        # Symlink can fail on restricted filesystems; fallback to copy is too expensive.
        return False
    return True


def _archive_markers(data_root: Path, archive_name: str):
    if archive_name == "How2Sign.tar.gz" or archive_name == "How2Sign.zip":
        return [
            data_root / "How2Sign" / "train" / "re_aligned" / "how2sign_realigned_train_preprocessed_fps.csv",
            data_root / "How2Sign" / "val" / "re_aligned" / "how2sign_realigned_val_preprocessed_fps.csv",
            data_root / "How2Sign" / "test" / "re_aligned" / "how2sign_realigned_test_preprocessed_fps.csv",
            data_root / "How2Sign" / "train" / "poses",
            data_root / "How2Sign" / "val" / "poses",
            data_root / "How2Sign" / "test" / "poses",
        ]
    if archive_name == "CSL-Daily.tar.gz" or archive_name == "CSL-Daily.zip":
        # Accept both canonical and legacy pose folder names.
        has_pose_root = (data_root / "CSL-Daily" / "poses").exists() or (data_root / "CSL-Daily" / "csl-daily_pose").exists()
        return [
            data_root / "CSL-Daily" / "csl_clean.train",
            data_root / "CSL-Daily" / "csl_clean.val",
            data_root / "CSL-Daily" / "csl_clean.test",
            data_root / "CSL-Daily" / "mean.pt",
            data_root / "CSL-Daily" / "std.pt",
            Path("/__virtual_exists__") if has_pose_root else Path("/__virtual_missing__"),
        ]
    if archive_name == "Phoenix_2014T.tar.gz" or archive_name == "Phoenix_2014T.zip":
        has_pose_root = any(
            (data_root / "Phoenix_2014T" / p).exists() for p in ("train", "dev", "test")
        )
        return [
            data_root / "Phoenix_2014T" / "phoenix14t.train",
            data_root / "Phoenix_2014T" / "phoenix14t.dev",
            data_root / "Phoenix_2014T" / "phoenix14t.test",
            Path("/__virtual_exists__") if has_pose_root else Path("/__virtual_missing__"),
        ]
    return []


def _archive_stamp_path(data_root: Path, archive_name: str) -> Path:
    safe = archive_name.replace("/", "_")
    return data_root / f".soke_extracted_{safe}.ok"


def _archive_expected_roots(data_root: Path, archive_name: str):
    if archive_name in ("How2Sign.tar.gz", "How2Sign.zip"):
        return [data_root / "How2Sign"]
    if archive_name in ("CSL-Daily.tar.gz", "CSL-Daily.zip"):
        return [data_root / "CSL-Daily"]
    if archive_name in ("Phoenix_2014T.tar.gz", "Phoenix_2014T.zip"):
        return [data_root / "Phoenix_2014T"]
    return []


def _write_archive_stamp(data_root: Path, archive_name: str):
    stamp = _archive_stamp_path(data_root, archive_name)
    stamp.write_text(f"ok {int(time.time())}\n")


def _archive_already_extracted(data_root: Path, archive_name: str):
    force = os.environ.get("SOKE_FORCE_REEXTRACT", "").strip().lower()
    if force in ("1", "true", "yes", "on"):
        return False

    stamp = _archive_stamp_path(data_root, archive_name)
    expected_roots = _archive_expected_roots(data_root, archive_name)
    if stamp.exists() and (not expected_roots or all(p.exists() for p in expected_roots)):
        return True

    markers = _archive_markers(data_root, archive_name)
    return len(markers) > 0 and all(m.exists() for m in markers)


def _extract_known_archives_if_present(data_root: Path):
    # Archive-first mode: upload a few big files instead of millions of tiny files.
    archive_names = [
        "How2Sign.tar.gz",
        "CSL-Daily.tar.gz",
        "Phoenix_2014T.tar.gz",
        "How2Sign.zip",
        "CSL-Daily.zip",
        "Phoenix_2014T.zip",
    ]
    found = False
    for name in archive_names:
        archive_path = data_root / name
        if archive_path.exists():
            found = True
            if _archive_already_extracted(data_root, name):
                _log(f"skip extract {archive_path.name} (all markers already present)")
                continue
            _extract_archive(archive_path, data_root)
            _ensure_csl_poses_alias(data_root)
            _write_archive_stamp(data_root, name)
    if found:
        _log("archive extraction completed")


def _sync_with_hf_hub(repo_id: str, data_root: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        _log(f"huggingface_hub not available ({e}), fallback to git/lfs")
        return False

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    _log("trying snapshot download via huggingface_hub")
    data_root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_root),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    return True


def ensure_dataset_available(cfg):
    req = _required_paths(cfg)

    repaired_local = _repair_split_files_from_repo(cfg)
    if repaired_local > 0:
        _log(f"local split repair applied: {repaired_local}")
        req = _required_paths(cfg)

    if _all_present(req):
        _log("dataset already present; skip HF sync")
        return

    missing_before = _missing_paths(req)
    _log(f"required files missing before sync: {len(missing_before)}")
    for p in missing_before[:5]:
        _log(f"missing: {p}")

    repo_id = os.environ.get("SOKE_HF_DATASET_REPO", "").strip()
    if not repo_id:
        missing = [str(p) for p in req if not p.exists()][:5]
        raise FileNotFoundError(
            "Dataset files are missing and SOKE_HF_DATASET_REPO is not set.\n"
            "Set SOKE_HF_DATASET_REPO=<user>/<private_dataset_repo> and login via 'huggingface-cli login'.\n"
            f"Example missing paths: {missing}"
        )

    data_root = _resolve_data_root(cfg)
    hf_url = f"https://huggingface.co/datasets/{repo_id}"
    _log(f"dataset missing, trying Hugging Face private dataset: {repo_id}")
    _log(f"target local data root: {data_root}")

    data_root.parent.mkdir(parents=True, exist_ok=True)

    # Preferred: HF Hub API (supports HF_TOKEN without git credentials).
    synced = False
    try:
        synced = _sync_with_hf_hub(repo_id, data_root)
    except Exception as e:
        _log(f"snapshot download failed ({e}), fallback to git/lfs")

    # Fallback: git-lfs clone/pull.
    if not synced:
        _run(["git", "lfs", "install"], check=False)

        if not data_root.exists():
            _run(["git", "clone", hf_url, str(data_root)])
        elif (data_root / ".git").exists():
            _run(["git", "-C", str(data_root), "pull"], check=False)
        else:
            raise RuntimeError(
                f"Data root exists but is not a git repo: {data_root}\n"
                "Please remove/rename it, or set SOKE_DATA_ROOT to a clean path."
            )

        if (data_root / ".git").exists():
            _run(["git", "-C", str(data_root), "lfs", "pull"], check=False)

    # Re-check
    if not _all_present(req):
        _extract_known_archives_if_present(data_root)
        repaired_after_extract = _repair_split_files_from_repo(cfg)
        if repaired_after_extract > 0:
            _log(f"post-extract split repair applied: {repaired_after_extract}")

    req_after = _required_paths(cfg)
    if not _all_present(req_after):
        missing = [str(p) for p in req_after if not p.exists()][:10]
        raise FileNotFoundError(
            "Hugging Face dataset sync finished but required files are still missing.\n"
            f"Missing examples: {missing}"
        )

    _log("dataset ready")
