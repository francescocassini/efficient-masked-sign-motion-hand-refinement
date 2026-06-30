#!/usr/bin/env python3
"""Download private P3/P5 runtime artifacts from Hugging Face."""

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    repo_id = os.environ.get("SOKENAR_MODEL_REPO")
    if not repo_id:
        raise SystemExit("Set SOKENAR_MODEL_REPO to the private HF model repo.")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    root = Path(os.environ.get("SOKENAR_MODEL_ROOT", "artifacts/models")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=root,
        token=token,
    )

    mappings = {
        "p3.ckpt": Path(os.environ.get("SOKENAR_CHECKPOINT", "artifacts/checkpoints/p3.ckpt")),
        "tokenizer.ckpt": Path("deps/tokenizer_ckpt/tokenizer.ckpt"),
    }
    for source_name, destination in mappings.items():
        source = root / source_name
        if not source.exists():
            print(f"[models] optional artifact not present: {source_name}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[models] installed {source_name} -> {destination}")

    mbart_source = root / "mbart-h2s-csl-phoenix"
    mbart_destination = Path("deps/mbart-h2s-csl-phoenix")
    if mbart_source.exists() and not mbart_destination.exists():
        mbart_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(mbart_source, mbart_destination)
        print(f"[models] installed mBART assets -> {mbart_destination}")


if __name__ == "__main__":
    main()

