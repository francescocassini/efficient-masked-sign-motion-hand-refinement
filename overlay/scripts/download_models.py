#!/usr/bin/env python3
"""Download optional private runtime artifacts from Hugging Face.

The public delta does not ship checkpoints or tokenizer assets. This helper is
only a convenience for users who keep those files in a private Hugging Face
model repository. Manual placement in the same destination paths is equivalent.
"""

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def copy_file(source: Path, destination: Path, *, required: bool = False) -> bool:
    if not source.exists():
        level = "missing required" if required else "optional artifact not present"
        print(f"[models] {level}: {source.name}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"[models] installed {source.name} -> {destination}")
    return True


def copy_dir(source: Path, destination: Path) -> bool:
    if not source.exists():
        print(f"[models] optional artifact not present: {source.name}/")
        return False
    if destination.exists():
        print(f"[models] keeping existing directory: {destination}")
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"[models] installed {source.name}/ -> {destination}")
    return True


def main() -> None:
    repo_id = os.environ.get("SOKENAR_MODEL_REPO")
    if not repo_id:
        raise SystemExit(
            "Set SOKENAR_MODEL_REPO to a private Hugging Face model repo, "
            "or place artifacts manually as described in docs/EXTERNAL_ARTIFACTS.md."
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    root = Path(os.environ.get("SOKENAR_MODEL_ROOT", "artifacts/models")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=root,
        token=token,
    )

    checkpoint_root = Path(os.environ.get("SOKENAR_ARTIFACT_ROOT", "artifacts")) / "checkpoints"
    mappings = {
        "p3.ckpt": Path(os.environ.get("SOKENAR_CHECKPOINT", "artifacts/checkpoints/p3.ckpt")),
        "masked_nar_e19.ckpt": Path(
            os.environ.get("SOKENAR_MASKED_NAR_E19", checkpoint_root / "masked_nar_e19.ckpt")
        ),
        "masked_nar_e49.ckpt": Path(
            os.environ.get("SOKENAR_MASKED_NAR_E49", checkpoint_root / "masked_nar_e49.ckpt")
        ),
        "soke_ar_e69.ckpt": Path(
            os.environ.get("SOKENAR_SOKE_AR_E69", checkpoint_root / "soke_ar_e69.ckpt")
        ),
        "poseselect.ckpt": Path(
            os.environ.get("SOKENAR_POSESELECT_CKPT", "artifacts/poseselect/poseselect.ckpt")
        ),
        "gainedit.ckpt": Path(
            os.environ.get("SOKENAR_GAINEDIT_CKPT", "artifacts/gainedit/gainedit.ckpt")
        ),
        "p6b.ckpt": Path(
            os.environ.get("SOKENAR_P6B_CKPT", "artifacts/p6b/p6b.ckpt")
        ),
        "tokenizer.ckpt": Path("deps/tokenizer_ckpt/tokenizer.ckpt"),
    }

    installed = {}
    for source_name, destination in mappings.items():
        installed[source_name] = copy_file(root / source_name, destination)

    if not installed.get("p3.ckpt") and installed.get("masked_nar_e19.ckpt"):
        default_checkpoint = Path(os.environ.get("SOKENAR_CHECKPOINT", "artifacts/checkpoints/p3.ckpt"))
        copy_file(root / "masked_nar_e19.ckpt", default_checkpoint)

    copy_dir(root / "mbart-h2s-csl-phoenix", Path("deps/mbart-h2s-csl-phoenix"))

    print("[models] Done. Missing optional files can be placed manually later.")


if __name__ == "__main__":
    main()
