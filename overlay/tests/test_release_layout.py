#!/usr/bin/env python3
"""Dependency-free smoke checks for the private paper-release layout."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(path: str) -> None:
    assert (ROOT / path).exists(), f"Missing required release file: {path}"


def main() -> None:
    required = [
        "mGPT/archs/mgpt_mbart_nar_p3_train_aligned.py",
        "mGPT/archs/mgpt_mbart_nar_p5_hand_polish_base.py",
        "mGPT/archs/mgpt_mbart_nar_p5_hand_polish_aggressive.py",
        "mGPT/archs/mgpt_mbart_nar_p5_hand_polish_aggressive_t5c_conf.py",
        "mGPT/models/utils/p6_topk_candidate_selector.py",
        "mGPT/models/utils/p6_hand_gain_regressor.py",
        "mGPT/models/utils/p6_hand_token_editor.py",
        "configs/train/p3_csl_phoenix.yaml",
        "configs/infer/p3_csl.yaml",
        "configs/infer/p3_phoenix.yaml",
        "configs/infer/p5_csl.yaml",
        "configs/infer/p5_phoenix.yaml",
        "configs/paper/table2_soke_ar.yaml",
        "configs/paper/table2_masked_nar_direct.yaml",
        "configs/paper/table2_masked_nar_handpolish_base.yaml",
        "configs/paper/table2_handpolish.yaml",
        "configs/paper/table3_soke_ar_phoenix200.yaml",
        "configs/paper/table3_masked_nar_phoenix200.yaml",
        "configs/paper/table3_handpolish_phoenix200.yaml",
        "configs/paper/table3_soke_ar_csl200.yaml",
        "configs/paper/table3_masked_nar_csl200.yaml",
        "configs/paper/table3_handpolish_csl200.yaml",
        "configs/paper/table4_poseselect_postcache.yaml",
        "scripts/train_poseselect.py",
        "scripts/eval_poseselect.py",
        "scripts/eval_gainedit.py",
        "scripts/eval_oracleselect.py",
        "scripts/reproduce_table2_pose.py",
        "scripts/reproduce_table3_efficiency.py",
        "scripts/reproduce_table4_refinement.py",
        "scripts/reproduce_table5_overhead.py",
        "scripts/make_paper_qualitative_figure.py",
        "scripts/download_models.py",
        "scripts/download_dataset_from_hf.sh",
        "docs/METHOD_MAPPING.md",
        "docs/PAPER_RESULTS.md",
        "docs/PROVENANCE.md",
        "docs/RELEASE_CHECKLIST.md",
        "docs/results/table4_poseselect/summary.md",
        "docs/results/table4_poseselect/summary.json",
        "docs/results/table5_overhead/summary.json",
        "assets/figures/paper_qualitative_contrastive_compact_with_maskednar_2x.png",
    ]
    for path in required:
        require(path)

    forbidden = [
        "mGPT/archs/mgpt_mbart_nar_p4a_no_smoothing.py",
        "mGPT/models/mgpt_t4_residual_hand_refiner.py",
        "deps/tokenizer_ckpt/tokenizer.ckpt",
        "artifacts/checkpoints/p3.ckpt",
    ]
    for path in forbidden:
        assert not (ROOT / path).exists(), f"Forbidden release artifact present: {path}"

    blocked_suffixes = (".ckpt", ".pt", ".pth", ".safetensors", ".pkl", ".npy", ".npz")
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        assert not path.name.endswith(blocked_suffixes), f"Heavy/generated artifact present: {path}"

    print("[OK] AIxIA paper-release layout is complete.")


if __name__ == "__main__":
    main()
