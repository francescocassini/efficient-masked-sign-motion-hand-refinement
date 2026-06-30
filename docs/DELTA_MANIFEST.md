# Delta manifest

Pinned upstream base:

- Repository: `https://github.com/2000ZRL/SOKE`
- Commit: `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00`

Delta layout:

- `overlay/configs/`: paper, training and inference configs added by this work.
- `overlay/mGPT/archs/`: masked-NAR and hand-polish architectures.
- `overlay/mGPT/models/utils/`: selector, gain-edit and residual-refinement helpers.
- `overlay/scripts/`: training, evaluation, benchmark and reproduction wrappers.
- `overlay/docs/`: paper result summaries and method mapping.
- `overlay/assets/figures/`: paper qualitative figure assets.
- `patches/soke-integration.patch`: modifications required in existing SOKE files.

Paper contribution coverage:

- **Masked SOKE-compatible generation**: `overlay/mGPT/archs/mgpt_mbart_nar*.py`,
  `overlay/configs/lm/*nar*.yaml`, `overlay/configs/train/p3_csl_phoenix.yaml`.
- **Training-free hand polishing**: `overlay/mGPT/archs/mgpt_mbart_nar_p5_hand_polish*.py`,
  `overlay/configs/infer/p5_*.yaml`.
- **Deployable hand-token selection**: `overlay/scripts/train_poseselect.py`,
  `overlay/scripts/eval_poseselect.py`,
  `overlay/mGPT/models/utils/p6_topk_candidate_selector.py`.
- **Controlled geometry, runtime and vitality evaluation**:
  `overlay/scripts/reproduce_table2_pose.py`,
  `overlay/scripts/reproduce_table3_efficiency.py`,
  `overlay/scripts/reproduce_table4_refinement.py`,
  `overlay/scripts/reproduce_table5_overhead.py`,
  `overlay/docs/PAPER_RESULTS.md`.

Excluded from the public delta:

- model checkpoints;
- generated prediction caches;
- dataset archives;
- dataset split CSV/TXT files;
- Python bytecode/cache folders;
- the full modified SOKE tree.
