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

Excluded from the public delta:

- model checkpoints;
- generated prediction caches;
- dataset archives;
- dataset split CSV/TXT files;
- Python bytecode/cache folders;
- the full modified SOKE tree.
