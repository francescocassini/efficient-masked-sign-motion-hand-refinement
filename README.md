# Efficient Masked Sign Motion: public delta artifact

This repository is a public delta artifact for the paper draft:

**Efficient Masked Sign Motion Generation with Conservative Hand Refinement**

It does not redistribute the full SOKE codebase. Instead, it provides:

- `overlay/`: files added by this work;
- `patches/soke-integration.patch`: integration changes for files already present in SOKE;
- `scripts/apply_delta.sh`: a reproducible helper that applies the delta to an official SOKE checkout;
- documentation and paper-result summaries needed to understand the artifact.

## Why this is a delta

The upstream SOKE repository is licensed under `CC BY-NC-ND 4.0`. The
NoDerivatives term means that publishing a full modified copy of SOKE is not the
right release shape without separate permission from the SOKE authors.

This repository is therefore structured as a delta: users obtain SOKE from the
official source and apply the additional research files from this artifact.

Upstream SOKE:

```bash
git clone https://github.com/2000ZRL/SOKE.git
cd SOKE
git checkout 5cbc55d84b5a7cbf05a9cf020c468052e8d94d00
```

## Apply the delta

From any directory:

```bash
git clone https://github.com/francescocassini/efficient-masked-sign-motion-delta.git
cd efficient-masked-sign-motion-delta
bash scripts/apply_delta.sh /path/to/SOKE
```

The script checks the target checkout, applies `patches/soke-integration.patch`,
and copies `overlay/` into the SOKE tree.

## What the delta adds

The artifact adds the paper implementation around the SOKE-compatible tokenizer,
decoder and evaluation stack:

- Masked non-autoregressive generation variants;
- train-aligned corruption for masked generation;
- conservative hand-only refinement;
- PoseSelect and gain-edit selection utilities;
- paper experiment configs;
- reproduction wrappers for the main tables;
- paper result summaries and qualitative figure assets.

The delta does not include datasets, model checkpoints, generated caches, or
dataset-derived split archives. Those must be obtained through the original
dataset/model channels and placed in the paths described by the applied README
and configs.

## Verification

After applying the delta inside a SOKE checkout, run:

```bash
python tests/test_release_layout.py
python -m py_compile \
  scripts/reproduce_table2_pose.py \
  scripts/reproduce_table3_efficiency.py \
  scripts/reproduce_table4_refinement.py \
  scripts/reproduce_table5_overhead.py \
  scripts/train_poseselect.py \
  scripts/eval_poseselect.py
```

Full training/evaluation requires the external SOKE-compatible datasets and
checkpoints.

## Citation

If this artifact is useful, cite the paper and the upstream SOKE work. This
delta depends on SOKE and is not a standalone replacement for it.
