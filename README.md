# Efficient Masked Sign Motion

Public delta artifact for:

**Efficient Masked Decoding for Sign Motion Generation with Hand-Aware Token Refinement**

This repository contains the code, configs and documentation needed to recreate
the complete working codebase locally from the official SOKE repository. It is a
**delta artifact**, not a full modified SOKE fork.

## Why This Is a Delta

The upstream SOKE project is distributed under `CC BY-NC-ND 4.0`. Because that
license includes a NoDerivatives term, this repository does not redistribute a
complete modified SOKE tree. Instead, users apply this delta to the official
SOKE source on their own machine.

Pinned upstream base:

| Item | Value |
|---|---|
| Upstream SOKE | https://github.com/2000ZRL/SOKE |
| Required commit | `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00` |

This repository provides:

| Path | Purpose |
|---|---|
| `patches/soke-integration.patch` | edits to files that already exist in SOKE |
| `overlay/` | new files added by this work |
| `scripts/apply_delta.sh` | patch + overlay reconstruction helper |
| `docs/EXTERNAL_ARTIFACTS.md` | datasets, weights and cache placement guide |

## Method Summary

The artifact implements three SOKE-compatible extensions while keeping the SOKE
tokenizer, anatomical token streams and frozen VQ decoder interface intact.

| Paper name | Historical name | Summary |
|---|---|---|
| Masked-NAR | P3 | masked non-autoregressive body/left-hand/right-hand token generation with dataset-specific text-length estimation |
| HandPolish | P5 | training-free hand-only remasking that keeps body tokens fixed |
| PoseSelect | P6-K | learned post-cache top-k hand-token selector using inference-available features |

The repo also includes GainEdit and OracleSelect as post-cache baseline and
diagnostic comparator, plus configs/wrappers for the paper Tables 2-5.

## Reconstruct the Complete Code

From an empty directory:

```bash
git clone https://github.com/2000ZRL/SOKE.git
git -C SOKE checkout 5cbc55d84b5a7cbf05a9cf020c468052e8d94d00

git clone https://github.com/francescocassini/efficient-masked-sign-motion-delta.git
bash efficient-masked-sign-motion-delta/scripts/apply_delta.sh SOKE

cd SOKE
python tests/test_release_layout.py
```

The helper checks the target checkout, applies
`patches/soke-integration.patch`, and copies `overlay/` into the SOKE tree.

Manual equivalent:

```bash
git -C SOKE apply ../efficient-masked-sign-motion-delta/patches/soke-integration.patch
cp -a ../efficient-masked-sign-motion-delta/overlay/. SOKE/
```

Manual application is useful for auditing the patch, resolving conflicts, or
adapting to a newer SOKE commit. The pinned commit above is the validated target.

## Environment

Use the upstream SOKE Python environment as the base:

```bash
conda create python=3.10 --name soke
conda activate soke
pip install -r requirements.txt
python -m pip install huggingface_hub hf-transfer
```

After applying the delta, the reconstructed tree also includes Docker helpers:

```bash
docker compose build
docker compose run --rm sokenar smoke
```

Docker still requires datasets and model artifacts to be mounted or downloaded
into the paths documented below.

## Datasets and Weights

This Git repository intentionally excludes datasets, generated caches,
checkpoints, model weights and private split archives.

After applying the delta:

```bash
cp .env.example .env
```

Then edit `.env` for your local paths or private Hugging Face repositories.

### Upstream SOKE Assets

These links come from the official SOKE README and are needed by SOKE-compatible
training/evaluation.

| Asset | Download/source | Expected path after setup |
|---|---|---|
| How2Sign raw videos | https://how2sign.github.io/ | `datasets/How2Sign/` |
| How2Sign SOKE split files | https://drive.google.com/drive/folders/1sPhBwmiWCXLZSHtM3fpotbz3BDgoYmco?usp=sharing | `datasets/How2Sign/` |
| CSL-Daily raw videos | http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/ | `datasets/CSL-Daily/` |
| CSL-Daily SOKE split files | https://drive.google.com/drive/folders/17uPm6r5_DQ9CIYZonfwQLpw1XI8LeNEr?usp=drive_link | `datasets/CSL-Daily/` |
| Phoenix-2014T raw videos | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ | `datasets/Phoenix_2014T/` |
| Phoenix-2014T SOKE split files | https://drive.google.com/drive/folders/1Z2zjOH5wvwT7x_F6IycWAN-nh2wgJOx1?usp=sharing | `datasets/Phoenix_2014T/` |
| SOKE SMPL-X poses | https://2000zrl.github.io/soke/ | dataset-specific pose folders under `datasets/` |
| Human body models | https://drive.google.com/file/d/1YIXddvvBJPQVRuKON2Xc9EEDXikRTteo/view?usp=sharing | `deps/smpl_models/` |
| mBART assets | https://drive.google.com/drive/folders/1GnaHrI0PC4ZRr-GK3FS2GXcQwlrpA5Gi?usp=sharing | `deps/mbart-h2s-csl-phoenix/` |
| CSL mean | https://drive.google.com/file/d/1NH-eVtS0nNjMjCwae-A1ii5sxj44C3bo/view?usp=sharing | `datasets/CSL-Daily/mean.pt` |
| CSL std | https://drive.google.com/file/d/1FHHWS0GPM2s6S2PB2JHv4ufdEbzezuKW/view?usp=sharing | `datasets/CSL-Daily/std.pt` |
| SOKE tokenizer checkpoint | https://drive.google.com/file/d/18HdPeXwz4O6LY4FZMC5BZ9rja4pcUTFk/view?usp=sharing | `deps/tokenizer_ckpt/tokenizer.ckpt` |

### Paper-Specific Weights

The following weights are **not** provided by upstream SOKE. They must be
trained locally or distributed separately by this artifact's authors.

| Artifact | Expected path |
|---|---|
| SOKE-AR e69 baseline checkpoint | `artifacts/checkpoints/soke_ar_e69.ckpt` |
| Masked-NAR e19 checkpoint | `artifacts/checkpoints/masked_nar_e19.ckpt` |
| Masked-NAR e49 checkpoint | `artifacts/checkpoints/masked_nar_e49.ckpt` |
| Default runtime Masked-NAR checkpoint | `artifacts/checkpoints/p3.ckpt` |
| P6-B hand-token editor checkpoint | `artifacts/p6b/p6b.ckpt` |
| GainEdit regressor checkpoint | `artifacts/gainedit/gainedit.ckpt` |
| PoseSelect checkpoint | `artifacts/poseselect/poseselect.ckpt` |
| HandPolish R5 caches | `artifacts/handpolish_cache/rep0/` ... `rep4/` |

`artifacts/checkpoints/p3.ckpt` is the default checkpoint used by
`configs/infer/p3_*.yaml` and `configs/infer/p5_*.yaml`. It should usually be
the same file as `masked_nar_e19.ckpt`; copying or symlinking is fine:

```bash
ln -s masked_nar_e19.ckpt artifacts/checkpoints/p3.ckpt
```

### Optional Private Hugging Face Repositories

If you keep model artifacts in a private Hugging Face model repo, use these file
names:

```text
p3.ckpt
masked_nar_e19.ckpt
masked_nar_e49.ckpt
soke_ar_e69.ckpt
p6b.ckpt
gainedit.ckpt
poseselect.ckpt
tokenizer.ckpt
mbart-h2s-csl-phoenix/
```

Then run:

```bash
export HF_TOKEN=...
export SOKENAR_MODEL_REPO=YOUR_HF_USER/YOUR_MODEL_REPO
python scripts/download_models.py
```

If you keep prepared datasets in a private Hugging Face dataset repo, package
them as:

```text
How2Sign.tar.gz
CSL-Daily.tar.gz
Phoenix_2014T.tar.gz
```

Then run:

```bash
export HF_TOKEN=...
export SOKENAR_DATA_REPO=YOUR_HF_USER/YOUR_DATASET_REPO
bash scripts/download_dataset_from_hf.sh "$SOKENAR_DATA_REPO" datasets
```

Full placement details are in `docs/EXTERNAL_ARTIFACTS.md`.

## Smoke Checks

Source-only checks:

```bash
python tests/test_release_layout.py
python -m py_compile \
  scripts/reproduce_table2_pose.py \
  scripts/reproduce_table3_efficiency.py \
  scripts/reproduce_table4_refinement.py \
  scripts/reproduce_table5_overhead.py \
  scripts/train_poseselect.py \
  scripts/eval_poseselect.py \
  scripts/train_gainedit.py \
  scripts/eval_gainedit.py \
  scripts/eval_oracleselect.py
```

Artifact-aware checks:

```bash
test -f deps/tokenizer_ckpt/tokenizer.ckpt
test -d deps/mbart-h2s-csl-phoenix
test -f artifacts/checkpoints/p3.ckpt
test -f datasets/CSL-Daily/mean.pt
test -f datasets/CSL-Daily/std.pt
```

## Main Commands

Train Masked-NAR:

```bash
python -m train --cfg configs/train/p3_csl_phoenix.yaml \
  --nodebug --use_gpus 0 --device 0 --num_nodes 1
```

Run Masked-NAR inference:

```bash
python -m test --cfg configs/infer/p3_phoenix.yaml \
  --task t2m --nodebug --use_gpus 0 --device 0 --num_nodes 1
```

Run Masked-NAR + HandPolish inference:

```bash
python -m test --cfg configs/infer/p5_phoenix.yaml \
  --task t2m --nodebug --use_gpus 0 --device 0 --num_nodes 1
```

Train and evaluate PoseSelect from saved HandPolish caches:

```bash
python scripts/train_poseselect.py \
  --cache-dir artifacts/handpolish_cache/train \
  --val-cache-dir artifacts/handpolish_cache/val \
  --p6b-checkpoint artifacts/p6b/p6b.ckpt \
  --regressor-checkpoint artifacts/gainedit/gainedit.ckpt \
  --output-dir artifacts/poseselect

python scripts/eval_poseselect.py \
  --dataset both \
  --cache-dir artifacts/handpolish_cache/test \
  --p6b-checkpoint artifacts/p6b/p6b.ckpt \
  --regressor-checkpoint artifacts/gainedit/gainedit.ckpt \
  --selector-checkpoint artifacts/poseselect/poseselect.ckpt \
  --output-dir results/poseselect_eval \
  --mean-path datasets/CSL-Daily/mean.pt \
  --std-path datasets/CSL-Daily/std.pt
```

## Reproduce Paper Tables

```bash
python scripts/reproduce_table2_pose.py
python scripts/reproduce_table3_efficiency.py
python scripts/reproduce_table4_refinement.py --help
python scripts/reproduce_table5_overhead.py --help
```

Required inputs:

| Table | Scope | Required artifacts |
|---|---|---|
| Table 2 | native PA-JPE benchmark | tokenizer, mBART assets, datasets, SOKE-AR e69, Masked-NAR e49/e19 |
| Table 3 | end-to-end efficiency | same as Table 2, plus GPU runtime monitor for energy numbers |
| Table 4 | post-cache refinement | HandPolish R5 caches, tokenizer, P6-B, GainEdit, PoseSelect |
| Table 5 | post-cache overhead | validated HandPolish cache, P6-B/GainEdit/PoseSelect, tokenizer |

Recorded table values are in `docs/PAPER_RESULTS.md` after applying the delta.

## Code Map

| Path | Purpose |
|---|---|
| `mGPT/archs/mgpt_mbart_nar_p3_train_aligned.py` | final Masked-NAR implementation |
| `mGPT/archs/mgpt_mbart_nar_p5_hand_polish_aggressive.py` | final HandPolish implementation |
| `mGPT/models/utils/p6_topk_candidate_selector.py` | PoseSelect model utility |
| `scripts/train_poseselect.py`, `scripts/eval_poseselect.py` | paper-facing PoseSelect entrypoints |
| `configs/train/p3_csl_phoenix.yaml` | Masked-NAR training config |
| `configs/infer/p3_*.yaml`, `configs/infer/p5_*.yaml` | direct inference configs |
| `configs/paper/table*.yaml` | sanitized table reproduction configs |
| `docs/METHOD_MAPPING.md` | detailed paper-name to code mapping |
| `docs/PAPER_RESULTS.md` | paper table values |
| `docs/EXTERNAL_ARTIFACTS.md` | data/model/caches setup |

## Limitations

- This is a delta artifact, not a standalone SOKE replacement.
- Datasets, checkpoints and caches are intentionally not committed.
- PoseSelect is a learned post-cache selector, not a native generation row.
- OracleSelect uses ground-truth information and is only a diagnostic ceiling.
- Exact paper numbers require the documented external artifacts and protocol.
- The combined SOKE+delta checkout should not be redistributed as a full
  modified SOKE tree without a separate license/provenance decision.

## Citation

If this artifact is useful, cite both this work and upstream SOKE:

```bibtex
@inproceedings{zuo2025soke,
  title={Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator},
  author={Zuo, Ronglai and Potamias, Rolandos Alexandros and Ververas, Evangelos and Deng, Jiankang and Zafeiriou, Stefanos},
  booktitle={ICCV},
  year={2025}
}
```
