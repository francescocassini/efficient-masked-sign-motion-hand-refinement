# Efficient Masked Decoding for Sign Motion Generation with Hand-Aware Token Refinement

Text-to-sign motion generation maps written sentences to articulated signing
sequences. Token-based generators such as SOKE encode motion as separate body,
left hand and right hand Vector Quantized (VQ) streams and generate them
autoregressively from text, so decoding cost grows with sequence length. This
work keeps the tokenizer, the anatomical streams and the VQ decoder fixed and
replaces sequential decoding with a three-stage hand-aware pipeline:

- **Masked-NAR** generates the three token streams in parallel through
  iterative masked non-autoregressive decoding, with the token length
  estimated from the text;
- **HandPolish** reopens and regenerates low-confidence hand tokens at
  inference time while the body stream remains fixed, without training an
  additional refinement network;
- **PoseSelect** selects among top-k alternative hand tokens with a learned
  selector that uses only pose features available at inference time.

On CSL-Daily and Phoenix-2014T, Masked-NAR improves three of the four direct
pose metrics over the autoregressive baseline, with end-to-end speedups
between 1.73x and 2.02x and about a quarter of the measured GPU energy.
HandPolish and PoseSelect further improve hand articulation at a measured
extra cost.

This repository contains the code needed to reproduce the experiments of the
paper.

![Pipeline overview](overlay/assets/figures/figure_2.png)

## Code Map

| Stage | Main code |
|---|---|
| **Masked-NAR** | `mGPT/archs/mgpt_mbart_nar_p3_train_aligned.py` |
| **HandPolish** | `mGPT/archs/mgpt_mbart_nar_p5_hand_polish_aggressive.py` |
| **PoseSelect** | `scripts/train_poseselect.py`, `scripts/eval_poseselect.py`, `mGPT/models/utils/p6_topk_candidate_selector.py` |

The evaluation also includes **SOKE-AR** (the autoregressive baseline),
**GainEdit** (a deployable post-cache hand-token editing baseline),
**OracleSelect** (a diagnostic ceiling that uses ground-truth supervision) and
the wrapper scripts for the paper tables (`scripts/reproduce_table*.py`).

## Repository Shape

This work is built on top of the official SOKE codebase:

| Item | Value |
|---|---|
| Upstream repository | https://github.com/2000ZRL/SOKE |
| Required upstream commit | `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00` |

Because upstream SOKE is distributed under `CC BY-NC-ND 4.0`, this repository
does not publish a full modified SOKE tree. Instead, it provides a patch and an
overlay that reconstruct the complete working code locally.

| Path | Purpose |
|---|---|
| `patches/soke-integration.patch` | modifications to upstream SOKE files |
| `overlay/` | new files added by this paper |
| `scripts/apply_delta.sh` | applies the patch and copies the overlay |
| `docs/EXTERNAL_ARTIFACTS.md` | detailed data, weights and cache setup |

All paths and commands below refer to the reconstructed SOKE checkout.

## 1. Reconstruct the Codebase

From an empty directory:

```bash
git clone https://github.com/2000ZRL/SOKE.git
git -C SOKE checkout 5cbc55d84b5a7cbf05a9cf020c468052e8d94d00

git clone <THIS_REPOSITORY_URL> paper-delta
bash paper-delta/scripts/apply_delta.sh SOKE

cd SOKE
python tests/test_release_layout.py
```

The final command checks that the reconstructed tree contains the paper code,
configs and wrappers.

Manual reconstruction is equivalent to:

```bash
git -C SOKE apply ../paper-delta/patches/soke-integration.patch
cp -a ../paper-delta/overlay/. SOKE/
```

## 2. Create the Environment

Use the upstream SOKE environment as the base:

```bash
conda create python=3.10 --name soke
conda activate soke
pip install -r requirements.txt
python -m pip install huggingface_hub hf-transfer
```

The reconstructed codebase also contains Docker helpers:

```bash
docker compose build
docker compose run --rm sokenar smoke
```

Docker still expects datasets and model artifacts to be placed or mounted in the
paths described below.

## 3. Download External Assets

Datasets, generated caches, checkpoints and large model weights are not
committed to this repository.

After reconstruction:

```bash
cp .env.example .env
```

Then edit `.env` if you use non-default local paths.

### Upstream SOKE Assets

These assets are inherited from SOKE and are required before running full
training/evaluation.

| Asset | Source | Expected path |
|---|---|---|
| How2Sign raw videos | https://how2sign.github.io/ | `datasets/How2Sign/` |
| How2Sign SOKE split files | https://drive.google.com/drive/folders/1sPhBwmiWCXLZSHtM3fpotbz3BDgoYmco?usp=sharing | `datasets/How2Sign/` |
| CSL-Daily raw videos | http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/ | `datasets/CSL-Daily/` |
| CSL-Daily SOKE split files | https://drive.google.com/drive/folders/17uPm6r5_DQ9CIYZonfwQLpw1XI8LeNEr?usp=drive_link | `datasets/CSL-Daily/` |
| Phoenix-2014T raw videos | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ | `datasets/Phoenix_2014T/` |
| Phoenix-2014T SOKE split files | https://drive.google.com/drive/folders/1Z2zjOH5wvwT7x_F6IycWAN-nh2wgJOx1?usp=sharing | `datasets/Phoenix_2014T/` |
| SOKE SMPL-X poses | https://2000zrl.github.io/soke/ | dataset-specific pose folders under `datasets/` |
| Human models | https://drive.google.com/file/d/1YIXddvvBJPQVRuKON2Xc9EEDXikRTteo/view?usp=sharing | `deps/smpl_models/` |
| mBART assets | https://drive.google.com/drive/folders/1GnaHrI0PC4ZRr-GK3FS2GXcQwlrpA5Gi?usp=sharing | `deps/mbart-h2s-csl-phoenix/` |
| CSL mean | https://drive.google.com/file/d/1NH-eVtS0nNjMjCwae-A1ii5sxj44C3bo/view?usp=sharing | `datasets/CSL-Daily/mean.pt` |
| CSL std | https://drive.google.com/file/d/1FHHWS0GPM2s6S2PB2JHv4ufdEbzezuKW/view?usp=sharing | `datasets/CSL-Daily/std.pt` |
| SOKE tokenizer checkpoint | https://drive.google.com/file/d/18HdPeXwz4O6LY4FZMC5BZ9rja4pcUTFk/view?usp=sharing | `deps/tokenizer_ckpt/tokenizer.ckpt` |

### Paper-Specific Artifacts

The paper-specific checkpoints and caches are not upstream SOKE assets. They
must be created by running the experiments below, or downloaded from a public
artifact release if one is published.

| Paper-facing role | Expected path |
|---|---|
| SOKE-AR baseline checkpoint | `artifacts/checkpoints/soke_ar_e69.ckpt` |
| Masked-NAR checkpoint used for HandPolish | `artifacts/checkpoints/masked_nar_e19.ckpt` |
| Masked-NAR direct-generation checkpoint | `artifacts/checkpoints/masked_nar_e49.ckpt` |
| Default Masked-NAR runtime checkpoint | `artifacts/checkpoints/p3.ckpt` |
| Hand-token editor used by PoseSelect features | `artifacts/p6b/p6b.ckpt` |
| GainEdit regressor checkpoint | `artifacts/gainedit/gainedit.ckpt` |
| PoseSelect checkpoint | `artifacts/poseselect/poseselect.ckpt` |
| HandPolish cache replicas | `artifacts/handpolish_cache/rep0/` ... `rep4/` |

The default runtime checkpoint `artifacts/checkpoints/p3.ckpt` is used by the
simple inference configs. It should usually point to the Masked-NAR checkpoint
used for HandPolish:

```bash
ln -s masked_nar_e19.ckpt artifacts/checkpoints/p3.ckpt
```

### Creating Paper-Specific Artifacts

The intended reproducibility path is to create the paper artifacts from data:

| Artifact | Creation path |
|---|---|
| SOKE-AR baseline checkpoint | train/evaluate the upstream-compatible SOKE-AR baseline using the reconstructed SOKE environment |
| Masked-NAR checkpoints | run **Experiment 1** below with `configs/train/p3_csl_phoenix.yaml`; select the checkpoints used by the table configs |
| HandPolish cache replicas | run **Experiment 2** with prediction saving enabled, or `configs/paper/table4_poseselect_postcache.yaml` for matched-cache evaluation |
| Hand-token editor | train the included hand-token editor scripts on saved HandPolish caches |
| GainEdit regressor | train with `scripts/train_gainedit.py` on saved HandPolish caches |
| PoseSelect selector | train with `scripts/train_poseselect.py` on saved HandPolish caches |

If a public model artifact bundle is released, it should use the file names in
the table above so the expected paths remain unchanged.

### Optional Dataset Archive Mirror

If you maintain a local or institutional mirror of prepared datasets, package
them as:

```text
How2Sign.tar.gz
CSL-Daily.tar.gz
Phoenix_2014T.tar.gz
```

Then run:

```bash
bash scripts/download_dataset_from_hf.sh YOUR_DATASET_REPO_OR_MIRROR datasets
```

See `docs/EXTERNAL_ARTIFACTS.md` for the full artifact checklist.

## 4. Source and Artifact Sanity Checks

Run these checks before the scientific experiments.

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

Reference values for every paper table are recorded in
`docs/PAPER_RESULTS.md`.

## 5. Experiment 1: Masked-NAR Generation

Masked-NAR replaces autoregressive token generation with iterative masked
parallel decoding in the same SOKE-compatible token space. Length is estimated
from text at inference, the body, left-hand and right-hand prediction heads
are stream-specific, iterative decoding reopens low-confidence positions with
a masked schedule, and the generated tokens are decoded by the unchanged VQ
decoder.

| Role | File |
|---|---|
| generator | `mGPT/archs/mgpt_mbart_nar_p3_train_aligned.py` |
| LM config | `configs/lm/mbart_h2s_csl_phoenix_nar_p3_train_aligned.yaml` |
| training config | `configs/train/p3_csl_phoenix.yaml` |
| inference configs | `configs/infer/p3_csl.yaml`, `configs/infer/p3_phoenix.yaml` |

Train Masked-NAR:

```bash
python -m train --cfg configs/train/p3_csl_phoenix.yaml \
  --nodebug --use_gpus 0 --device 0 --num_nodes 1
```

Run Masked-NAR inference on Phoenix:

```bash
python -m test --cfg configs/infer/p3_phoenix.yaml \
  --task t2m --nodebug --use_gpus 0 --device 0 --num_nodes 1
```

List the paper Table 2 configurations:

```bash
python scripts/reproduce_table2_pose.py
```

Then execute the printed Masked-NAR config with:

```bash
python -u -m test --cfg configs/paper/table2_masked_nar_direct.yaml \
  --task t2m --nodebug
```

## 6. Experiment 2: HandPolish

HandPolish reuses the Masked-NAR probabilities at inference to reopen
low-confidence hand tokens while keeping body tokens fixed. It adds no
trainable parameters: body tokens are frozen before hand remasking begins,
only low-confidence left/right hand positions are reopened, and the
refinement stays inside the discrete SOKE-compatible hand codebooks.

| Role | File |
|---|---|
| hand-only refinement | `mGPT/archs/mgpt_mbart_nar_p5_hand_polish_aggressive.py` |
| LM config | `configs/lm/mbart_h2s_csl_phoenix_nar_p5_hand_polish_aggressive.yaml` |
| inference configs | `configs/infer/p5_csl.yaml`, `configs/infer/p5_phoenix.yaml` |

Run Masked-NAR + HandPolish inference:

```bash
python -m test --cfg configs/infer/p5_phoenix.yaml \
  --task t2m --nodebug --use_gpus 0 --device 0 --num_nodes 1
```

Run the paper Table 2 HandPolish operating point:

```bash
python -u -m test --cfg configs/paper/table2_handpolish.yaml \
  --task t2m --nodebug
```

## 7. Experiment 3: End-to-End Efficiency

The efficiency benchmark measures the complete test loop (generation, VQ
decoding, exact-DTW metrics and prediction saving) for SOKE-AR, Masked-NAR and
Masked-NAR + HandPolish on Phoenix-200 and CSL-200.

Entrypoint:

```bash
python scripts/reproduce_table3_efficiency.py
```

This wrapper lists the benchmark scripts used for Table 3:

| Dataset | Script/config family |
|---|---|
| Phoenix-200 | `scripts/t0a_efficiency_benchmark.py`, `configs/paper/table3_*_phoenix200.yaml` |
| CSL-200 | `scripts/t0a_efficiency_benchmark_csl.py`, `configs/paper/table3_*_csl200.yaml` |

GPU energy numbers require the same hardware-monitoring setup used in the
paper.

## 8. Experiment 4: PoseSelect Post-Cache Refinement

PoseSelect is a learned selector over top-k hand-token candidates built from
saved HandPolish caches. Training labels are computed offline; at inference
the selector uses only generated-cache and candidate-token features, and the
body stream is copied unchanged from the cache.

| Role | File |
|---|---|
| training wrapper | `scripts/train_poseselect.py` |
| evaluation wrapper | `scripts/eval_poseselect.py` |
| selector model | `mGPT/models/utils/p6_topk_candidate_selector.py` |
| Table 4 aggregation | `scripts/reproduce_table4_refinement.py` |

Train PoseSelect from saved HandPolish caches:

```bash
python scripts/train_poseselect.py \
  --cache-dir artifacts/handpolish_cache/train \
  --val-cache-dir artifacts/handpolish_cache/val \
  --p6b-checkpoint artifacts/p6b/p6b.ckpt \
  --regressor-checkpoint artifacts/gainedit/gainedit.ckpt \
  --output-dir artifacts/poseselect
```

Evaluate PoseSelect:

```bash
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

Aggregate the paper Table 4 matched-cache protocol:

```bash
python scripts/reproduce_table4_refinement.py
```

## 9. Experiment 5: Post-Cache Overhead

The overhead benchmark starts from a validated HandPolish cache and measures
the deployable post-cache refinement variants (PoseSelect and GainEdit)
separately from native generation.

Run:

```bash
python scripts/reproduce_table5_overhead.py --help
```

The underlying benchmark is `scripts/benchmark_p6k_t0a_style.py`.

## Citation

If this artifact is useful, cite the paper:

```bibtex
@inproceedings{anonymous2026maskedsign,
  title={Efficient Masked Decoding for Sign Motion Generation with Hand-Aware Token Refinement},
  author={Anonymous},
  booktitle={Under review},
  year={2026}
}
```
