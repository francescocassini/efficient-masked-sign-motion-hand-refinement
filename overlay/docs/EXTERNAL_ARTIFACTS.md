# External Artifacts

This document describes the files that are intentionally not stored in the
public delta repository. Paths are relative to this reconstructed SOKE checkout.

Do not commit datasets, checkpoints, generated caches, split archives or access
tokens to Git.

## Required Directory Layout

```text
SOKE/
  datasets/
    How2Sign/
    CSL-Daily/
      mean.pt
      std.pt
    Phoenix_2014T/
  deps/
    tokenizer_ckpt/tokenizer.ckpt
    mbart-h2s-csl-phoenix/
  artifacts/
    checkpoints/
      soke_ar_e69.ckpt
      masked_nar_e19.ckpt
      masked_nar_e49.ckpt
      p3.ckpt
    handpolish_cache/
    p6b/
    poseselect/
    gainedit/
```

`p3.ckpt` is the default runtime checkpoint used by `configs/infer/p3_*.yaml`
and `configs/infer/p5_*.yaml`. It should usually point to the Masked-NAR e19
checkpoint, either by copying the file or by creating a local symlink.

## Artifact Table

| Artifact | Required for | Expected path | How to obtain |
|---|---|---|---|
| SOKE tokenizer/VQ checkpoint | all decoding and metrics | `deps/tokenizer_ckpt/tokenizer.ckpt` | SOKE tokenizer checkpoint: `https://drive.google.com/file/d/18HdPeXwz4O6LY4FZMC5BZ9rja4pcUTFk/view?usp=sharing` |
| mBART/text assets | text-conditioned generation | `deps/mbart-h2s-csl-phoenix/` | SOKE mBART assets: `https://drive.google.com/drive/folders/1GnaHrI0PC4ZRr-GK3FS2GXcQwlrpA5Gi?usp=sharing` |
| human body models | mesh rendering / SMPL-X utilities | `deps/smpl_models/` | SOKE human models bundle: `https://drive.google.com/file/d/1YIXddvvBJPQVRuKON2Xc9EEDXikRTteo/view?usp=sharing` |
| SOKE SMPL-X poses | tokenization, training and evaluation | dataset-specific pose folders under `datasets/` | SOKE project homepage: `https://2000zrl.github.io/soke/` |
| CSL mean/std | normalization | `datasets/CSL-Daily/mean.pt`, `datasets/CSL-Daily/std.pt` | mean: `https://drive.google.com/file/d/1NH-eVtS0nNjMjCwae-A1ii5sxj44C3bo/view?usp=sharing`; std: `https://drive.google.com/file/d/1FHHWS0GPM2s6S2PB2JHv4ufdEbzezuKW/view?usp=sharing` |
| How2Sign dataset files | optional mixed-dataset compatibility | `datasets/How2Sign/` | raw videos: `https://how2sign.github.io/`; SOKE split files: `https://drive.google.com/drive/folders/1sPhBwmiWCXLZSHtM3fpotbz3BDgoYmco?usp=sharing` |
| CSL-Daily dataset files | CSL training/eval | `datasets/CSL-Daily/` | raw videos: `http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/`; SOKE split files: `https://drive.google.com/drive/folders/17uPm6r5_DQ9CIYZonfwQLpw1XI8LeNEr?usp=drive_link` |
| Phoenix-2014T dataset files | Phoenix training/eval | `datasets/Phoenix_2014T/` | raw videos: `https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/`; SOKE split files: `https://drive.google.com/drive/folders/1Z2zjOH5wvwT7x_F6IycWAN-nh2wgJOx1?usp=sharing` |
| SOKE-AR e69 checkpoint | Table 2/3 baseline | `artifacts/checkpoints/soke_ar_e69.ckpt` | train from the upstream-compatible SOKE-AR baseline, or download from a public artifact release if provided |
| Masked-NAR e19 checkpoint | default P3/P5 runtime, HandPolish base | `artifacts/checkpoints/masked_nar_e19.ckpt` and usually `artifacts/checkpoints/p3.ckpt` | train with `configs/train/p3_csl_phoenix.yaml`, or download from a public artifact release if provided |
| Masked-NAR e49 checkpoint | Table 2 direct Masked-NAR row | `artifacts/checkpoints/masked_nar_e49.ckpt` | train/select checkpoint from Masked-NAR training, or download from a public artifact release if provided |
| HandPolish R5 caches | Table 4 post-cache benchmark | `artifacts/handpolish_cache/rep0/` ... `rep4/` or documented cache dirs | regenerate with `configs/paper/table4_poseselect_postcache.yaml`, or download from a public artifact release if provided |
| P6-B hand-token editor checkpoint | PoseSelect/GainEdit candidate scoring dependency | `artifacts/p6b/p6b.ckpt` | train with the included P6-B scripts, or download from a public artifact release if provided |
| GainEdit regressor checkpoint | Table 4/5 baseline and PoseSelect feature dependency | `artifacts/gainedit/gainedit.ckpt` | train with `scripts/train_gainedit.py`, or download from a public artifact release if provided |
| PoseSelect checkpoint | Table 4/5 deployable selector | `artifacts/poseselect/poseselect.ckpt` | train with `scripts/train_poseselect.py`, or download from a public artifact release if provided |

The upstream links above are inherited from the official SOKE README. The
paper-specific weights are new to this artifact and should be regenerated from
the experiments unless the authors publish a public artifact bundle. If such a
bundle is released, it should use the file names in the table so expected paths
remain stable.

## Dataset Download Notes

The datasets themselves are governed by their original licenses and access
procedures. A typical setup is:

1. Download raw videos from the dataset owners.
2. Download SOKE split files from the Google Drive folders listed above.
3. Download SOKE SMPL-X poses from `https://2000zrl.github.io/soke/`.
4. Place or extract everything under `datasets/How2Sign`,
   `datasets/CSL-Daily` and `datasets/Phoenix_2014T`.
5. Place `mean.pt` and `std.pt` under `datasets/CSL-Daily/`.

If you maintain a local, institutional or public mirror of prepared datasets,
package the folders as:

```text
How2Sign.tar.gz
CSL-Daily.tar.gz
Phoenix_2014T.tar.gz
```

Then run:

```bash
bash scripts/download_dataset_from_hf.sh YOUR_DATASET_REPO_OR_MIRROR datasets
```

## Optional Public Artifact Bundle Layout

The model downloader is optional and is intended for a public artifact bundle,
if one is released. The bundle may contain any subset of these files:

```text
p3.ckpt
masked_nar_e19.ckpt
masked_nar_e49.ckpt
soke_ar_e69.ckpt
poseselect.ckpt
gainedit.ckpt
p6b.ckpt
tokenizer.ckpt
mbart-h2s-csl-phoenix/
```

Then run:

```bash
export HF_TOKEN=...
export SOKENAR_MODEL_REPO=YOUR_ORG/YOUR_MODEL_REPO
python scripts/download_models.py
```

For datasets, a prepared dataset mirror may contain archives named:

```text
How2Sign.tar.gz
CSL-Daily.tar.gz
Phoenix_2014T.tar.gz
```

or the equivalent `.zip` files. Download and extract with:

```bash
export HF_TOKEN=...
export SOKENAR_DATA_REPO=YOUR_ORG/YOUR_DATASET_REPO
bash scripts/download_dataset_from_hf.sh "$SOKENAR_DATA_REPO" datasets
```

## Manual Placement

If you do not use an artifact bundle, place files manually in the expected paths
above. The configs in `configs/infer/` use `artifacts/checkpoints/p3.ckpt`.
The paper configs in `configs/paper/` use the explicit checkpoint names under
`artifacts/checkpoints/`.

To keep one checkpoint under multiple expected names, local symlinks are fine:

```bash
ln -s masked_nar_e19.ckpt artifacts/checkpoints/p3.ckpt
```

## Validation

Source-only validation:

```bash
python tests/test_release_layout.py
python -m py_compile scripts/reproduce_table2_pose.py scripts/reproduce_table3_efficiency.py
```

Artifact-aware validation:

```bash
test -f deps/tokenizer_ckpt/tokenizer.ckpt
test -d deps/mbart-h2s-csl-phoenix
test -f artifacts/checkpoints/p3.ckpt
test -f datasets/CSL-Daily/mean.pt
test -f datasets/CSL-Daily/std.pt
```

Then run a small inference:

```bash
python -m test --cfg configs/infer/p3_phoenix.yaml \
  --task t2m --nodebug --use_gpus 0 --device 0 --num_nodes 1
```
