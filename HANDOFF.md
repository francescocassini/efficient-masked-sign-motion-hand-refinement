# Handoff for the next programmer

Date: 2026-06-30

## Current state

There are two related repositories:

- Private full working artifact:
  `https://github.com/francescocassini/efficient-masked-sign-motion`
- Public delta artifact:
  `https://github.com/francescocassini/efficient-masked-sign-motion-delta`

The public delta repository is the one intended for citation in the paper. It
does not redistribute the full modified SOKE codebase. It contains:

- `overlay/`: files added by the paper work;
- `patches/soke-integration.patch`: modifications to files that already exist
  in upstream SOKE;
- `scripts/apply_delta.sh`: helper to apply patch + overlay;
- docs and paper-result summaries.

Pinned upstream SOKE base:

- Repository: `https://github.com/2000ZRL/SOKE`
- Commit: `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00`

The delta has already been tested on a clean local clone of SOKE:

- `scripts/apply_delta.sh` applies successfully.
- `python tests/test_release_layout.py` passes after applying the delta.
- `python -m py_compile` passes for the core paper wrappers.
- The public repo contains no dataset splits, checkpoints, caches, bytecode or
  files larger than 5 MB.

Latest pushed commit before this handoff update:

- `7a2c100 docs: map delta to four paper contributions`

## Paper contribution coverage

The delta covers the four paper contributions:

1. **Masked SOKE-compatible generation**
   - `overlay/mGPT/archs/mgpt_mbart_nar*.py`
   - `overlay/configs/lm/*nar*.yaml`
   - `overlay/configs/train/p3_csl_phoenix.yaml`
   - `overlay/configs/infer/p3_*.yaml`

2. **Training-free hand polishing**
   - `overlay/mGPT/archs/mgpt_mbart_nar_p5_hand_polish*.py`
   - `overlay/configs/infer/p5_*.yaml`

3. **Deployable hand-token selection**
   - `overlay/scripts/train_poseselect.py`
   - `overlay/scripts/eval_poseselect.py`
   - `overlay/mGPT/models/utils/p6_topk_candidate_selector.py`
   - GainEdit and OracleSelect are included as baseline/diagnostic comparators.

4. **Controlled geometry, runtime and vitality evaluation**
   - `overlay/scripts/reproduce_table2_pose.py`
   - `overlay/scripts/reproduce_table3_efficiency.py`
   - `overlay/scripts/reproduce_table4_refinement.py`
   - `overlay/scripts/reproduce_table5_overhead.py`
   - `overlay/docs/PAPER_RESULTS.md`

## Next objective

Make the public delta README fully executable for an external reviewer.

The next programmer should follow:

- `docs/README_PATCH_MODEL_PLAN.md`

Primary deliverables:

1. Expand `README.md` with a full quick start.
2. Document automatic and manual patch application.
3. Add external model/checkpoint/dataset artifact instructions.
4. Update `.env.example` with all required paths.
5. Explain how to download or place model files.
6. Add smoke-test commands.
7. Add commands for reproducing Tables 2-5.
8. Re-run clean-application validation.
9. Then request professional code review.

## Important legal/provenance constraint

SOKE uses `CC BY-NC-ND 4.0`. Because of the NoDerivatives term, do not publish a
full modified SOKE tree unless the SOKE authors give permission or the license
situation changes.

The public release shape should remain a delta:

1. Users clone official SOKE.
2. Users checkout the pinned commit.
3. Users apply this repository's patch and overlay locally.

## Recommended clean validation commands

From `/tmp` or another scratch directory:

```bash
git clone https://github.com/2000ZRL/SOKE.git
git -C SOKE checkout 5cbc55d84b5a7cbf05a9cf020c468052e8d94d00
git clone https://github.com/francescocassini/efficient-masked-sign-motion-delta.git
bash efficient-masked-sign-motion-delta/scripts/apply_delta.sh SOKE
cd SOKE
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

Also check the delta repo itself:

```bash
find . -path ./.git -prune -o -type f -size +5M -print
git ls-files | grep -E '(^overlay/data/splits/|__pycache__|\.pyc$|\.ckpt$|\.pt$|\.pth$|\.safetensors$)'
```

Expected result: both commands should print nothing.

## Known docs issue to finish

`README.md` currently explains the delta concept and the basic application flow,
but it is not yet complete enough for a reviewer who needs model artifacts.

Missing or incomplete sections:

- external artifacts table;
- artifact download/place instructions;
- exact path variables;
- table-specific prerequisites;
- manual patch application details;
- fuller smoke-test and troubleshooting section.

## Suggested reviewer prompt

After the README/model-artifact pass is complete, ask a reviewer:

> Please review this public delta artifact as a code reviewer. Focus on whether
> a clean user can apply the patch to upstream SOKE, understand every external
> artifact needed, trace the four paper contributions to code, and reproduce the
> documented smoke tests. Prioritize bugs, missing files, license/provenance
> risks and documentation gaps.
