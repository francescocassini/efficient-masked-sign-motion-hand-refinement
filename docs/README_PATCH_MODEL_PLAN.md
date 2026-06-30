# README, patch and model-artifact completion plan

Date: 2026-06-30

Goal: make the public delta artifact fully executable for an external reviewer
without redistributing the full modified SOKE tree.

## Target outcome

The final README should let a reviewer start from an empty directory and reach a
validated SOKE+delta checkout with clear instructions for optional full
experiment reproduction.

The expected reviewer path is:

1. Clone official SOKE.
2. Checkout the pinned upstream commit.
3. Clone this delta repository.
4. Apply the delta automatically or manually.
5. Install the SOKE-compatible environment.
6. Configure external dataset/model/checkpoint paths.
7. Run smoke tests.
8. Reproduce paper tables when full artifacts are available.

## Phase 1 - README restructure

Update `README.md` with these sections, in this order:

1. **Artifact status**
   - Explain that this is a public delta, not a full SOKE fork.
   - State the pinned SOKE base commit:
     `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00`.
   - Point to upstream SOKE: `https://github.com/2000ZRL/SOKE`.

2. **Quick start**
   - Include exact commands:
     ```bash
     git clone https://github.com/2000ZRL/SOKE.git
     git -C SOKE checkout 5cbc55d84b5a7cbf05a9cf020c468052e8d94d00
     git clone https://github.com/francescocassini/efficient-masked-sign-motion-delta.git
     bash efficient-masked-sign-motion-delta/scripts/apply_delta.sh SOKE
     cd SOKE
     python tests/test_release_layout.py
     ```

3. **Manual patch application**
   - Document the equivalent commands:
     ```bash
     git -C SOKE apply ../efficient-masked-sign-motion-delta/patches/soke-integration.patch
     cp -a ../efficient-masked-sign-motion-delta/overlay/. SOKE/
     ```
   - Explain when manual application is useful: auditing, conflict resolution,
     or adapting to a newer SOKE commit.

4. **Environment**
   - Defer to SOKE for the base environment.
   - Add the extra Python packages needed by the delta if any are missing.
   - Document Docker usage only if it is tested from a clean checkout.

5. **External artifacts**
   - Add a table with artifact name, purpose, expected path and how to obtain it.
   - Do not put private URLs in the public README.
   - Use placeholders such as `YOUR_HF_USER/YOUR_MODEL_REPO`.

6. **Configure paths**
   - Explain `.env.example`.
   - Show both shell exports and copying `.env.example` to `.env`.

7. **Smoke tests**
   - Layout test.
   - Py-compile of paper wrappers.
   - Optional import test if dependencies are available.

8. **Reproduce paper tables**
   - Table 2: `scripts/reproduce_table2_pose.py`.
   - Table 3: `scripts/reproduce_table3_efficiency.py`.
   - Table 4: `scripts/reproduce_table4_refinement.py`.
   - Table 5: `scripts/reproduce_table5_overhead.py`.
   - For each table, list required checkpoint/cache inputs.

9. **Limitations**
   - No datasets/checkpoints/caches in Git.
   - PoseSelect is post-cache, not a native generation row.
   - Full modified SOKE redistribution still requires license permission.

## Phase 2 - model-artifact documentation

Create `docs/EXTERNAL_ARTIFACTS.md` with this table:

| Artifact | Required for | Expected path after setup | Source/status |
|---|---|---|---|
| SOKE tokenizer/VQ checkpoint | all decoding/eval | `deps/tokenizer_ckpt/tokenizer.ckpt` | upstream/private artifact store |
| mBART/text assets | generation | `deps/mbart-h2s-csl-phoenix/` | upstream/private artifact store |
| SOKE-AR e69 checkpoint | Table 2/3 baseline | configurable in `configs/paper/table2_soke_ar.yaml` | external |
| Masked-NAR e49 checkpoint | Table 2 direct row | configurable in `configs/paper/table2_masked_nar_direct.yaml` | external |
| Masked-NAR e19 checkpoint | HandPolish/Table 3 | configurable in P3/P5 configs | external |
| HandPolish R5 caches | Table 4 | `artifacts/handpolish_cache/` or configured cache dir | regenerate or external |
| PoseSelect checkpoint | Table 4/5 | `artifacts/poseselect/` | train or external |
| Dataset roots | all full runs | `SOKE_DATA_ROOT` / dataset-specific env vars | original dataset channels |

Also update `overlay/.env.example` so it includes all path variables mentioned in
the README and `docs/EXTERNAL_ARTIFACTS.md`.

## Phase 3 - downloader policy

Review `overlay/scripts/download_models.py`.

Required decisions:

1. Keep it as a placeholder downloader for private Hugging Face repos, or
   replace it with clearer manual instructions.
2. If kept, document expected repo file names:
   - `p3.ckpt`
   - `tokenizer.ckpt`
   - `mbart-h2s-csl-phoenix/`
   - optional PoseSelect checkpoint name.
3. Do not hard-code non-public tokens or private artifact URLs.
4. Make failure messages explicit when an artifact is missing.

## Phase 4 - validation matrix

Run and record these checks from a clean temporary clone:

1. `scripts/apply_delta.sh` applies on pinned SOKE commit.
2. Manual `git apply` plus overlay copy works.
3. `python tests/test_release_layout.py` passes.
4. `python -m py_compile` passes for the paper wrappers.
5. No files larger than 5 MB are in the delta repo.
6. No dataset split files, checkpoints, caches or bytecode are committed.
7. README commands match the actual file layout.

## Phase 5 - professional code review

After Phases 1-4 are complete, hand the repo to a reviewer with
`HANDOFF.md` and ask for a code-review style assessment:

- correctness of patch application;
- missing entrypoints or undocumented prerequisites;
- reproducibility gaps;
- license/provenance risks;
- checkpoint/dataset path hygiene;
- whether the four paper contributions are traceable from README to code.
