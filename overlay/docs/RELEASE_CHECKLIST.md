# Release checklist

Date: 2026-06-30

## Scope

- [ ] Repository title and README refer to the AIxIA paper, not only P3/P5.
- [ ] Masked-NAR is present and documented.
- [ ] HandPolish is present and documented as training-free/body-protected.
- [ ] PoseSelect is present and documented as learned/GT-free at inference.
- [ ] GainEdit and OracleSelect are included only as baselines/diagnostics.
- [ ] Historical P3/P5/P6 names are confined to provenance or config mapping.

## Code

- [ ] Final PoseSelect scripts imported from `SOKE/`.
- [ ] Final GainEdit scripts imported from `SOKE/`.
- [ ] Final OracleSelect script imported from `SOKE/`.
- [ ] Paper-facing wrapper commands added.
- [ ] No DIMOLIKE/D3/RAG/EditKeep/MoMask code imported.
- [ ] No dead experimental variants exposed as primary commands.

## Configs

- [ ] `configs/paper/` exists.
- [ ] Table 2 configs are present and sanitized.
- [ ] Table 3 configs are present and sanitized.
- [ ] Table 4 configs are present and sanitized.
- [ ] Table 5 configs are present and sanitized.
- [ ] No public config contains `/home/cirillo`.
- [ ] No public config assumes a fixed local Docker path.
- [ ] No public config assumes a fixed CUDA device index.
- [ ] Checkpoints are referenced by documented artifact names.

## Data and artifacts

- [ ] Dataset layout documented in `docs/DATASETS.md`.
- [ ] Checkpoint layout documented in `docs/CHECKPOINTS.md`.
- [ ] Artifact hashes/manifests documented where available.
- [ ] No dataset files beyond allowed split metadata are committed.
- [ ] No checkpoints are committed.
- [ ] No generated caches are committed.
- [ ] No large binary payloads are committed unless explicitly approved.

## Paper reproducibility

- [ ] Table 2 reproduction command documented.
- [ ] Table 3 reproduction command documented.
- [ ] Table 4 reproduction command documented.
- [ ] Table 5 reproduction command documented.
- [ ] Final table values documented in `docs/PAPER_RESULTS.md`.
- [ ] Figure generation command documented.
- [ ] Known protocol caveats documented.

## Tests

- [ ] Import smoke test passes.
- [ ] Config loading test passes.
- [ ] HandPolish body-lock test passes.
- [ ] PoseSelect inference-without-GT test passes or is covered by a dry unit.
- [ ] Large/generated artifact guard test passes.
- [ ] CI runs the lightweight test suite.

## License and provenance

- [ ] `NOTICE.md` exists.
- [ ] `THIRD_PARTY_LICENSES.md` exists.
- [ ] Upstream SOKE/MotionGPT license status documented.
- [ ] `license.txt` / CC BY-NC-ND concern resolved before public release.
- [ ] Dataset redistribution rights checked.
- [ ] Checkpoint/model redistribution rights checked.
- [ ] Container image does not embed non-redistributable artifacts.

## Final GitHub steps

- [ ] Branch pushed to GitHub.
- [ ] PR opened into `main`.
- [ ] Private release candidate tagged.
- [ ] Paper repo URL finalized.
- [ ] Public/private status decided based on license gate.
