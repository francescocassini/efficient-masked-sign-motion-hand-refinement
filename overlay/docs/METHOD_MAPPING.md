# Method-to-code mapping

Date: 2026-06-30

This document maps the paper method names to repository code.

## Public method names

| Paper name | Historical name | Summary |
|---|---|---|
| Masked-NAR | P3 | masked non-autoregressive text-to-token generator |
| HandPolish | P5 | training-free hand-only remasking with body tokens protected |
| PoseSelect | P6-K | learned post-cache top-k hand-token selector |
| GainEdit | P6-D | deployable gain-regression baseline for post-cache hand-token editing |
| OracleSelect | P6-G | non-deployable ground-truth diagnostic ceiling |

## Current repository coverage

| Paper component | Status in this repo | Main files |
|---|---|---|
| Masked-NAR | present | `mGPT/archs/mgpt_mbart_nar_p3_train_aligned.py` |
| HandPolish | present | `mGPT/archs/mgpt_mbart_nar_p5_hand_polish_aggressive.py` |
| PoseSelect | present | `scripts/train_poseselect.py`, `scripts/eval_poseselect.py` |
| GainEdit | present | `scripts/train_gainedit.py`, `scripts/eval_gainedit.py` |
| OracleSelect | present | `scripts/eval_oracleselect.py` |
| Table 2 final configs | present | `configs/paper/table2_*.yaml` |
| Table 3 efficiency scripts | present | `scripts/reproduce_table3_efficiency.py` |
| Table 4 R5 post-cache aggregation | present | `scripts/reproduce_table4_refinement.py` |
| Table 5 overhead script | present | `scripts/reproduce_table5_overhead.py` |
| paper qualitative figure | present | `scripts/make_paper_qualitative_figure.py` |

## Masked-NAR

### Paper description

Masked-NAR estimates sequence length from text and fills body, left-hand and
right-hand token streams through iterative masked decoding. It uses the frozen
SOKE tokenizer/VQ decoder and does not use ground-truth length at test time.

### Current code

| Role | File |
|---|---|
| base masked NAR | `mGPT/archs/mgpt_mbart_nar.py` |
| encoder-cache lineage | `mGPT/archs/mgpt_mbart_nar_p0_encoder_cache.py` |
| K-step lineage | `mGPT/archs/mgpt_mbart_nar_p1_step_sweep.py` |
| final train-aligned implementation | `mGPT/archs/mgpt_mbart_nar_p3_train_aligned.py` |
| final LM config | `configs/lm/mbart_h2s_csl_phoenix_nar_p3_train_aligned.yaml` |
| training config | `configs/train/p3_csl_phoenix.yaml` |
| inference configs | `configs/infer/p3_csl.yaml`, `configs/infer/p3_phoenix.yaml` |

### Paper-facing wrappers to add

| Command | Purpose |
|---|---|
| `scripts/train_masked_nar.py` | train Masked-NAR |
| `scripts/infer_masked_nar.py` | run Masked-NAR inference |
| `configs/paper/table2_masked_nar_direct.yaml` | Table 2 direct comparison operating point |
| `configs/paper/table2_masked_nar_handpolish_base.yaml` | Table 2 matched HandPolish base |

## HandPolish

### Paper description

HandPolish reuses Masked-NAR probabilities at inference. It reopens only
low-confidence left/right hand tokens and keeps the body stream fixed. It has
no trainable parameters and does not require a separate checkpoint.

### Current code

| Role | File |
|---|---|
| base hand-polish policy | `mGPT/archs/mgpt_mbart_nar_p5_hand_polish_base.py` |
| final aggressive policy | `mGPT/archs/mgpt_mbart_nar_p5_hand_polish_aggressive.py` |
| final LM config | `configs/lm/mbart_h2s_csl_phoenix_nar_p5_hand_polish_aggressive.yaml` |
| inference configs | `configs/infer/p5_csl.yaml`, `configs/infer/p5_phoenix.yaml` |

### Paper-facing wrappers to add

| Command | Purpose |
|---|---|
| `scripts/run_handpolish.py` | run HandPolish inference from Masked-NAR checkpoint |
| `configs/paper/table2_handpolish.yaml` | Table 2 HandPolish operating point |

### Required test

Add a synthetic unit test proving:

```text
body_tokens_after_handpolish == body_tokens_before_handpolish
```

## PoseSelect

### Paper description

PoseSelect is a learned post-cache selector over top-k hand-token candidates.
It is trained with offline oracle labels but uses only inference-available
features at test time. It is deployable, but not parameter-free.

### Final local source files

| Role | Source file in final `SOKE/` tree |
|---|---|
| training | `SOKE/scripts/p6_pose_aware_candidate_selector_train.py` |
| paired eval | `SOKE/scripts/p6_pose_aware_candidate_selector_paired_eval.py` |
| end-to-end runner | `SOKE/scripts/p6k_end_to_end_runner.py` |
| alignment report | `SOKE/scripts/a3_p6k_r5_alignment_report.py` |
| R5 aggregation | `SOKE/scripts/a3_p6k_r5_aggregate.py` |

### Paper-facing wrappers to add

| Command | Purpose |
|---|---|
| `scripts/build_poseselect_cache.py` | prepare matched HandPolish caches if needed |
| `scripts/train_poseselect.py` | train PoseSelect on train cache with separate val cache |
| `scripts/eval_poseselect.py` | evaluate PoseSelect on held-out/generated cache |
| `scripts/reproduce_table4_refinement.py` | reproduce post-cache Table 4 |
| `scripts/reproduce_table5_overhead.py` | reproduce overhead Table 5 |

### Required tests

Add lightweight checks that:

- the train script accepts separate train and validation cache dirs;
- the inference/eval path can run without ground-truth fields as model inputs;
- the body stream is copied unchanged from HandPolish cache to PoseSelect
  output;
- OracleSelect is clearly marked as non-deployable.

## Paper table mapping

| Paper table | Meaning | Required repo entrypoints |
|---|---|---|
| Table 1 | protocol summary | documentation only |
| Table 2 | native PA-JPE benchmark | `scripts/reproduce_table2_pose.py`, `configs/paper/table2_*.yaml` |
| Table 3 | end-to-end test-loop efficiency | `scripts/reproduce_table3_efficiency.py`, `configs/paper/table3_*.yaml` |
| Table 4 | post-cache hand-token refinement | `scripts/reproduce_table4_refinement.py`, `configs/paper/table4_poseselect_postcache.yaml` |
| Table 5 | post-cache runtime overhead | `scripts/reproduce_table5_overhead.py`, `configs/paper/table5_overhead.yaml` |

## Artifact mapping

| Artifact | Required for | Storage recommendation |
|---|---|---|
| SOKE-AR e69 checkpoint | Table 2 baseline | external artifact store |
| Masked-NAR e49 checkpoint | Table 2 direct row | external artifact store |
| Masked-NAR e19 checkpoint | HandPolish base | external artifact store |
| HandPolish caches R5 | Table 4 | external artifact store or regeneration recipe |
| PoseSelect checkpoint | Table 4/5 | external artifact store |
| VQ tokenizer checkpoint | all decoding/eval | external artifact store |
| mBART/text assets | generation | external artifact store if redistributable |

## Naming rule

Top-level docs and commands should use paper names. Historical names are allowed
only when they help reproduce a run or match an old checkpoint/config name.
