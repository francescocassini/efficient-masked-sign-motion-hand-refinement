# Paper results

Date: 2026-06-30

This file records the table values used by the AIxIA 2026 paper draft. It is a
compact index; scripts and configs remain the source of reproducibility.

## Table 2 - Main PA-JPE

Full CSL-Daily + Phoenix-2014T benchmark, 1,818 effective samples, five
replicas. Lower is better.

| Method | CSL body | CSL hand | PHX body | PHX hand |
|---|---:|---:|---:|---:|
| SOKE-AR | 9.6791 +/- 0.0000 | 1.8732 +/- 0.0000 | 8.0912 +/- 0.0000 | 1.6491 +/- 0.0000 |
| Masked-NAR direct | 9.1851 +/- 0.0502 | 1.8869 +/- 0.0056 | 6.9940 +/- 0.0429 | 1.4045 +/- 0.0047 |
| Masked-NAR HandPolish base | 8.7414 +/- 0.0226 | 1.9343 +/- 0.0019 | 6.8017 +/- 0.0307 | 1.4031 +/- 0.0013 |
| Masked-NAR + HandPolish | 8.7329 +/- 0.0223 | 1.8867 +/- 0.0013 | 6.7983 +/- 0.0306 | 1.3555 +/- 0.0037 |

Configs:

- `configs/paper/table2_soke_ar.yaml`
- `configs/paper/table2_masked_nar_direct.yaml`
- `configs/paper/table2_masked_nar_handpolish_base.yaml`
- `configs/paper/table2_handpolish.yaml`

## Table 3 - End-to-end efficiency

Phoenix-200 and CSL-200, three replicas, one RTX 4090. Includes generation,
VQ decoding, exact-DTW metrics and prediction saving.

| Dataset | Method | Time 200 | Sample/s | Speedup | J/sample | Ratio |
|---|---|---:|---:|---:|---:|---:|
| Phoenix | SOKE-AR | 71.62 +/- 0.63 s | 2.793 +/- 0.024 | 1.000x | 77.74 +/- 0.46 | 1.000 |
| Phoenix | Masked-NAR | 34.93 +/- 0.35 s | 5.727 +/- 0.058 | 2.051x | 18.37 +/- 0.10 | 0.236 |
| Phoenix | Masked-NAR + HandPolish | 35.41 +/- 0.25 s | 5.648 +/- 0.040 | 2.023x | 19.73 +/- 0.09 | 0.254 |
| CSL | SOKE-AR | 63.12 +/- 0.24 s | 3.169 +/- 0.012 | 1.000x | 72.12 +/- 0.16 | 1.000 |
| CSL | Masked-NAR | 36.15 +/- 0.08 s | 5.533 +/- 0.013 | 1.746x | 18.89 +/- 0.04 | 0.262 |
| CSL | Masked-NAR + HandPolish | 36.49 +/- 0.21 s | 5.480 +/- 0.032 | 1.730x | 19.92 +/- 0.10 | 0.276 |

Entrypoint:

- `scripts/reproduce_table3_efficiency.py`

## Table 4 - Post-cache hand-token refinement

Matched HandPolish caches, five aligned cache replicas. Lower PA-JPE is better;
speed/GT and tstd/GT closer to one indicate stronger motion vitality.

| Dataset | Method | PA-body | PA-hand | Speed/GT | Tstd/GT |
|---|---|---:|---:|---:|---:|
| CSL | HandPolish cache | 8.3125 +/- 0.0199 | 1.8072 +/- 0.0012 | 0.772 +/- 0.001 | 0.611 +/- 0.000 |
| CSL | + GainEdit | 8.2962 +/- 0.0200 | 1.6908 +/- 0.0018 | 0.734 +/- 0.001 | 0.586 +/- 0.002 |
| CSL | + PoseSelect | 8.2931 +/- 0.0200 | 1.6854 +/- 0.0017 | 0.767 +/- 0.001 | 0.634 +/- 0.001 |
| CSL | + OracleSelect | 8.2966 +/- 0.0224 | 1.5802 +/- 0.0021 | 0.751 +/- 0.001 | 0.613 +/- 0.001 |
| Phoenix | HandPolish cache | 6.7732 +/- 0.0272 | 1.3519 +/- 0.0032 | 0.546 +/- 0.002 | 0.588 +/- 0.002 |
| Phoenix | + GainEdit | 6.7662 +/- 0.0270 | 1.3127 +/- 0.0023 | 0.455 +/- 0.001 | 0.509 +/- 0.002 |
| Phoenix | + PoseSelect | 6.7631 +/- 0.0273 | 1.3127 +/- 0.0023 | 0.522 +/- 0.001 | 0.578 +/- 0.002 |
| Phoenix | + OracleSelect | 6.7650 +/- 0.0305 | 1.2498 +/- 0.0015 | 0.513 +/- 0.002 | 0.559 +/- 0.002 |

Summary files:

- `docs/results/table4_poseselect/summary.md`
- `docs/results/table4_poseselect/summary.json`

## Table 5 - Post-cache overhead

240 measured samples after 16 warmup samples. Includes final hand VQ decoding
for GainEdit and PoseSelect; excludes fresh generation, metrics, rendering and
prediction saving.

| Variant | sec/sample | sample/s | peak CUDA MB | J/sample |
|---|---:|---:|---:|---:|
| GainEdit | 0.004415 | 226.48 | 259.1 | 0.246 |
| PoseSelect | 0.084813 | 11.79 | 259.1 | 3.722 |

Summary file:

- `docs/results/table5_overhead/summary.json`

## Qualitative figure

Selected asset:

- `assets/figures/paper_qualitative_contrastive_compact_2x.png`

Regeneration wrapper:

- `scripts/make_paper_qualitative_figure.py`
