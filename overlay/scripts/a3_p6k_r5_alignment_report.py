#!/usr/bin/env python3
"""Create a compact A3 alignment report for P6-K against final-fair P5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf


FAIR_FIELDS = [
    "SEED_VALUE",
    "EVAL.SPLIT",
    "EVAL.BATCH_SIZE",
    "EVAL.MAX_SAMPLES",
    "TEST.SPLIT",
    "TEST.BATCH_SIZE",
    "TEST.MAX_SAMPLES",
    "DATASET.target",
    "DATASET.H2S.DATASET_NAME",
    "DATASET.H2S.ROOT",
    "DATASET.H2S.CSL_ROOT",
    "DATASET.H2S.PHOENIX_ROOT",
    "DATASET.H2S.MEAN_PATH",
    "DATASET.H2S.STD_PATH",
    "DATASET.H2S.MAX_MOTION_LEN",
    "DATASET.H2S.MIN_MOTION_LEN",
    "DATASET.H2S.MAX_TEXT_LEN",
    "DATASET.H2S.PICK_ONE_TEXT",
    "DATASET.H2S.FRAME_RATE",
    "DATASET.H2S.UNIT_LEN",
    "DATASET.H2S.STD_TEXT",
    "DATASET.CODE_PATH",
    "METRIC.TYPE",
    "METRIC.DTW_STRIDE",
    "METRIC.DTW_MAX_FRAMES",
    "METRIC.DTW_WINDOW",
    "METRIC.EXACT_DTW_BACKEND",
    "METRIC.EXACT_DTW_CHUNK",
    "model.params.motion_vae",
    "model.params.hand_vae_cfg",
    "model.params.rhand_vae_cfg",
]


def select(cfg, key):
    value = cfg
    for part in key.split("."):
        if part not in value:
            return "<MISSING>"
        value = value[part]
    return value


def load_cfg(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def read_csv_last(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return rows[-1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row, key):
    value = row.get(key)
    return None if value in {None, ""} else float(value)


def variant(report, dataset, key):
    return report[dataset]["reports"][key]


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-config", default="configs/final_fair/test_final_fair_p5_e19_cslphx_full_r5.yaml", type=Path)
    parser.add_argument("--cache-config", default="configs/final_fair/test_final_fair_p6k_cache_e19_cslphx_full_r1.yaml", type=Path)
    parser.add_argument("--reference-csv", default="results/mgpt/FINAL_FAIR_P5_E19_CSLPHX_FULL_R5/csv_logs/metrics.csv", type=Path)
    parser.add_argument("--p6-report", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_cfg = load_cfg(args.reference_config)
    cache_cfg = load_cfg(args.cache_config)
    mismatches = []
    for field in FAIR_FIELDS:
        ref = select(ref_cfg, field)
        actual = select(cache_cfg, field)
        if ref != actual:
            mismatches.append({"field": field, "reference": ref, "actual": actual})

    cache_files = sorted(args.cache_dir.glob("*.pkl"))
    ref_row = read_csv_last(args.reference_csv)
    p6_report = read_json(args.p6_report)

    summary = {
        "verdict": "PASS_R1_ALIGNED_NOT_FULL_R5" if not mismatches and len(cache_files) == 1818 else "CHECK_REQUIRED",
        "scope": "R1 cache regenerated from final_fair fields; P6-K evaluated on that cache with clean P6 checkpoints.",
        "cache_samples": len(cache_files),
        "fair_field_mismatches": mismatches,
        "reference_p5_e19_r5_last_row": {
            "csl_pa_body": metric(ref_row, "Metrics/csl_DTW_PA_JPE_body"),
            "csl_pa_hand": metric(ref_row, "Metrics/csl_DTW_PA_JPE_hand"),
            "phoenix_pa_body": metric(ref_row, "Metrics/phoenix_DTW_PA_JPE_body"),
            "phoenix_pa_hand": metric(ref_row, "Metrics/phoenix_DTW_PA_JPE_hand"),
        },
        "a3_cache_p5_r1_metrics": {
            "csl_pa_body": variant(p6_report, "csl", "base")["pa_body"],
            "csl_pa_hand": variant(p6_report, "csl", "base")["pa_hand"],
            "phoenix_pa_body": variant(p6_report, "phoenix", "base")["pa_body"],
            "phoenix_pa_hand": variant(p6_report, "phoenix", "base")["pa_hand"],
        },
        "a3_p6_eval": {
            "csl_p5_cache": variant(p6_report, "csl", "base"),
            "csl_p6d": variant(p6_report, "csl", "p6d_top1_b0.20"),
            "csl_p6k": variant(p6_report, "csl", "p6k_pose_selector_top5_b0.20"),
            "phoenix_p5_cache": variant(p6_report, "phoenix", "base"),
            "phoenix_p6d": variant(p6_report, "phoenix", "p6d_top1_b0.20"),
            "phoenix_p6k": variant(p6_report, "phoenix", "p6k_pose_selector_top5_b0.20"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# A3 P6-K R5 Alignment Report",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "Scope: regenerated an R1 cache from a `configs/final_fair` config that matches the P5 e19 final-fair fields except the expected wrapper/model fields needed to save P6 inputs.",
        "The cache wrapper does not update Lightning metrics; A3 cache metrics below are computed by the same paired evaluator used for P6-D/P6-K.",
        "",
        f"Cache samples: **{len(cache_files)}**",
        "",
        "## Fair-Field Audit",
        "",
    ]
    if mismatches:
        md.extend(["| Field | Reference | Actual |", "|---|---|---|"])
        for item in mismatches:
            md.append(f"| `{item['field']}` | `{item['reference']}` | `{item['actual']}` |")
    else:
        md.append("All audited fair-critical fields match.")
    md.extend([
        "",
        "## P5 Reference vs A3 Cache",
        "",
        "| Source | CSL PA-body | CSL PA-hand | Phoenix PA-body | Phoenix PA-hand |",
        "|---|---:|---:|---:|---:|",
    ])
    r = summary["reference_p5_e19_r5_last_row"]
    c = summary["a3_cache_p5_r1_metrics"]
    md.append(f"| P5 e19 R5 last replica | {fmt(r['csl_pa_body'])} | {fmt(r['csl_pa_hand'])} | {fmt(r['phoenix_pa_body'])} | {fmt(r['phoenix_pa_hand'])} |")
    md.append(f"| A3 regenerated P5-cache R1 | {fmt(c['csl_pa_body'])} | {fmt(c['csl_pa_hand'])} | {fmt(c['phoenix_pa_body'])} | {fmt(c['phoenix_pa_hand'])} |")
    md.extend([
        "",
        "## A3 P6-K Results",
        "",
        "| Dataset | Variant | PA-body | PA-hand | speed/GT | tstd/GT |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for dataset in ["csl", "phoenix"]:
        for label, key in [("P5-cache", "base"), ("P6-D clean", "p6d_top1_b0.20"), ("P6-K clean", "p6k_pose_selector_top5_b0.20")]:
            item = variant(p6_report, dataset, key)
            md.append(
                f"| {dataset} | {label} | {fmt(item['pa_body'])} | {fmt(item['pa_hand'])} | "
                f"{fmt(item['vitality']['speed']['mean'], 3)} | {fmt(item['vitality']['tstd']['mean'], 3)} |"
            )
    md.extend([
        "",
        "## Claim Rule",
        "",
        "- This supports an R1-aligned P6-K comparison on a final-fair regenerated cache.",
        "- The A3 cache base differs numerically from the existing Lightning R5 P5 table, so P6-K should not yet be inserted into that exact table as an R5 row.",
        "- It is not yet a five-cache stochastic R5 P6-K result.",
        "- If the paper needs P6-K inside the exact R5 table, run five separately saved cache replicas and evaluate P6-K on each.",
    ])
    (args.output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
