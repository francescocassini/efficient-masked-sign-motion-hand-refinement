#!/usr/bin/env python3
"""Aggregate P6-K metrics across five saved cache replicas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = {
    "p5_cache": "base",
    "p6d_clean": "p6d_top1_b0.20",
    "p6k_clean": "p6k_pose_selector_top5_b0.20",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(rep_reports):
    rows = []
    for rep_index, report in enumerate(rep_reports):
        for dataset in ["csl", "phoenix"]:
            for label, key in VARIANTS.items():
                item = report[dataset]["reports"][key]
                rows.append(
                    {
                        "replica": rep_index,
                        "dataset": dataset,
                        "variant": label,
                        "pa_body": float(item["pa_body"]),
                        "pa_hand": float(item["pa_hand"]),
                        "pa_lhand": float(item["pa_lhand"]),
                        "pa_rhand": float(item["pa_rhand"]),
                        "speed_gt": float(item["vitality"]["speed"]["mean"]),
                        "tstd_gt": float(item["vitality"]["tstd"]["mean"]),
                    }
                )
    return rows


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    ci = float(1.96 * values.std(ddof=0) / np.sqrt(len(values)))
    return mean, ci


def aggregate(rows):
    out = {}
    for dataset in ["csl", "phoenix"]:
        out[dataset] = {}
        for variant in VARIANTS:
            group = [row for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            out[dataset][variant] = {}
            for metric in ["pa_body", "pa_hand", "pa_lhand", "pa_rhand", "speed_gt", "tstd_gt"]:
                mean, ci = mean_ci([row[metric] for row in group])
                out[dataset][variant][metric] = {"mean": mean, "ci95": ci}
    return out


def fmt(item, metric, digits=4):
    return f"{item[metric]['mean']:.{digits}f} +/- {item[metric]['ci95']:.{digits}f}"


def markdown(summary):
    lines = [
        "# P6-K Full R5 Aggregate",
        "",
        f"Replicas: **{summary['replicas']}**",
        "",
        "| Dataset | Variant | PA-body | PA-hand | speed/GT | tstd/GT |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in ["csl", "phoenix"]:
        for variant in ["p5_cache", "p6d_clean", "p6k_clean"]:
            item = summary["aggregate"][dataset][variant]
            lines.append(
                f"| {dataset} | {variant} | {fmt(item, 'pa_body')} | {fmt(item, 'pa_hand')} | "
                f"{fmt(item, 'speed_gt', 3)} | {fmt(item, 'tstd_gt', 3)} |"
            )
    lines.extend(
        [
            "",
            "## Claim Rule",
            "",
            "- This is a five-replica P6-K evaluation over separately saved final-fair cache replicas.",
            "- P6-K remains a post-P5-cache method; it should be compared as an extension over the matched P5-cache rows.",
            "- Insert into the SOKE/P3/P5 R5 table only with the explicit note that the P6-K base is the saved P5-cache wrapper path.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-report", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rep_reports = [load_json(path) for path in args.rep_report]
    rows = collect(rep_reports)
    summary = {
        "replicas": len(rep_reports),
        "rep_reports": [str(path) for path in args.rep_report],
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
