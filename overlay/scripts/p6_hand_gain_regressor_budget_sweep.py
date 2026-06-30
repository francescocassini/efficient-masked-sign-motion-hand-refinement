#!/usr/bin/env python3
"""Budget sweep for P6-D/P6-F dataset-aware hand-token policies.

This script keeps the P6-D regressor fixed and evaluates several top-budget
values on the same cached P5 outputs. It is a cheap test for P6-F: if CSL and
Phoenix prefer different budgets, a dataset-aware/adaptive policy is justified.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from mGPT.models.utils.p6_hand_gain_regressor import P6HandGainRegressor
from scripts.p6_hand_gain_gate_train import predict_candidates
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_token_edit_oracle_ceiling import (
    decode_hand_tokens,
    evaluate_variant,
    load_mean_std,
    load_vq_pack,
    normalize_token_shape,
    report_to_markdown,
)
from scripts.t5_gated_residual_feature_sweep import dataset_names
from scripts.t5_residual_tokenizer_oracle import resample_time


LH_START = 30
LH_END = 75
RH_START = 75
RH_END = 120


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["csl", "phoenix", "both"])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--p6b-checkpoint", required=True, type=Path)
    parser.add_argument("--regressor-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/soke.yaml", type=Path)
    parser.add_argument("--default-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--vae-ckpt", default="deps/tokenizer_ckpt/tokenizer.ckpt", type=Path)
    parser.add_argument("--mean-path", default="/workspace/SOKE_DATA/CSL-Daily/mean.pt", type=Path)
    parser.add_argument("--std-path", default="/workspace/SOKE_DATA/CSL-Daily/std.pt", type=Path)
    parser.add_argument("--budgets", default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_regressor(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu")
    args = state.get("args", {})
    model = P6HandGainRegressor(
        hidden_features=int(args.get("hidden_features", 192)),
        layers=int(args.get("layers", 3)),
        heads=int(args.get("heads", 4)),
        dropout=float(args.get("dropout", 0.1)),
    )
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return model, args


def select_top_budget(scores, budget):
    scores = np.asarray(scores, dtype=np.float32)
    length = len(scores)
    if length == 0:
        return np.zeros((0,), dtype=bool)
    count = max(1, int(round(length * max(float(budget), 0.0))))
    order = np.argsort(-scores)
    mask = np.zeros((length,), dtype=bool)
    mask[order[:count]] = True
    return mask


def score_payload(editor, regressor, payload, device):
    token_len = min(
        len(normalize_token_shape(payload["tokens_lhand"])),
        len(normalize_token_shape(payload["tokens_rhand"])),
    )
    body, lhand, rhand, cand_lhand, cand_rhand, conf, l_model_conf, r_model_conf = predict_candidates(
        editor, payload, token_len, device
    )
    progress = np.linspace(0.0, 1.0, token_len, dtype=np.float32)
    meta = np.concatenate(
        [
            conf,
            l_model_conf[:, None],
            r_model_conf[:, None],
            (cand_lhand != lhand).astype(np.float32)[:, None],
            (cand_rhand != rhand).astype(np.float32)[:, None],
        ],
        axis=-1,
    ).astype(np.float32)
    meta[:, 0] = 0.5 * meta[:, 0] + 0.5 * progress
    with torch.no_grad():
        body_t = torch.from_numpy(body[None]).long().to(device)
        l_t = torch.from_numpy(lhand[None]).long().to(device)
        r_t = torch.from_numpy(rhand[None]).long().to(device)
        cand_l_t = torch.from_numpy(cand_lhand[None]).long().to(device)
        cand_r_t = torch.from_numpy(cand_rhand[None]).long().to(device)
        meta_t = torch.from_numpy(meta[None]).float().to(device)
        mask = torch.ones((1, token_len), dtype=torch.float32, device=device)
        text = P6HandGainRegressor.hash_text([str(payload.get("text", ""))], 4096, 96, device)
        l_score, r_score = regressor(body_t, l_t, r_t, cand_l_t, cand_r_t, meta_t, text, mask)
    return {
        "base": np.asarray(payload["feats_rst"], dtype=np.float32),
        "lhand": lhand,
        "rhand": rhand,
        "cand_lhand": cand_lhand,
        "cand_rhand": cand_rhand,
        "l_score": l_score.detach().cpu().numpy()[0],
        "r_score": r_score.detach().cpu().numpy()[0],
    }


def build_refined(scored, pack, budget, device):
    base = scored["base"]
    l_sel = select_top_budget(scored["l_score"], budget)
    r_sel = select_top_budget(scored["r_score"], budget)
    refined = base.copy()
    if l_sel.any():
        l_edit = scored["lhand"].copy()
        l_edit[l_sel] = scored["cand_lhand"][l_sel]
        refined[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_edit, len(base), device)
    if r_sel.any():
        r_edit = scored["rhand"].copy()
        r_edit[r_sel] = scored["cand_rhand"][r_sel]
        refined[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_edit, len(base), device)
    return refined.astype(np.float32), {
        "lh_replaced": int(l_sel.sum()),
        "rh_replaced": int(r_sel.sum()),
    }


def run_dataset(args, dataset, budgets, editor, regressor, pack, device):
    names = dataset_names(args.cache_dir, dataset, args.max_samples)
    payloads = [load_payload(args.cache_dir / f"{name}.pkl") for name in names]
    mean, std = load_mean_std(args, device)
    refs = [resample_time(np.asarray(item["feats_ref"], dtype=np.float32), len(item["feats_rst"])) for item in payloads]
    base_preds = [np.asarray(item["feats_rst"], dtype=np.float32) for item in payloads]
    scored = [score_payload(editor, regressor, payload, device) for payload in payloads]

    reports = {
        "base": evaluate_variant(dataset, names, refs, base_preds, args.batch_size, mean, std, device)
    }
    replacement_stats = {}
    for budget in budgets:
        key = f"p6d_budget_b{budget:.2f}"
        preds = []
        counts = []
        for item in scored:
            pred, count = build_refined(item, pack, budget, device)
            preds.append(pred)
            counts.append(count)
        reports[key] = evaluate_variant(dataset, names, refs, preds, args.batch_size, mean, std, device)
        replacement_stats[key] = {
            "lh_replaced_mean": float(np.mean([item["lh_replaced"] for item in counts])),
            "rh_replaced_mean": float(np.mean([item["rh_replaced"] for item in counts])),
        }

    base_hand = reports["base"]["pa_hand"]
    candidates = []
    for budget in budgets:
        key = f"p6d_budget_b{budget:.2f}"
        report = reports[key]
        candidates.append(
            {
                "budget": budget,
                "key": key,
                "pa_hand": report["pa_hand"],
                "delta_hand": report["pa_hand"] - base_hand,
                "speed": report["vitality"]["speed"]["mean"],
                "tstd": report["vitality"]["tstd"]["mean"],
            }
        )
    best_hand = min(candidates, key=lambda item: item["pa_hand"])
    best_hand_threshold = best_hand["pa_hand"] + 0.01
    near_best = [item for item in candidates if item["pa_hand"] <= best_hand_threshold]
    best_conservative = max(near_best, key=lambda item: (item["speed"] + item["tstd"], -item["pa_hand"]))
    return {
        "dataset": dataset,
        "samples": len(names),
        "cache_dir": str(args.cache_dir),
        "budgets": budgets,
        "reports": reports,
        "replacement_stats": replacement_stats,
        "best_hand": best_hand,
        "best_conservative": best_conservative,
    }


def write_budget_table(report):
    rows = [
        "| Budget | PA-hand ↓ | Δ vs P5 | Speed/GT ↑ | Tstd/GT ↑ | LH repl. | RH repl. |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = report["reports"]["base"]["pa_hand"]
    for budget in report["budgets"]:
        key = f"p6d_budget_b{budget:.2f}"
        values = report["reports"][key]
        stats = report["replacement_stats"][key]
        rows.append(
            f"| {budget:.2f} | {values['pa_hand']:.4f} | {values['pa_hand'] - base:+.4f} | "
            f"{values['vitality']['speed']['mean']:.3f} | {values['vitality']['tstd']['mean']:.3f} | "
            f"{stats['lh_replaced_mean']:.1f} | {stats['rh_replaced_mean']:.1f} |"
        )
    return "\n".join(
        [
            f"# P6-F budget sweep - {report['dataset']}",
            "",
            f"Samples: **{report['samples']}**",
            f"Cache: `{report['cache_dir']}`",
            "",
            f"P5 base PA-hand: **{base:.4f}**",
            "",
            *rows,
            "",
            f"Best PA-hand budget: **{report['best_hand']['budget']:.2f}** "
            f"({report['best_hand']['pa_hand']:.4f})",
            f"Best conservative budget: **{report['best_conservative']['budget']:.2f}** "
            f"({report['best_conservative']['pa_hand']:.4f}, "
            f"speed={report['best_conservative']['speed']:.3f}, "
            f"tstd={report['best_conservative']['tstd']:.3f})",
            "",
        ]
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budgets = [float(item) for item in args.budgets.split(",") if item.strip()]
    device = torch.device(args.device)
    pack = load_vq_pack(args, device)
    editor, editor_args = load_p6_editor(args.p6b_checkpoint, device)
    regressor, regressor_args = load_regressor(args.regressor_checkpoint, device)
    datasets = ["csl", "phoenix"] if args.dataset == "both" else [args.dataset]
    all_reports = {}
    for dataset in datasets:
        report = run_dataset(args, dataset, budgets, editor, regressor, pack, device)
        report["editor_args"] = editor_args
        report["regressor_args"] = regressor_args
        all_reports[dataset] = report
        (args.output_dir / f"{dataset}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        markdown = report_to_markdown(report)
        markdown = markdown.replace("P6-A hand-token oracle ceiling", "P6-F P6-D budget sweep")
        markdown += "\n" + write_budget_table(report)
        (args.output_dir / f"{dataset}_summary.md").write_text(markdown, encoding="utf-8")
        print(markdown)
    (args.output_dir / "p6_hand_gain_regressor_budget_sweep.json").write_text(
        json.dumps(all_reports, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
