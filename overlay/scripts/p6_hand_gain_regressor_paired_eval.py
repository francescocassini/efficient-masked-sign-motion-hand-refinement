#!/usr/bin/env python3
"""Paired PA/DTW evaluation for P6-D continuous gain regressor."""

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
from scripts.p6_hand_token_editor_paired_eval import predict_one as predict_candidate_oracle
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
    parser.add_argument("--mean-path", default="datasets/CSL-Daily/mean.pt", type=Path)
    parser.add_argument("--std-path", default="datasets/CSL-Daily/std.pt", type=Path)
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--policy", default="top_budget", choices=["threshold", "top_budget", "positive_top_budget"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
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


def select_from_scores(scores, budget, policy, threshold):
    scores = np.asarray(scores, dtype=np.float32)
    length = len(scores)
    if length == 0:
        return np.zeros((0,), dtype=bool)
    if policy == "threshold":
        return scores >= threshold
    count = length if budget >= 1.0 else max(1, int(round(length * max(budget, 0.0))))
    candidates = np.arange(length)
    if policy == "positive_top_budget":
        candidates = np.where(scores > threshold)[0]
        if len(candidates) == 0:
            return np.zeros((length,), dtype=bool)
    order = candidates[np.argsort(-scores[candidates])]
    mask = np.zeros((length,), dtype=bool)
    mask[order[:count]] = True
    return mask


def predict_regressor_one(editor, regressor, payload, pack, args, device):
    base = np.asarray(payload["feats_rst"], dtype=np.float32)
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
        l_score = l_score.detach().cpu().numpy()[0]
        r_score = r_score.detach().cpu().numpy()[0]
    l_sel = select_from_scores(l_score, args.budget, args.policy, args.threshold)
    r_sel = select_from_scores(r_score, args.budget, args.policy, args.threshold)
    refined = base.copy()
    if l_sel.any():
        l_edit = lhand.copy()
        l_edit[l_sel] = cand_lhand[l_sel]
        refined[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_edit, len(base), device)
    if r_sel.any():
        r_edit = rhand.copy()
        r_edit[r_sel] = cand_rhand[r_sel]
        refined[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_edit, len(base), device)
    return refined.astype(np.float32), {
        "lh_replaced": int(l_sel.sum()),
        "rh_replaced": int(r_sel.sum()),
        "lh_score_mean": float(l_score.mean()),
        "rh_score_mean": float(r_score.mean()),
        "lh_score_p90": float(np.quantile(l_score, 0.90)),
        "rh_score_p90": float(np.quantile(r_score, 0.90)),
    }


def run_dataset(args, dataset, editor, regressor, pack, rng, device):
    names = dataset_names(args.cache_dir, dataset, args.max_samples)
    payloads = [load_payload(args.cache_dir / f"{name}.pkl") for name in names]
    mean, std = load_mean_std(args, device)
    refs = [resample_time(np.asarray(item["feats_ref"], dtype=np.float32), len(item["feats_rst"])) for item in payloads]
    base_preds = [np.asarray(item["feats_rst"], dtype=np.float32) for item in payloads]
    reg_preds = []
    oracle_preds = []
    counts = []
    for payload in payloads:
        pred, count = predict_regressor_one(editor, regressor, payload, pack, args, device)
        oracle, _oracle_count = predict_candidate_oracle(
            editor, payload, pack, args.budget, "positive_gain_oracle", rng, device
        )
        reg_preds.append(pred)
        oracle_preds.append(oracle)
        counts.append(count)
    key = f"p6d_{args.policy}_b{args.budget:.2f}"
    reports = {
        "base": evaluate_variant(dataset, names, refs, base_preds, args.batch_size, mean, std, device),
        key: evaluate_variant(dataset, names, refs, reg_preds, args.batch_size, mean, std, device),
        f"candidate_positive_gain_oracle_b{args.budget:.2f}": evaluate_variant(
            dataset, names, refs, oracle_preds, args.batch_size, mean, std, device
        ),
    }
    return {
        "dataset": dataset,
        "samples": len(names),
        "cache_dir": str(args.cache_dir),
        "checkpoint": str(args.regressor_checkpoint),
        "budget": args.budget,
        "policy": args.policy,
        "threshold": args.threshold,
        "reports": reports,
        "replacement_stats": {
            "lh_replaced_mean": float(np.mean([item["lh_replaced"] for item in counts])),
            "rh_replaced_mean": float(np.mean([item["rh_replaced"] for item in counts])),
            "lh_score_mean": float(np.mean([item["lh_score_mean"] for item in counts])),
            "rh_score_mean": float(np.mean([item["rh_score_mean"] for item in counts])),
            "lh_score_p90": float(np.mean([item["lh_score_p90"] for item in counts])),
            "rh_score_p90": float(np.mean([item["rh_score_p90"] for item in counts])),
        },
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    pack = load_vq_pack(args, device)
    editor, editor_args = load_p6_editor(args.p6b_checkpoint, device)
    regressor, regressor_args = load_regressor(args.regressor_checkpoint, device)
    datasets = ["csl", "phoenix"] if args.dataset == "both" else [args.dataset]
    all_reports = {}
    for dataset in datasets:
        report = run_dataset(args, dataset, editor, regressor, pack, rng, device)
        report["editor_args"] = editor_args
        report["regressor_args"] = regressor_args
        all_reports[dataset] = report
        (args.output_dir / f"{dataset}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        markdown = report_to_markdown(report)
        markdown = markdown.replace("P6-A hand-token oracle ceiling", "P6-D continuous gain regressor paired eval")
        markdown = markdown.replace(
            "`all_gt_b1.00` is an oracle ceiling, not a deployable model.",
            "`candidate_positive_gain_oracle` uses GT only to choose which P6-B candidates would improve; "
            "it is an oracle ceiling, not a deployable model.",
        )
        (args.output_dir / f"{dataset}_summary.md").write_text(markdown, encoding="utf-8")
        print(markdown)
    (args.output_dir / "p6_hand_gain_regressor_paired_eval.json").write_text(
        json.dumps(all_reports, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
