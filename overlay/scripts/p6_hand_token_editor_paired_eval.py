#!/usr/bin/env python3
"""Paired PA/DTW evaluation for P6-B learned hand-token editor."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from mGPT.models.utils.p6_hand_token_editor import P6HandTokenEditor
from scripts.p6_token_edit_oracle_ceiling import (
    build_variant,
    decode_hand_tokens,
    edit_tokens,
    evaluate_variant,
    load_mean_std,
    load_vq_pack,
    normalize_token_shape,
    pad_or_crop_np,
    select_positions,
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
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/soke.yaml", type=Path)
    parser.add_argument("--default-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--vae-ckpt", default="deps/tokenizer_ckpt/tokenizer.ckpt", type=Path)
    parser.add_argument("--mean-path", default="/workspace/SOKE_DATA/CSL-Daily/mean.pt", type=Path)
    parser.add_argument("--std-path", default="/workspace/SOKE_DATA/CSL-Daily/std.pt", type=Path)
    parser.add_argument("--budget", type=float, default=0.30)
    parser.add_argument(
        "--selection",
        default="low_conf",
        choices=["low_conf", "model_conf", "model_change", "positive_gain_oracle"],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu")
    args = state.get("args", {})
    model = P6HandTokenEditor(
        hidden_features=int(args.get("hidden_features", 256)),
        layers=int(args.get("layers", 4)),
        heads=int(args.get("heads", 4)),
        dropout=float(args.get("dropout", 0.1)),
    )
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return model, args


def pad_or_crop_tokens(tokens, target_len):
    tokens = normalize_token_shape(tokens)
    if len(tokens) == target_len:
        return tokens
    if len(tokens) > target_len:
        return tokens[:target_len]
    if len(tokens) == 0:
        return np.zeros((target_len,), dtype=np.int64)
    pad = np.repeat(tokens[-1:], target_len - len(tokens), axis=0)
    return np.concatenate([tokens, pad], axis=0).astype(np.int64)


def token_gain(base_hand, candidate_hand, ref_hand, token_count):
    gains = np.zeros((token_count,), dtype=np.float32)
    for index in range(token_count):
        start = index * 4
        end = min((index + 1) * 4, len(base_hand), len(candidate_hand), len(ref_hand))
        if end <= start:
            continue
        base_err = np.abs(base_hand[start:end] - ref_hand[start:end]).mean()
        cand_err = np.abs(candidate_hand[start:end] - ref_hand[start:end]).mean()
        gains[index] = base_err - cand_err
    return gains


def select_top_positive_gains(gains, budget):
    length = len(gains)
    if length == 0:
        return np.zeros((0,), dtype=bool)
    count = length if budget >= 1.0 else max(1, int(round(length * max(budget, 0.0))))
    positive = np.where(gains > 0.0)[0]
    if len(positive) == 0:
        return np.zeros((length,), dtype=bool)
    order = positive[np.argsort(-gains[positive])]
    mask = np.zeros((length,), dtype=bool)
    mask[order[:count]] = True
    return mask


def predict_one(model, payload, pack, budget, selection, rng, device):
    base = np.asarray(payload["feats_rst"], dtype=np.float32)
    l_pred = normalize_token_shape(payload["tokens_lhand"])
    r_pred = normalize_token_shape(payload["tokens_rhand"])
    token_len = min(len(l_pred), len(r_pred))
    l_pred = pad_or_crop_tokens(l_pred, token_len)
    r_pred = pad_or_crop_tokens(r_pred, token_len)
    body = pad_or_crop_tokens(payload["tokens_body"], token_len)
    conf = np.concatenate(
        [
            pad_or_crop_np(np.asarray(payload["confidence_body"], dtype=np.float32)[:, None], token_len),
            pad_or_crop_np(np.asarray(payload["confidence_lhand"], dtype=np.float32)[:, None], token_len),
            pad_or_crop_np(np.asarray(payload["confidence_rhand"], dtype=np.float32)[:, None], token_len),
        ],
        axis=-1,
    ).astype(np.float32)

    with torch.no_grad():
        body_t = torch.from_numpy(body[None]).long().to(device)
        l_t = torch.from_numpy(l_pred[None]).long().to(device)
        r_t = torch.from_numpy(r_pred[None]).long().to(device)
        conf_t = torch.from_numpy(conf[None]).float().to(device)
        mask = torch.ones((1, token_len), dtype=torch.float32, device=device)
        text = P6HandTokenEditor.hash_text([str(payload.get("text", ""))], 4096, 96, device)
        l_logits, r_logits = model(body_t, l_t, r_t, conf_t, text, mask)
        l_prob = torch.softmax(l_logits, dim=-1)
        r_prob = torch.softmax(r_logits, dim=-1)
        l_new = l_prob.argmax(dim=-1).detach().cpu().numpy()[0].astype(np.int64)
        r_new = r_prob.argmax(dim=-1).detach().cpu().numpy()[0].astype(np.int64)
        l_model_conf = l_prob.max(dim=-1).values.detach().cpu().numpy()[0].astype(np.float32)
        r_model_conf = r_prob.max(dim=-1).values.detach().cpu().numpy()[0].astype(np.float32)

    if selection == "low_conf":
        l_sel = select_positions("low_conf", budget, conf[:, 1], base[:, LH_START:LH_END], base[:, LH_START:LH_END], rng)
        r_sel = select_positions("low_conf", budget, conf[:, 2], base[:, RH_START:RH_END], base[:, RH_START:RH_END], rng)
    elif selection == "model_conf":
        l_sel = select_positions("low_conf", budget, l_model_conf, base[:, LH_START:LH_END], base[:, LH_START:LH_END], rng)
        r_sel = select_positions("low_conf", budget, r_model_conf, base[:, RH_START:RH_END], base[:, RH_START:RH_END], rng)
    else:
        if selection == "model_change":
            l_change = (l_new != l_pred).astype(np.float32)
            r_change = (r_new != r_pred).astype(np.float32)
            l_sel = select_positions("high_error_oracle", budget, l_change[:, None], np.zeros((token_len, 1)), l_change[:, None], rng)
            r_sel = select_positions("high_error_oracle", budget, r_change[:, None], np.zeros((token_len, 1)), r_change[:, None], rng)
        else:
            ref = np.asarray(payload["feats_ref"], dtype=np.float32)
            ref_aligned = resample_time(ref, len(base)).astype(np.float32)
            l_candidate = decode_hand_tokens(pack.hand_vae, l_new, len(base), device)
            r_candidate = decode_hand_tokens(pack.rhand_vae, r_new, len(base), device)
            l_gains = token_gain(
                base[:, LH_START:LH_END],
                l_candidate,
                ref_aligned[:, LH_START:LH_END],
                token_len,
            )
            r_gains = token_gain(
                base[:, RH_START:RH_END],
                r_candidate,
                ref_aligned[:, RH_START:RH_END],
                token_len,
            )
            l_sel = select_top_positive_gains(l_gains, budget)
            r_sel = select_top_positive_gains(r_gains, budget)

    l_edit = edit_tokens(l_pred, l_new, l_sel)
    r_edit = edit_tokens(r_pred, r_new, r_sel)
    refined = base.copy()
    if l_sel.any():
        refined[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_edit, len(base), device)
    if r_sel.any():
        refined[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_edit, len(base), device)
    return refined.astype(np.float32), {
        "lh_replaced": int(l_sel.sum()),
        "rh_replaced": int(r_sel.sum()),
        "lh_changed_pred": int((l_new != l_pred).sum()),
        "rh_changed_pred": int((r_new != r_pred).sum()),
    }


def run_dataset(args, dataset, model, pack, rng, device):
    names = dataset_names(args.cache_dir, dataset, args.max_samples)
    payloads = [load_payload(args.cache_dir / f"{name}.pkl") for name in names]
    mean, std = load_mean_std(args, device)
    refs = [resample_time(np.asarray(item["feats_ref"], dtype=np.float32), len(item["feats_rst"])) for item in payloads]
    base_preds = [np.asarray(item["feats_rst"], dtype=np.float32) for item in payloads]
    p6_preds = []
    counts = []
    oracle_preds = []
    for payload in payloads:
        pred, count = predict_one(model, payload, pack, args.budget, args.selection, rng, device)
        oracle, _ref, _oracle_count = build_variant(payload, pack, "low_conf", args.budget, rng, device)
        p6_preds.append(pred)
        oracle_preds.append(oracle)
        counts.append(count)

    reports = {
        "base": evaluate_variant(dataset, names, refs, base_preds, args.batch_size, mean, std, device),
        f"p6_{args.selection}_b{args.budget:.2f}": evaluate_variant(dataset, names, refs, p6_preds, args.batch_size, mean, std, device),
        f"oracle_low_conf_b{args.budget:.2f}": evaluate_variant(dataset, names, refs, oracle_preds, args.batch_size, mean, std, device),
    }
    return {
        "dataset": dataset,
        "samples": len(names),
        "cache_dir": str(args.cache_dir),
        "checkpoint": str(args.checkpoint),
        "budget": args.budget,
        "selection": args.selection,
        "reports": reports,
        "replacement_stats": {
            "lh_replaced_mean": float(np.mean([item["lh_replaced"] for item in counts])),
            "rh_replaced_mean": float(np.mean([item["rh_replaced"] for item in counts])),
            "lh_changed_pred_mean": float(np.mean([item["lh_changed_pred"] for item in counts])),
            "rh_changed_pred_mean": float(np.mean([item["rh_changed_pred"] for item in counts])),
        },
    }


def report_to_markdown(dataset_report):
    rows = [
        "| Variant | PA-body ↓ | PA-hand ↓ | PA-LH ↓ | PA-RH ↓ | speed/GT | tstd/GT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    base_hand = dataset_report["reports"]["base"]["pa_hand"]
    for key, values in dataset_report["reports"].items():
        delta = values["pa_hand"] - base_hand
        label = key if key == "base" else f"{key} ({delta:+.4f} hand)"
        rows.append(
            f"| {label} | {values['pa_body']:.4f} | {values['pa_hand']:.4f} | "
            f"{values['pa_lhand']:.4f} | {values['pa_rhand']:.4f} | "
            f"{values['vitality']['speed']['mean']:.3f} | {values['vitality']['tstd']['mean']:.3f} |"
        )
    return "\n".join(
        [
            f"# P6-B learned hand-token editor paired eval - {dataset_report['dataset']}",
            "",
            f"Samples: **{dataset_report['samples']}**",
            f"Cache: `{dataset_report['cache_dir']}`",
            f"Checkpoint: `{dataset_report['checkpoint']}`",
            "",
            *rows,
            "",
            "Interpretation: P6-B predicts replacement LH/RH VQ tokens without GT tokens in input.",
            "The oracle row is included only as a ceiling reference.",
            "",
        ]
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    pack = load_vq_pack(args, device)
    model, model_args = load_model(args.checkpoint, device)
    datasets = ["csl", "phoenix"] if args.dataset == "both" else [args.dataset]
    all_reports = {}
    for dataset in datasets:
        report = run_dataset(args, dataset, model, pack, rng, device)
        report["model_args"] = model_args
        all_reports[dataset] = report
        (args.output_dir / f"{dataset}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        markdown = report_to_markdown(report)
        (args.output_dir / f"{dataset}_summary.md").write_text(markdown, encoding="utf-8")
        print(markdown)
    (args.output_dir / "p6_hand_token_editor_paired_eval.json").write_text(
        json.dumps(all_reports, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
