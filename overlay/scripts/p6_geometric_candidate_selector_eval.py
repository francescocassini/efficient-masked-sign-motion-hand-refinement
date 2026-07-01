#!/usr/bin/env python3
"""P6-J training-free geometric selector for P6-B top-k hand candidates.

This script is intentionally inference-only: positions are selected by P6-D,
P6-B proposes top-k hand tokens, and P6-J picks among them with a geometric
score computed from decoded candidate motion patches. No GT is used for P6-J.
P6-G oracle is kept only as a ceiling.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.p6_hand_gain_regressor_budget_sweep import select_top_budget
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_hand_topk_candidate_oracle import (
    choose_topk_oracle_tokens,
    editor_topk,
    load_regressor,
    score_top1_with_p6d,
)
from scripts.p6_token_edit_oracle_ceiling import (
    decode_hand_tokens,
    evaluate_variant,
    load_mean_std,
    load_vq_pack,
    normalize_token_shape,
    report_to_markdown,
)
from scripts.p6_topk_candidate_selector_paired_eval import topk_probs
from scripts.t5_gated_residual_feature_sweep import dataset_names
from scripts.t5_residual_tokenizer_oracle import resample_time


LH_START = 30
LH_END = 75
RH_START = 75
RH_END = 120
TOKEN_FRAMES = 4


@dataclass(frozen=True)
class Policy:
    name: str
    logprob_w: float = 1.0
    jump_w: float = 1.0
    accel_w: float = 0.0
    change_w: float = 0.0
    vitality_w: float = 0.0


POLICIES = [
    Policy("p6j0_logprob_jump", logprob_w=1.0, jump_w=1.0),
    Policy("p6j1_logprob_jump_accel", logprob_w=1.0, jump_w=1.0, accel_w=0.5),
    Policy("p6j2_smooth_conservative", logprob_w=1.0, jump_w=1.0, accel_w=0.5, change_w=0.25),
    Policy("p6j3_vitality_light", logprob_w=1.0, jump_w=1.0, accel_w=0.5, change_w=0.10, vitality_w=0.25),
    Policy("p6j4_vitality_strong", logprob_w=1.0, jump_w=1.0, accel_w=0.25, change_w=0.05, vitality_w=0.75),
    Policy("p6j5_prob_only", logprob_w=1.0, jump_w=0.0),
]


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
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def local_slice(index, length):
    start = int(index) * TOKEN_FRAMES
    end = min((int(index) + 1) * TOKEN_FRAMES, int(length))
    return start, end


def mean_abs_velocity(patch):
    if len(patch) < 2:
        return 0.0
    return float(np.abs(np.diff(patch, axis=0)).mean())


def mean_abs_accel(patch):
    if len(patch) < 3:
        return 0.0
    return float(np.abs(np.diff(patch, n=2, axis=0)).mean())


def candidate_features(vae, orig_tokens, token, prob, index, base_hand, target_frames, device):
    trial = orig_tokens.copy()
    trial[index] = int(token)
    decoded = decode_hand_tokens(vae, trial, target_frames, device)
    start, end = local_slice(index, target_frames)
    if end <= start:
        return {
            "token": int(token),
            "prob": float(prob),
            "logprob": float(np.log(max(float(prob), 1e-8))),
            "jump": 1e6,
            "accel": 1e6,
            "change": 1e6,
            "vitality": 0.0,
        }
    cand_patch = decoded[start:end]
    base_patch = base_hand[start:end]
    jump = 0.0
    if start > 0:
        jump += float(np.abs(cand_patch[0] - base_hand[start - 1]).mean())
    if end < len(base_hand):
        jump += float(np.abs(base_hand[end] - cand_patch[-1]).mean())
    jump *= 0.5
    cand_vel = mean_abs_velocity(cand_patch)
    base_vel = mean_abs_velocity(base_patch)
    vitality = cand_vel / max(base_vel, 1e-6)
    # Cap vitality to avoid rewarding implausible spikes.
    vitality = float(np.clip(vitality, 0.0, 3.0))
    return {
        "token": int(token),
        "prob": float(prob),
        "logprob": float(np.log(max(float(prob), 1e-8))),
        "jump": float(jump),
        "accel": mean_abs_accel(cand_patch),
        "change": float(np.abs(cand_patch - base_patch).mean()),
        "vitality": vitality,
    }


def score_feature(feature, policy):
    return (
        policy.logprob_w * feature["logprob"]
        - policy.jump_w * feature["jump"]
        - policy.accel_w * feature["accel"]
        - policy.change_w * feature["change"]
        + policy.vitality_w * feature["vitality"]
    )


def choose_geometric_tokens(
    vae,
    original_tokens,
    topk_tokens,
    topk_probs,
    selected,
    base_hand,
    target_frames,
    device,
    policies,
):
    edited = {policy.name: original_tokens.copy() for policy in policies}
    chosen_ranks = {policy.name: [] for policy in policies}
    for index in np.where(selected)[0]:
        features = [
            candidate_features(
                vae,
                original_tokens,
                int(token),
                float(prob),
                int(index),
                base_hand,
                target_frames,
                device,
            )
            for token, prob in zip(topk_tokens[index], topk_probs[index])
        ]
        for policy in policies:
            scores = [score_feature(feature, policy) for feature in features]
            best_rank = int(np.argmax(scores))
            edited[policy.name][index] = int(features[best_rank]["token"])
            chosen_ranks[policy.name].append(best_rank)
    stats = {
        policy.name: {
            "selected": int(selected.sum()),
            "mean_rank": float(np.mean(chosen_ranks[policy.name])) if chosen_ranks[policy.name] else -1.0,
        }
        for policy in policies
    }
    return edited, stats


def predict_one(editor, regressor, payload, pack, args, device):
    base = np.asarray(payload["feats_rst"], dtype=np.float32)
    ref = np.asarray(payload["feats_ref"], dtype=np.float32)
    ref_aligned = resample_time(ref, len(base)).astype(np.float32)
    token_len = min(
        len(normalize_token_shape(payload["tokens_lhand"])),
        len(normalize_token_shape(payload["tokens_rhand"])),
    )
    body, lhand, rhand, conf, l_top, r_top = editor_topk(editor, payload, token_len, args.topk, device)
    l_probs, r_probs = topk_probs(editor, payload, body, lhand, rhand, conf, args.topk, device)
    regressor._p6_editor = editor
    l_top1, r_top1, l_score, r_score = score_top1_with_p6d(
        regressor, body, lhand, rhand, conf, l_top, r_top, payload, device
    )
    l_sel = select_top_budget(l_score, args.budget)
    r_sel = select_top_budget(r_score, args.budget)

    top1_pred = base.copy()
    l_top1_edit = lhand.copy()
    r_top1_edit = rhand.copy()
    l_top1_edit[l_sel] = l_top1[l_sel]
    r_top1_edit[r_sel] = r_top1[r_sel]
    if l_sel.any():
        top1_pred[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_top1_edit, len(base), device)
    if r_sel.any():
        top1_pred[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_top1_edit, len(base), device)

    l_policy_edits, l_policy_stats = choose_geometric_tokens(
        pack.hand_vae,
        lhand,
        l_top,
        l_probs,
        l_sel,
        base[:, LH_START:LH_END],
        len(base),
        device,
        POLICIES,
    )
    r_policy_edits, r_policy_stats = choose_geometric_tokens(
        pack.rhand_vae,
        rhand,
        r_top,
        r_probs,
        r_sel,
        base[:, RH_START:RH_END],
        len(base),
        device,
        POLICIES,
    )
    policy_preds = {}
    for policy in POLICIES:
        pred = base.copy()
        if l_sel.any():
            pred[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_policy_edits[policy.name], len(base), device)
        if r_sel.any():
            pred[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_policy_edits[policy.name], len(base), device)
        policy_preds[policy.name] = pred.astype(np.float32)

    l_oracle_edit, l_oracle_stats = choose_topk_oracle_tokens(
        pack.hand_vae,
        lhand,
        l_top,
        l_sel,
        base[:, LH_START:LH_END],
        ref_aligned[:, LH_START:LH_END],
        len(base),
        device,
    )
    r_oracle_edit, r_oracle_stats = choose_topk_oracle_tokens(
        pack.rhand_vae,
        rhand,
        r_top,
        r_sel,
        base[:, RH_START:RH_END],
        ref_aligned[:, RH_START:RH_END],
        len(base),
        device,
    )
    oracle_pred = base.copy()
    if l_sel.any():
        oracle_pred[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_oracle_edit, len(base), device)
    if r_sel.any():
        oracle_pred[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_oracle_edit, len(base), device)

    stats = {
        "lh_selected": int(l_sel.sum()),
        "rh_selected": int(r_sel.sum()),
        "lh_oracle_improved": l_oracle_stats["improved"],
        "rh_oracle_improved": r_oracle_stats["improved"],
    }
    for policy in POLICIES:
        stats[f"lh_{policy.name}_rank"] = l_policy_stats[policy.name]["mean_rank"]
        stats[f"rh_{policy.name}_rank"] = r_policy_stats[policy.name]["mean_rank"]
    return top1_pred.astype(np.float32), policy_preds, oracle_pred.astype(np.float32), stats


def run_dataset(args, dataset, editor, regressor, pack, device):
    names = dataset_names(args.cache_dir, dataset, args.max_samples)
    payloads = [load_payload(args.cache_dir / f"{name}.pkl") for name in names]
    mean, std = load_mean_std(args, device)
    refs = [resample_time(np.asarray(item["feats_ref"], dtype=np.float32), len(item["feats_rst"])) for item in payloads]
    base_preds = [np.asarray(item["feats_rst"], dtype=np.float32) for item in payloads]
    top1_preds = []
    policy_preds = {policy.name: [] for policy in POLICIES}
    oracle_preds = []
    counts = []
    for payload in payloads:
        top1, policies, oracle, count = predict_one(editor, regressor, payload, pack, args, device)
        top1_preds.append(top1)
        for name, pred in policies.items():
            policy_preds[name].append(pred)
        oracle_preds.append(oracle)
        counts.append(count)
    reports = {
        "base": evaluate_variant(dataset, names, refs, base_preds, args.batch_size, mean, std, device),
        f"p6d_top1_b{args.budget:.2f}": evaluate_variant(dataset, names, refs, top1_preds, args.batch_size, mean, std, device),
    }
    for policy in POLICIES:
        reports[f"{policy.name}_top{args.topk}_b{args.budget:.2f}"] = evaluate_variant(
            dataset, names, refs, policy_preds[policy.name], args.batch_size, mean, std, device
        )
    reports[f"p6g_top{args.topk}_oracle_b{args.budget:.2f}"] = evaluate_variant(
        dataset, names, refs, oracle_preds, args.batch_size, mean, std, device
    )
    stats = {
        "lh_selected_mean": float(np.mean([item["lh_selected"] for item in counts])),
        "rh_selected_mean": float(np.mean([item["rh_selected"] for item in counts])),
        "lh_oracle_improved": float(np.mean([item["lh_oracle_improved"] for item in counts])),
        "rh_oracle_improved": float(np.mean([item["rh_oracle_improved"] for item in counts])),
    }
    for policy in POLICIES:
        stats[f"lh_{policy.name}_rank"] = float(np.mean([item[f"lh_{policy.name}_rank"] for item in counts if item[f"lh_{policy.name}_rank"] >= 0]))
        stats[f"rh_{policy.name}_rank"] = float(np.mean([item[f"rh_{policy.name}_rank"] for item in counts if item[f"rh_{policy.name}_rank"] >= 0]))
    return {
        "dataset": dataset,
        "samples": len(names),
        "cache_dir": str(args.cache_dir),
        "budget": args.budget,
        "topk": args.topk,
        "policies": [policy.__dict__ for policy in POLICIES],
        "reports": reports,
        "selection_stats": stats,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    pack = load_vq_pack(args, device)
    editor, editor_args = load_p6_editor(args.p6b_checkpoint, device)
    regressor, regressor_args = load_regressor(args.regressor_checkpoint, device)
    datasets = ["csl", "phoenix"] if args.dataset == "both" else [args.dataset]
    all_reports = {}
    for dataset in datasets:
        report = run_dataset(args, dataset, editor, regressor, pack, device)
        report["editor_args"] = editor_args
        report["regressor_args"] = regressor_args
        all_reports[dataset] = report
        (args.output_dir / f"{dataset}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        markdown = report_to_markdown(report)
        markdown = markdown.replace("P6-A hand-token oracle ceiling", "P6-J geometric top-k candidate selector")
        markdown = markdown.replace(
            "`all_gt_b1.00` is an oracle ceiling, not a deployable model.",
            "`p6j*` policies are training-free and deployable; `p6g_topk_oracle` uses GT only as a ceiling.",
        )
        (args.output_dir / f"{dataset}_summary.md").write_text(markdown, encoding="utf-8")
        print(markdown)
        print(json.dumps(report["selection_stats"], indent=2))
    (args.output_dir / "p6_geometric_candidate_selector_eval.json").write_text(
        json.dumps(all_reports, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
