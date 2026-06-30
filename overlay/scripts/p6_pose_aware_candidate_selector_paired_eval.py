#!/usr/bin/env python3
"""Paired evaluation for P6-K pose-aware top-k selector."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from mGPT.models.utils.p6_topk_candidate_selector import P6TopKCandidateSelector
from scripts.p6_geometric_candidate_selector_eval import candidate_features, mean_abs_velocity
from scripts.p6_hand_gain_regressor_budget_sweep import select_top_budget
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_hand_topk_candidate_oracle import (
    choose_topk_oracle_tokens,
    editor_topk,
    load_regressor,
    score_top1_with_p6d,
)
from scripts.p6_pose_aware_candidate_selector_train import META_DIM, build_meta
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["csl", "phoenix", "both"])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--p6b-checkpoint", required=True, type=Path)
    parser.add_argument("--regressor-checkpoint", required=True, type=Path)
    parser.add_argument("--selector-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/soke.yaml", type=Path)
    parser.add_argument("--default-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--vae-ckpt", default="deps/tokenizer_ckpt/tokenizer.ckpt", type=Path)
    parser.add_argument("--mean-path", default="/workspace/SOKE_DATA/CSL-Daily/mean.pt", type=Path)
    parser.add_argument("--std-path", default="/workspace/SOKE_DATA/CSL-Daily/std.pt", type=Path)
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_selector(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu")
    args = state.get("args", {})
    model = P6TopKCandidateSelector(
        hidden_features=int(args.get("hidden_features", 256)),
        dropout=float(args.get("dropout", 0.1)),
        meta_dim=int(args.get("meta_dim", META_DIM)),
        max_rank=int(args.get("topk", 5)),
    )
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return model, args


def local_base_velocity(base_hand, index):
    start = int(index) * 4
    end = min((int(index) + 1) * 4, len(base_hand))
    if end <= start:
        return 0.0
    return mean_abs_velocity(base_hand[start:end])


def selector_scores(selector, vae, side, body, lhand, rhand, conf, top_tokens, top_probs, p6d_scores, index, base_hand, device):
    orig = lhand if side == 0 else rhand
    other = rhand if side == 0 else lhand
    progress = index / max(1, len(orig) - 1)
    candidates = list(top_tokens[index]) + [int(orig[index])]
    probs = list(top_probs[index]) + [1.0]
    base_vel = local_base_velocity(base_hand, index)
    rows = []
    for rank, (token, prob) in enumerate(zip(candidates, probs)):
        geom = candidate_features(vae, orig, int(token), float(prob), index, base_hand, len(base_hand), device)
        rows.append(
            {
                "body": int(body[index]),
                "orig": int(orig[index]),
                "other": int(other[index]),
                "candidate": int(token),
                "side": int(side),
                "rank": int(rank),
                "meta": build_meta(conf[index], prob, p6d_scores[index], progress, token, orig[index], rank, len(top_tokens[index]), geom, base_vel),
            }
        )
    with torch.no_grad():
        score = selector(
            torch.tensor([row["body"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["orig"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["other"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["candidate"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["side"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["rank"] for row in rows], dtype=torch.long, device=device),
            torch.tensor(np.stack([row["meta"] for row in rows]), dtype=torch.float32, device=device),
        ).detach().cpu().numpy()
    best = int(np.argmax(score))
    return rows[best]["candidate"], best, float(score[best])


def choose_selector_tokens(selector, side, selected, body, lhand, rhand, conf, top_tokens, top_probs, p6d_scores, base_hand, pack, device):
    vae = pack.hand_vae if side == 0 else pack.rhand_vae
    edited = lhand.copy() if side == 0 else rhand.copy()
    chosen_rank = []
    chosen_score = []
    for index in np.where(selected)[0]:
        token, rank, score = selector_scores(
            selector,
            vae,
            side,
            body,
            lhand,
            rhand,
            conf,
            top_tokens,
            top_probs,
            p6d_scores,
            int(index),
            base_hand,
            device,
        )
        edited[index] = token
        chosen_rank.append(rank)
        chosen_score.append(score)
    return edited, {
        "selected": int(selected.sum()),
        "mean_rank": float(np.mean(chosen_rank)) if chosen_rank else -1.0,
        "mean_score": float(np.mean(chosen_score)) if chosen_score else 0.0,
    }


def predict_one(editor, regressor, selector, payload, pack, args, device):
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

    l_selector_edit, l_stats = choose_selector_tokens(
        selector, 0, l_sel, body, lhand, rhand, conf, l_top, l_probs, l_score, base[:, LH_START:LH_END], pack, device
    )
    r_selector_edit, r_stats = choose_selector_tokens(
        selector, 1, r_sel, body, lhand, rhand, conf, r_top, r_probs, r_score, base[:, RH_START:RH_END], pack, device
    )
    selector_pred = base.copy()
    if l_sel.any():
        selector_pred[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_selector_edit, len(base), device)
    if r_sel.any():
        selector_pred[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_selector_edit, len(base), device)

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

    return top1_pred.astype(np.float32), selector_pred.astype(np.float32), oracle_pred.astype(np.float32), {
        "lh_selected": l_stats["selected"],
        "rh_selected": r_stats["selected"],
        "lh_selector_rank": l_stats["mean_rank"],
        "rh_selector_rank": r_stats["mean_rank"],
        "lh_oracle_improved": l_oracle_stats["improved"],
        "rh_oracle_improved": r_oracle_stats["improved"],
    }


def run_dataset(args, dataset, editor, regressor, selector, pack, device):
    names = dataset_names(args.cache_dir, dataset, args.max_samples)
    payloads = [load_payload(args.cache_dir / f"{name}.pkl") for name in names]
    total = len(payloads)
    progress_every = max(0, int(args.progress_every))
    start_time = time.time()
    print(
        f"[P6-K paired_eval:{dataset}] start samples={total} cache={args.cache_dir} "
        f"topk={args.topk} budget={args.budget}",
        flush=True,
    )
    mean, std = load_mean_std(args, device)
    refs = [resample_time(np.asarray(item["feats_ref"], dtype=np.float32), len(item["feats_rst"])) for item in payloads]
    base_preds = [np.asarray(item["feats_rst"], dtype=np.float32) for item in payloads]
    top1_preds = []
    selector_preds = []
    oracle_preds = []
    counts = []
    for index, payload in enumerate(payloads, start=1):
        top1, selector_pred, oracle_pred, count = predict_one(editor, regressor, selector, payload, pack, args, device)
        top1_preds.append(top1)
        selector_preds.append(selector_pred)
        oracle_preds.append(oracle_pred)
        counts.append(count)
        if progress_every and (index % progress_every == 0 or index == total):
            elapsed = time.time() - start_time
            rate = index / max(elapsed, 1e-9)
            eta = (total - index) / max(rate, 1e-9)
            print(
                f"[P6-K paired_eval:{dataset}] {index}/{total} samples "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    print(f"[P6-K paired_eval:{dataset}] evaluating metrics", flush=True)
    reports = {
        "base": evaluate_variant(dataset, names, refs, base_preds, args.batch_size, mean, std, device),
        f"p6d_top1_b{args.budget:.2f}": evaluate_variant(dataset, names, refs, top1_preds, args.batch_size, mean, std, device),
        f"p6k_pose_selector_top{args.topk}_b{args.budget:.2f}": evaluate_variant(dataset, names, refs, selector_preds, args.batch_size, mean, std, device),
        f"p6g_top{args.topk}_oracle_b{args.budget:.2f}": evaluate_variant(dataset, names, refs, oracle_preds, args.batch_size, mean, std, device),
    }
    return {
        "dataset": dataset,
        "samples": len(names),
        "cache_dir": str(args.cache_dir),
        "budget": args.budget,
        "topk": args.topk,
        "reports": reports,
        "selection_stats": {
            "lh_selected_mean": float(np.mean([item["lh_selected"] for item in counts])),
            "rh_selected_mean": float(np.mean([item["rh_selected"] for item in counts])),
            "lh_selector_rank": float(np.mean([item["lh_selector_rank"] for item in counts if item["lh_selector_rank"] >= 0])),
            "rh_selector_rank": float(np.mean([item["rh_selector_rank"] for item in counts if item["rh_selector_rank"] >= 0])),
            "lh_oracle_improved": float(np.mean([item["lh_oracle_improved"] for item in counts])),
            "rh_oracle_improved": float(np.mean([item["rh_oracle_improved"] for item in counts])),
        },
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    pack = load_vq_pack(args, device)
    editor, editor_args = load_p6_editor(args.p6b_checkpoint, device)
    regressor, regressor_args = load_regressor(args.regressor_checkpoint, device)
    selector, selector_args = load_selector(args.selector_checkpoint, device)
    datasets = ["csl", "phoenix"] if args.dataset == "both" else [args.dataset]
    all_reports = {}
    for dataset in datasets:
        report = run_dataset(args, dataset, editor, regressor, selector, pack, device)
        report["editor_args"] = editor_args
        report["regressor_args"] = regressor_args
        report["selector_args"] = selector_args
        all_reports[dataset] = report
        (args.output_dir / f"{dataset}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        markdown = report_to_markdown(report)
        markdown = markdown.replace("P6-A hand-token oracle ceiling", "P6-K pose-aware candidate selector")
        markdown = markdown.replace(
            "`all_gt_b1.00` is an oracle ceiling, not a deployable model.",
            "`p6k_pose_selector` is deployable; `p6g_topk_oracle` uses GT only as a ceiling.",
        )
        (args.output_dir / f"{dataset}_summary.md").write_text(markdown, encoding="utf-8")
        print(markdown)
        print(json.dumps(report["selection_stats"], indent=2))
    (args.output_dir / "p6_pose_aware_candidate_selector_paired_eval.json").write_text(
        json.dumps(all_reports, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
