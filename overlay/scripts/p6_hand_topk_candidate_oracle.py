#!/usr/bin/env python3
"""P6-G top-k candidate oracle for hand-token editing.

This is a diagnostic, not a deployable model. It asks whether P6-B already
places better hand tokens inside its top-k logits. Positions are selected by
the real P6-D gain regressor; for those positions, this oracle chooses the
best token among top-k candidates using the aligned GT hand error.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from mGPT.models.utils.p6_hand_gain_regressor import P6HandGainRegressor
from mGPT.models.utils.p6_hand_token_editor import P6HandTokenEditor
from scripts.p6_hand_gain_gate_train import predict_candidates
from scripts.p6_hand_gain_regressor_budget_sweep import select_top_budget
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_token_edit_oracle_ceiling import (
    decode_hand_tokens,
    evaluate_variant,
    load_mean_std,
    load_vq_pack,
    normalize_token_shape,
    pad_or_crop_np,
    pad_or_crop_tokens,
    report_to_markdown,
)
from scripts.t5_gated_residual_feature_sweep import dataset_names
from scripts.t5_residual_tokenizer_oracle import resample_time


LH_START = 30
LH_END = 75
RH_START = 75
RH_END = 120
TOKEN_FRAMES = 4


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
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=5)
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


def editor_topk(editor, payload, token_len, topk, device):
    body = pad_or_crop_tokens(payload["tokens_body"], token_len)
    lhand = pad_or_crop_tokens(payload["tokens_lhand"], token_len)
    rhand = pad_or_crop_tokens(payload["tokens_rhand"], token_len)
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
        l_t = torch.from_numpy(lhand[None]).long().to(device)
        r_t = torch.from_numpy(rhand[None]).long().to(device)
        conf_t = torch.from_numpy(conf[None]).float().to(device)
        mask = torch.ones((1, token_len), dtype=torch.float32, device=device)
        text = P6HandTokenEditor.hash_text([str(payload.get("text", ""))], 4096, 96, device)
        l_logits, r_logits = editor(body_t, l_t, r_t, conf_t, text, mask)
        l_prob = torch.softmax(l_logits, dim=-1)
        r_prob = torch.softmax(r_logits, dim=-1)
        l_top = torch.topk(l_prob, k=topk, dim=-1).indices.detach().cpu().numpy()[0].astype(np.int64)
        r_top = torch.topk(r_prob, k=topk, dim=-1).indices.detach().cpu().numpy()[0].astype(np.int64)
    return body, lhand, rhand, conf, l_top, r_top


def score_top1_with_p6d(regressor, body, lhand, rhand, conf, l_top, r_top, payload, device):
    cand_lhand = l_top[:, 0]
    cand_rhand = r_top[:, 0]
    l_model_conf = np.ones((len(lhand),), dtype=np.float32)
    r_model_conf = np.ones((len(rhand),), dtype=np.float32)
    # Reuse the original P6-D feature format. For exact confidence values,
    # predict_candidates is called because it already exposes top-1 prob.
    token_len = len(lhand)
    _, _, _, cand_lhand, cand_rhand, conf2, l_model_conf, r_model_conf = predict_candidates(
        regressor._p6_editor, payload, token_len, device
    )
    progress = np.linspace(0.0, 1.0, token_len, dtype=np.float32)
    meta = np.concatenate(
        [
            conf2,
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
    return cand_lhand, cand_rhand, l_score.detach().cpu().numpy()[0], r_score.detach().cpu().numpy()[0]


def local_error(decoded_hand, ref_hand, index):
    start = index * TOKEN_FRAMES
    end = min((index + 1) * TOKEN_FRAMES, len(decoded_hand), len(ref_hand))
    if end <= start:
        return np.inf
    return float(np.abs(decoded_hand[start:end] - ref_hand[start:end]).mean())


def choose_topk_oracle_tokens(vae, original_tokens, topk_tokens, selected, base_hand, ref_hand, target_frames, device):
    edited = original_tokens.copy()
    chosen_rank = []
    improved = 0
    for index in np.where(selected)[0]:
        base_err = local_error(base_hand, ref_hand, index)
        best_token = original_tokens[index]
        best_err = base_err
        best_rank = -1
        for rank, token in enumerate(topk_tokens[index]):
            trial = original_tokens.copy()
            trial[index] = token
            decoded = decode_hand_tokens(vae, trial, target_frames, device)
            err = local_error(decoded, ref_hand, index)
            if err < best_err:
                best_err = err
                best_token = token
                best_rank = rank
        edited[index] = best_token
        if best_rank >= 0:
            improved += 1
        chosen_rank.append(best_rank)
    return edited, {
        "improved": int(improved),
        "selected": int(selected.sum()),
        "mean_rank": float(np.mean([rank for rank in chosen_rank if rank >= 0])) if improved else -1.0,
    }


def predict_one(editor, regressor, payload, pack, args, device):
    base = np.asarray(payload["feats_rst"], dtype=np.float32)
    ref = np.asarray(payload["feats_ref"], dtype=np.float32)
    ref_aligned = resample_time(ref, len(base)).astype(np.float32)
    token_len = min(
        len(normalize_token_shape(payload["tokens_lhand"])),
        len(normalize_token_shape(payload["tokens_rhand"])),
    )
    body, lhand, rhand, conf, l_top, r_top = editor_topk(editor, payload, token_len, args.topk, device)
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

    l_topk_edit, l_stats = choose_topk_oracle_tokens(
        pack.hand_vae,
        lhand,
        l_top,
        l_sel,
        base[:, LH_START:LH_END],
        ref_aligned[:, LH_START:LH_END],
        len(base),
        device,
    )
    r_topk_edit, r_stats = choose_topk_oracle_tokens(
        pack.rhand_vae,
        rhand,
        r_top,
        r_sel,
        base[:, RH_START:RH_END],
        ref_aligned[:, RH_START:RH_END],
        len(base),
        device,
    )
    topk_pred = base.copy()
    if l_sel.any():
        topk_pred[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_topk_edit, len(base), device)
    if r_sel.any():
        topk_pred[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_topk_edit, len(base), device)
    return top1_pred.astype(np.float32), topk_pred.astype(np.float32), {
        "lh_selected": int(l_sel.sum()),
        "rh_selected": int(r_sel.sum()),
        "lh_improved": l_stats["improved"],
        "rh_improved": r_stats["improved"],
        "lh_mean_rank": l_stats["mean_rank"],
        "rh_mean_rank": r_stats["mean_rank"],
    }


def run_dataset(args, dataset, editor, regressor, pack, device):
    names = dataset_names(args.cache_dir, dataset, args.max_samples)
    payloads = [load_payload(args.cache_dir / f"{name}.pkl") for name in names]
    mean, std = load_mean_std(args, device)
    refs = [resample_time(np.asarray(item["feats_ref"], dtype=np.float32), len(item["feats_rst"])) for item in payloads]
    base_preds = [np.asarray(item["feats_rst"], dtype=np.float32) for item in payloads]
    top1_preds = []
    topk_preds = []
    counts = []
    for payload in payloads:
        top1, topk, count = predict_one(editor, regressor, payload, pack, args, device)
        top1_preds.append(top1)
        topk_preds.append(topk)
        counts.append(count)
    reports = {
        "base": evaluate_variant(dataset, names, refs, base_preds, args.batch_size, mean, std, device),
        f"p6d_top1_b{args.budget:.2f}": evaluate_variant(dataset, names, refs, top1_preds, args.batch_size, mean, std, device),
        f"p6g_top{args.topk}_oracle_b{args.budget:.2f}": evaluate_variant(dataset, names, refs, topk_preds, args.batch_size, mean, std, device),
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
            "lh_improved_mean": float(np.mean([item["lh_improved"] for item in counts])),
            "rh_improved_mean": float(np.mean([item["rh_improved"] for item in counts])),
            "lh_mean_rank": float(np.mean([item["lh_mean_rank"] for item in counts if item["lh_mean_rank"] >= 0])),
            "rh_mean_rank": float(np.mean([item["rh_mean_rank"] for item in counts if item["rh_mean_rank"] >= 0])),
        },
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
        markdown = markdown.replace("P6-A hand-token oracle ceiling", "P6-G top-k candidate oracle")
        markdown = markdown.replace(
            "`all_gt_b1.00` is an oracle ceiling, not a deployable model.",
            "`p6g_topk_oracle` chooses among P6-B top-k candidates using GT local hand error; "
            "it is an oracle diagnostic, not deployable.",
        )
        (args.output_dir / f"{dataset}_summary.md").write_text(markdown, encoding="utf-8")
        print(markdown)
        print(json.dumps(report["selection_stats"], indent=2))
    (args.output_dir / "p6_hand_topk_candidate_oracle.json").write_text(
        json.dumps(all_reports, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
