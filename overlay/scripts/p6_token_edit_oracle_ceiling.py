#!/usr/bin/env python3
"""P6-A oracle ceiling for discrete hand-token editing.

This script does not train or modify P3/P5. It reads the extended P5 cache,
re-encodes the aligned GT hands with the frozen SOKE hand VQ tokenizers, and
tests how much quality can be recovered by replacing only LH/RH tokens.

The point is deliberately diagnostic:
- if GT hand-token replacement helps a lot, P6 should learn a hand-token editor;
- if it does not, the bottleneck is likely the tokenizer/feature representation,
  not the predictor.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from mGPT.config import get_module_config, instantiate_from_config
from mGPT.metrics.t2m import TM2TMetrics
from mGPT.utils.human_models import get_coord
from mGPT.utils.load_checkpoint import load_pretrained_vae
from scripts.t5_gated_residual_feature_sweep import dataset_names
from scripts.t5_residual_tokenizer_oracle import HAND_END, HAND_START, resample_time
from scripts.t5_residual_transformer_paired_eval import (
    pad_batch,
    sequence_stats,
    summarize,
)


LH_START = 30
LH_END = 75
RH_START = 75
RH_END = 120
TOKEN_FRAMES = 4

SHAPE_PARAM = torch.tensor(
    [[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
      0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]],
    dtype=torch.float32,
)


class VQPack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vae = instantiate_from_config(cfg.model.params.motion_vae)
        self.hand_vae = instantiate_from_config(cfg.model.params.hand_vae_cfg)
        self.rhand_vae = instantiate_from_config(cfg.model.params.rhand_vae_cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["csl", "phoenix", "both"])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/soke.yaml", type=Path)
    parser.add_argument("--default-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--vae-ckpt", default="deps/tokenizer_ckpt/tokenizer.ckpt", type=Path)
    parser.add_argument("--mean-path", default="datasets/CSL-Daily/mean.pt", type=Path)
    parser.add_argument("--std-path", default="datasets/CSL-Daily/std.pt", type=Path)
    parser.add_argument("--strategies", nargs="+", default=["random", "low_conf", "high_error_oracle", "all_gt"])
    parser.add_argument("--budgets", default="0.05,0.10,0.20,0.30,0.50,1.00")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_cfg(default_config: Path, config: Path, vae_ckpt: Path):
    cfg = OmegaConf.merge(OmegaConf.load(default_config), OmegaConf.load(config))
    cfg = get_module_config(cfg)
    cfg.TRAIN.PRETRAINED_VAE = str(vae_ckpt)
    return cfg


def load_vq_pack(args, device):
    cfg = load_cfg(args.default_config, args.config, args.vae_ckpt)
    pack = VQPack(cfg)
    load_pretrained_vae(cfg, pack)
    pack.to(device).eval()
    for param in pack.parameters():
        param.requires_grad_(False)
    return pack


def to_int_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.int64)
    return np.asarray(value, dtype=np.int64)


def normalize_token_shape(tokens):
    tokens = to_int_array(tokens)
    if tokens.ndim > 1:
        tokens = tokens.reshape(-1)
    return tokens.astype(np.int64)


def pad_or_crop_np(array, target_len):
    if len(array) == target_len:
        return array
    if len(array) > target_len:
        return array[:target_len]
    if len(array) == 0:
        return np.zeros((target_len, array.shape[-1]), dtype=np.float32)
    pad = np.repeat(array[-1:], target_len - len(array), axis=0)
    return np.concatenate([array, pad], axis=0)


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


def encode_hand_tokens(vae, hand_features, device):
    tensor = torch.from_numpy(hand_features[None]).float().to(device)
    with torch.no_grad():
        tokens, _ = vae.encode(tensor)
    return tokens.detach().cpu().numpy().reshape(-1).astype(np.int64)


def decode_hand_tokens(vae, tokens, target_frames, device):
    tensor = torch.from_numpy(normalize_token_shape(tokens)).long().to(device)
    with torch.no_grad():
        decoded = vae.decode(tensor).detach().cpu().numpy()[0].astype(np.float32)
    return pad_or_crop_np(decoded, target_frames)


def replace_budget_count(length, budget):
    if budget >= 1.0:
        return length
    if budget <= 0.0:
        return 0
    return max(1, int(round(length * budget)))


def token_frame_error(base_hand, gt_hand, token_count):
    errors = np.zeros((token_count,), dtype=np.float32)
    for index in range(token_count):
        start = index * TOKEN_FRAMES
        end = min((index + 1) * TOKEN_FRAMES, len(base_hand), len(gt_hand))
        if end <= start:
            errors[index] = 0.0
        else:
            errors[index] = np.abs(base_hand[start:end] - gt_hand[start:end]).mean()
    return errors


def select_positions(strategy, budget, confidence, base_hand, gt_hand, rng):
    length = len(confidence)
    count = replace_budget_count(length, budget)
    if count <= 0 or length == 0:
        return np.zeros((length,), dtype=bool)

    if strategy == "all_gt":
        count = length
        order = np.arange(length)
    elif strategy == "random":
        order = rng.permutation(length)
    elif strategy == "low_conf":
        order = np.argsort(np.asarray(confidence, dtype=np.float32))[:count]
    elif strategy == "high_error_oracle":
        errors = token_frame_error(base_hand, gt_hand, length)
        order = np.argsort(-errors)[:count]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    mask = np.zeros((length,), dtype=bool)
    mask[order[:count]] = True
    return mask


def edit_tokens(pred_tokens, gt_tokens, selection):
    pred_tokens = normalize_token_shape(pred_tokens)
    gt_tokens = pad_or_crop_tokens(gt_tokens, len(pred_tokens))
    edited = pred_tokens.copy()
    edited[selection] = gt_tokens[selection]
    return edited


def build_variant(payload, pack, strategy, budget, rng, device):
    base = np.asarray(payload["feats_rst"], dtype=np.float32)
    ref = np.asarray(payload["feats_ref"], dtype=np.float32)
    ref_aligned = resample_time(ref, len(base)).astype(np.float32)

    l_pred = normalize_token_shape(payload["tokens_lhand"])
    r_pred = normalize_token_shape(payload["tokens_rhand"])
    l_gt = encode_hand_tokens(pack.hand_vae, ref_aligned[:, LH_START:LH_END], device)
    r_gt = encode_hand_tokens(pack.rhand_vae, ref_aligned[:, RH_START:RH_END], device)
    l_gt = pad_or_crop_tokens(l_gt, len(l_pred))
    r_gt = pad_or_crop_tokens(r_gt, len(r_pred))

    l_conf = pad_or_crop_np(np.asarray(payload["confidence_lhand"], dtype=np.float32)[:, None], len(l_pred))[:, 0]
    r_conf = pad_or_crop_np(np.asarray(payload["confidence_rhand"], dtype=np.float32)[:, None], len(r_pred))[:, 0]

    l_sel = select_positions(
        strategy,
        budget,
        l_conf,
        base[:, LH_START:LH_END],
        ref_aligned[:, LH_START:LH_END],
        rng,
    )
    r_sel = select_positions(
        strategy,
        budget,
        r_conf,
        base[:, RH_START:RH_END],
        ref_aligned[:, RH_START:RH_END],
        rng,
    )
    l_edit = edit_tokens(l_pred, l_gt, l_sel)
    r_edit = edit_tokens(r_pred, r_gt, r_sel)

    refined = base.copy()
    if l_sel.any():
        refined[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_edit, len(base), device)
    if r_sel.any():
        refined[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_edit, len(base), device)

    return refined.astype(np.float32), ref_aligned, {
        "lh_replaced": int(l_sel.sum()),
        "rh_replaced": int(r_sel.sum()),
        "lh_total": int(len(l_sel)),
        "rh_total": int(len(r_sel)),
    }


def features_to_vertices_joints(features, mean, std, device):
    feats = features.to(device).float()
    batch, time, _ = feats.shape
    feats = feats * std.view(1, 1, -1) + mean.view(1, 1, -1)
    zero_pose = torch.zeros((batch, time, 36), device=device)
    full = torch.cat([zero_pose, feats], dim=-1).reshape(batch * time, -1)
    shape = SHAPE_PARAM.to(device).repeat(batch * time, 1)
    vertices, joints = get_coord(
        root_pose=full[..., 0:3],
        body_pose=full[..., 3:66],
        lhand_pose=full[..., 66:111],
        rhand_pose=full[..., 111:156],
        jaw_pose=full[..., 156:159],
        shape=shape,
        expr=full[..., 159:169],
    )
    return vertices, joints


def metric_cfg():
    return SimpleNamespace(
        METRIC={
            "DTW_STRIDE": 1,
            "DTW_MAX_FRAMES": 0,
            "DTW_WINDOW": 0,
            "EXACT_DTW_BACKEND": "torch",
            "EXACT_DTW_CHUNK": 128,
        }
    )


def load_mean_std(args, device):
    mean = torch.load(args.mean_path, map_location=device).float().to(device)
    std = torch.load(args.std_path, map_location=device).float().to(device)
    # Same 133-dim slicing used by the existing paired evaluators.
    mean = torch.cat([mean[(3 + 3 * 11):][:-20], mean[(3 + 3 * 11):][-10:]], dim=0)
    std = torch.cat([std[(3 + 3 * 11):][:-20], std[(3 + 3 * 11):][-10:]], dim=0)
    return mean, std


def evaluate_variant(dataset, names, refs, preds, batch_size, mean, std, device):
    metric = TM2TMetrics(metric_cfg()).to(device)
    vitality = {"speed": [], "tstd": []}

    for pred, ref in zip(preds, refs):
        stats = sequence_stats(pred[:, HAND_START:HAND_END], ref[:, HAND_START:HAND_END])
        vitality["speed"].append(stats["speed_ratio"])
        vitality["tstd"].append(stats["tstd_ratio"])

    for start in range(0, len(names), batch_size):
        batch_names = names[start:start + batch_size]
        batch_refs = refs[start:start + batch_size]
        batch_preds = preds[start:start + batch_size]
        ref_pad, ref_lengths = pad_batch(batch_refs)
        rst_pad, rst_lengths = pad_batch(batch_preds)
        max_len = max(ref_pad.shape[1], rst_pad.shape[1])
        if ref_pad.shape[1] < max_len:
            ref_pad = F.pad(ref_pad, (0, 0, 0, max_len - ref_pad.shape[1]), mode="replicate")
        if rst_pad.shape[1] < max_len:
            rst_pad = F.pad(rst_pad, (0, 0, 0, max_len - rst_pad.shape[1]), mode="replicate")

        with torch.no_grad():
            vertices_ref, joints_ref = features_to_vertices_joints(ref_pad, mean, std, device)
            vertices_rst, joints_rst = features_to_vertices_joints(rst_pad, mean, std, device)
            metric.update(
                feats_rst=rst_pad.to(device),
                feats_ref=ref_pad.to(device),
                joints_rst=joints_rst,
                joints_ref=joints_ref,
                vertices_rst=vertices_rst,
                vertices_ref=vertices_ref,
                lengths=ref_lengths,
                lengths_rst=rst_lengths,
                split="test",
                src=[dataset] * len(batch_names),
                name=batch_names,
            )

    body = []
    left = []
    right = []
    for name in names:
        raw = metric.name2scores[name]
        body.append(float(raw[f"{dataset}_DTW_PA_JPE_body"]))
        left.append(float(raw[f"{dataset}_DTW_PA_JPE_lhand"]))
        right.append(float(raw[f"{dataset}_DTW_PA_JPE_rhand"]))
    hand = [(l + r) / 2.0 for l, r in zip(left, right)]
    return {
        "pa_body": float(np.mean(body)),
        "pa_hand": float(np.mean(hand)),
        "pa_lhand": float(np.mean(left)),
        "pa_rhand": float(np.mean(right)),
        "vitality": {key: summarize(values) for key, values in vitality.items()},
    }


def run_dataset(args, dataset, pack, budgets, rng, device):
    names = dataset_names(args.cache_dir, dataset, args.max_samples)
    payloads = [load_payload(args.cache_dir / f"{name}.pkl") for name in names]
    mean, std = load_mean_std(args, device)

    refs = [resample_time(np.asarray(item["feats_ref"], dtype=np.float32), len(item["feats_rst"])) for item in payloads]
    base_preds = [np.asarray(item["feats_rst"], dtype=np.float32) for item in payloads]

    reports = {
        "base": evaluate_variant(dataset, names, refs, base_preds, args.batch_size, mean, std, device)
    }
    replacement_stats = {}

    for strategy in args.strategies:
        strategy_budgets = [1.0] if strategy == "all_gt" else budgets
        for budget in strategy_budgets:
            variant_key = f"{strategy}_b{budget:.2f}"
            preds = []
            counts = []
            for payload in payloads:
                pred, _ref_aligned, count = build_variant(payload, pack, strategy, budget, rng, device)
                preds.append(pred)
                counts.append(count)
            reports[variant_key] = evaluate_variant(dataset, names, refs, preds, args.batch_size, mean, std, device)
            replacement_stats[variant_key] = {
                "lh_replaced_mean": float(np.mean([item["lh_replaced"] for item in counts])),
                "rh_replaced_mean": float(np.mean([item["rh_replaced"] for item in counts])),
                "lh_total_mean": float(np.mean([item["lh_total"] for item in counts])),
                "rh_total_mean": float(np.mean([item["rh_total"] for item in counts])),
            }

    return {
        "dataset": dataset,
        "samples": len(names),
        "cache_dir": str(args.cache_dir),
        "strategies": args.strategies,
        "budgets": budgets,
        "reports": reports,
        "replacement_stats": replacement_stats,
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
            f"# P6-A hand-token oracle ceiling - {dataset_report['dataset']}",
            "",
            f"Samples: **{dataset_report['samples']}**",
            f"Cache: `{dataset_report['cache_dir']}`",
            "",
            *rows,
            "",
            "Interpretation: variants replace only LH/RH VQ tokens; body tokens and body features stay fixed.",
            "`all_gt_b1.00` is an oracle ceiling, not a deployable model.",
            "",
        ]
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    budgets = [float(item) for item in args.budgets.split(",") if item.strip()]

    pack = load_vq_pack(args, device)
    datasets = ["csl", "phoenix"] if args.dataset == "both" else [args.dataset]
    all_reports = {}
    for dataset in datasets:
        dataset_report = run_dataset(args, dataset, pack, budgets, rng, device)
        all_reports[dataset] = dataset_report
        (args.output_dir / f"{dataset}_summary.json").write_text(
            json.dumps(dataset_report, indent=2),
            encoding="utf-8",
        )
        markdown = report_to_markdown(dataset_report)
        (args.output_dir / f"{dataset}_summary.md").write_text(markdown, encoding="utf-8")
        print(markdown)

    (args.output_dir / "p6_token_edit_oracle_ceiling.json").write_text(
        json.dumps(all_reports, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
