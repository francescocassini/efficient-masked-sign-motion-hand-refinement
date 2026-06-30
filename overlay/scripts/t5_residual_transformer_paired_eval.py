#!/usr/bin/env python3
"""Paired PA/DTW evaluation for the T5-1 residual token transformer.

This is the first non-oracle T5 evaluation: the residual tokens are predicted
from fixed P5 hand features plus text-lite conditioning. GT residuals are used
only as the reference target for metrics, never as input to the model.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mGPT.metrics.t2m import TM2TMetrics
from mGPT.models.utils.t5_residual_transformer import T5ResidualTokenTransformer
from mGPT.models.utils.t5_residual_vq import T5ResidualVQVAE
from mGPT.utils.human_models import get_coord
from scripts.t5_residual_tokenizer_oracle import HAND_END, HAND_START, resample_time


SHAPE_PARAM = torch.tensor(
    [[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
      0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]],
    dtype=torch.float32,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["csl", "phoenix"])
    parser.add_argument("--p5-dir", required=True, type=Path)
    parser.add_argument("--transformer-ckpt", required=True, type=Path)
    parser.add_argument("--tokenizer-ckpt", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mean-path", default="/workspace/SOKE_DATA/CSL-Daily/mean.pt", type=Path)
    parser.add_argument("--std-path", default="/workspace/SOKE_DATA/CSL-Daily/std.pt", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_tokenizer(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu")
    args = state.get("args", {})
    model = T5ResidualVQVAE(
        hidden_features=int(args.get("hidden_features", 64)),
        codebook_size=int(args.get("codebook_size", 32)),
        code_dim=int(args.get("code_dim", 64)),
        max_residual=float(args.get("max_residual", 1.0)),
    )
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return model


def load_transformer(checkpoint, tokenizer, device):
    state = torch.load(checkpoint, map_location="cpu")
    args = state.get("args", {})
    model = T5ResidualTokenTransformer(
        codebook_size=tokenizer.quantizer.codebook_size,
        hidden_features=int(args.get("hidden_features", 128)),
        layers=int(args.get("layers", 2)),
        heads=int(args.get("heads", 4)),
    )
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return model, args


def pad_batch(arrays):
    max_len = max(len(item) for item in arrays)
    padded = np.zeros((len(arrays), max_len, arrays[0].shape[-1]), dtype=np.float32)
    lengths = []
    for index, item in enumerate(arrays):
        length = len(item)
        lengths.append(length)
        padded[index, :length] = item
        if length < max_len and length > 0:
            padded[index, length:] = item[-1]
    return torch.from_numpy(padded), lengths


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


def sequence_stats(sequence, reference):
    sequence = np.asarray(sequence, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    speed = np.abs(np.diff(sequence, axis=0)).mean()
    gt_speed = np.abs(np.diff(reference, axis=0)).mean()
    tstd = sequence.std(axis=0).mean()
    gt_tstd = reference.std(axis=0).mean()
    return {
        "speed_ratio": float(speed / max(gt_speed, 1e-9)),
        "tstd_ratio": float(tstd / max(gt_tstd, 1e-9)),
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "below_0_30": float(np.mean(values < 0.30)),
        "below_0_10": float(np.mean(values < 0.10)),
    }


def predict_refined_batch(model, tokenizer, bases, texts, device):
    base_tensor, lengths = pad_batch([base[:, HAND_START:HAND_END] for base in bases])
    base_tensor = base_tensor.to(device)
    mask = torch.zeros((len(bases), base_tensor.shape[1]), dtype=torch.float32, device=device)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1.0
    text_tokens = T5ResidualTokenTransformer.hash_text(texts, 4096, 96, device)
    with torch.no_grad():
        logits = model(base_tensor, text_tokens, mask)
        pred_tokens = logits.argmax(dim=-1)
        decoded = tokenizer.decode(pred_tokens).detach().cpu().numpy()

    refined = []
    for index, base in enumerate(bases):
        length = len(base)
        item = base.copy()
        item[:, HAND_START:HAND_END] = item[:, HAND_START:HAND_END] + decoded[index, :length]
        refined.append(item.astype(np.float32))
    return refined


def main():
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transformer_state = torch.load(args.transformer_ckpt, map_location="cpu")
    transformer_args = transformer_state.get("args", {})
    tokenizer_ckpt = args.tokenizer_ckpt or Path(transformer_args["tokenizer_ckpt"])
    tokenizer = load_tokenizer(tokenizer_ckpt, device)
    transformer, loaded_args = load_transformer(args.transformer_ckpt, tokenizer, device)

    names = sorted(path.stem for path in args.p5_dir.glob("*.pkl"))
    if args.max_samples:
        names = names[:args.max_samples]
    payloads = {name: load_payload(args.p5_dir / f"{name}.pkl") for name in names}

    mean = torch.load(args.mean_path, map_location=device).float().to(device)
    std = torch.load(args.std_path, map_location=device).float().to(device)
    mean = torch.cat([mean[(3 + 3 * 11):][:-20], mean[(3 + 3 * 11):][-10:]], dim=0)
    std = torch.cat([std[(3 + 3 * 11):][:-20], std[(3 + 3 * 11):][-10:]], dim=0)

    metric = TM2TMetrics(metric_cfg()).to(device)
    vitality = {"base_speed": [], "t5_speed": [], "base_tstd": [], "t5_tstd": []}

    for start in range(0, len(names), args.batch_size):
        batch_names = names[start:start + args.batch_size]
        refs = []
        bases = []
        texts = []
        for name in batch_names:
            payload = payloads[name]
            bases.append(np.asarray(payload["feats_rst"], dtype=np.float32))
            refs.append(np.asarray(payload["feats_ref"], dtype=np.float32))
            texts.append(str(payload.get("text", "")))

        t5_preds = predict_refined_batch(transformer, tokenizer, bases, texts, device)
        for base, ref, refined in zip(bases, refs, t5_preds):
            ref_resampled = resample_time(ref, len(base))
            base_stats = sequence_stats(base[:, HAND_START:HAND_END], ref_resampled[:, HAND_START:HAND_END])
            t5_stats = sequence_stats(refined[:, HAND_START:HAND_END], ref_resampled[:, HAND_START:HAND_END])
            vitality["base_speed"].append(base_stats["speed_ratio"])
            vitality["base_tstd"].append(base_stats["tstd_ratio"])
            vitality["t5_speed"].append(t5_stats["speed_ratio"])
            vitality["t5_tstd"].append(t5_stats["tstd_ratio"])

        ref_pad, ref_lengths = pad_batch(refs)
        rst_pad, rst_lengths = pad_batch(t5_preds)
        max_len = max(ref_pad.shape[1], rst_pad.shape[1])
        if ref_pad.shape[1] < max_len:
            ref_pad = torch.cat([ref_pad, ref_pad[:, -1:].repeat(1, max_len - ref_pad.shape[1], 1)], dim=1)
        if rst_pad.shape[1] < max_len:
            rst_pad = torch.cat([rst_pad, rst_pad[:, -1:].repeat(1, max_len - rst_pad.shape[1], 1)], dim=1)

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
                src=[args.dataset] * len(batch_names),
                name=batch_names,
            )

    scores = metric.name2scores
    per_sample = {}
    body = []
    left = []
    right = []
    for name in names:
        raw = scores[name]
        values = {
            f"{args.dataset}_DTW_PA_JPE_body": float(raw[f"{args.dataset}_DTW_PA_JPE_body"]),
            f"{args.dataset}_DTW_PA_JPE_lhand": float(raw[f"{args.dataset}_DTW_PA_JPE_lhand"]),
            f"{args.dataset}_DTW_PA_JPE_rhand": float(raw[f"{args.dataset}_DTW_PA_JPE_rhand"]),
        }
        per_sample[name] = values
        body.append(values[f"{args.dataset}_DTW_PA_JPE_body"])
        left.append(values[f"{args.dataset}_DTW_PA_JPE_lhand"])
        right.append(values[f"{args.dataset}_DTW_PA_JPE_rhand"])

    hand = [(l + r) / 2.0 for l, r in zip(left, right)]
    report = {
        "dataset": args.dataset,
        "samples": len(names),
        "transformer_checkpoint": str(args.transformer_ckpt),
        "tokenizer_checkpoint": str(tokenizer_ckpt),
        "transformer_args": loaded_args,
        "p5_dir": str(args.p5_dir),
        "pa_body": float(np.mean(body)),
        "pa_lhand": float(np.mean(left)),
        "pa_rhand": float(np.mean(right)),
        "pa_hand": float(np.mean(hand)),
        "vitality": {key: summarize(value) for key, value in vitality.items()},
        "scores": per_sample,
    }
    (args.output_dir / f"{args.dataset}_summary.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "scores"}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / f"{args.dataset}_scores.json").write_text(
        json.dumps(per_sample, indent=2),
        encoding="utf-8",
    )
    md = "\n".join(
        [
            f"# T5-1 residual transformer paired eval - {args.dataset}",
            "",
            f"Samples: **{len(names)}**",
            "",
            "| PA-body ↓ | PA-hand ↓ | PA-LH ↓ | PA-RH ↓ | Base speed/GT | T5-1 speed/GT | Base tstd/GT | T5-1 tstd/GT |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {report['pa_body']:.4f} | {report['pa_hand']:.4f} | "
                f"{report['pa_lhand']:.4f} | {report['pa_rhand']:.4f} | "
                f"{report['vitality']['base_speed']['mean']:.3f} | "
                f"{report['vitality']['t5_speed']['mean']:.3f} | "
                f"{report['vitality']['base_tstd']['mean']:.3f} | "
                f"{report['vitality']['t5_tstd']['mean']:.3f} |"
            ),
            "",
        ]
    )
    (args.output_dir / f"{args.dataset}_summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
