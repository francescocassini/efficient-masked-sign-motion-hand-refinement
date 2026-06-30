#!/usr/bin/env python3
"""Feature-level sweep for gated T5 residual application.

This is a T5-2 diagnostic: instead of applying the predicted residual
everywhere, apply it only to selected frames and compare against random and
oracle gates. It does not modify P3/P5/T5 checkpoints.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from mGPT.models.utils.t5_residual_transformer_conf import T5ResidualConfidenceTransformer
from mGPT.models.utils.t5_residual_transformer_fullfeat import T5ResidualFullFeatureTransformer
from mGPT.models.utils.t5_residual_transformer_tokenc import T5ResidualTokenConditionedTransformer
from scripts.t5_residual_tokenizer_oracle import HAND_END, HAND_START, resample_time
from scripts.t5_residual_transformer_conf_train import confidence_to_frames, tokens_to_frames
from scripts.t5_residual_transformer_paired_eval import load_tokenizer, pad_batch, sequence_stats, summarize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["csl", "phoenix", "both"])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--transformer-ckpt", required=True, type=Path)
    parser.add_argument("--tokenizer-ckpt", default=None, type=Path)
    parser.add_argument("--model-kind", default="conf", choices=["conf", "tokenc", "fullfeat"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", default="0.10,0.20,0.30,0.50,1.00")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dataset_names(cache_dir: Path, dataset: str, max_samples: int = 0):
    names = sorted(path.stem for path in cache_dir.glob("*.pkl"))
    if dataset == "csl":
        names = [name for name in names if name.startswith("S")]
    elif dataset == "phoenix":
        names = [name for name in names if not name.startswith("S")]
    if max_samples:
        names = names[:max_samples]
    return names


def load_transformer(checkpoint, tokenizer, model_kind, device):
    state = torch.load(checkpoint, map_location="cpu")
    args = state.get("args", {})
    common = {
        "codebook_size": tokenizer.quantizer.codebook_size,
        "hidden_features": int(args.get("hidden_features", 128)),
        "layers": int(args.get("layers", 2)),
        "heads": int(args.get("heads", 4)),
    }
    if model_kind == "conf":
        model = T5ResidualConfidenceTransformer(**common)
    elif model_kind == "tokenc":
        model = T5ResidualTokenConditionedTransformer(**common)
    else:
        model = T5ResidualFullFeatureTransformer(**common)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return model, args


def pad_int_batch(arrays):
    max_len = max(len(item) for item in arrays)
    padded = np.zeros((len(arrays), max_len), dtype=np.int64)
    for index, item in enumerate(arrays):
        length = len(item)
        padded[index, :length] = item
        if length < max_len and length > 0:
            padded[index, length:] = item[-1]
    return torch.from_numpy(padded)


def predict_residual_batch(model, tokenizer, payloads, model_kind, device):
    bases = [np.asarray(payload["feats_rst"], dtype=np.float32) for payload in payloads]
    texts = [str(payload.get("text", "")) for payload in payloads]
    base_tensor, lengths = pad_batch(bases)
    base_tensor = base_tensor.to(device)
    mask = torch.zeros((len(bases), base_tensor.shape[1]), dtype=torch.float32, device=device)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1.0

    text_tokens = T5ResidualConfidenceTransformer.hash_text(texts, 4096, 96, device)
    with torch.no_grad():
        if model_kind == "conf":
            body = pad_int_batch([tokens_to_frames(payload["tokens_body"], len(base)) for payload, base in zip(payloads, bases)]).to(device)
            lhand = pad_int_batch([tokens_to_frames(payload["tokens_lhand"], len(base)) for payload, base in zip(payloads, bases)]).to(device)
            rhand = pad_int_batch([tokens_to_frames(payload["tokens_rhand"], len(base)) for payload, base in zip(payloads, bases)]).to(device)
            confidence = [
                np.stack(
                    [
                        confidence_to_frames(payload["confidence_body"], len(base)),
                        confidence_to_frames(payload["confidence_lhand"], len(base)),
                        confidence_to_frames(payload["confidence_rhand"], len(base)),
                    ],
                    axis=-1,
                ).astype(np.float32)
                for payload, base in zip(payloads, bases)
            ]
            confidence, _ = pad_batch(confidence)
            logits = model(base_tensor, body, lhand, rhand, confidence.to(device), text_tokens, mask)
        elif model_kind == "tokenc":
            body = pad_int_batch([tokens_to_frames(payload["tokens_body"], len(base)) for payload, base in zip(payloads, bases)]).to(device)
            lhand = pad_int_batch([tokens_to_frames(payload["tokens_lhand"], len(base)) for payload, base in zip(payloads, bases)]).to(device)
            rhand = pad_int_batch([tokens_to_frames(payload["tokens_rhand"], len(base)) for payload, base in zip(payloads, bases)]).to(device)
            logits = model(base_tensor, body, lhand, rhand, text_tokens, mask)
        else:
            logits = model(base_tensor, text_tokens, mask)
        pred_tokens = logits.argmax(dim=-1)
        decoded = tokenizer.decode(pred_tokens).detach().cpu().numpy()
    return bases, decoded, lengths


def select_mask(gate, budget, base_hand, pred_hand, ref_hand, payload, rng):
    length = len(base_hand)
    if budget >= 1.0:
        return np.ones((length,), dtype=bool)
    count = max(1, int(round(length * budget)))
    if gate == "none":
        return np.zeros((length,), dtype=bool)
    if gate == "random":
        chosen = rng.choice(length, size=count, replace=False)
    elif gate == "low_conf":
        conf_l = confidence_to_frames(payload["confidence_lhand"], length)
        conf_r = confidence_to_frames(payload["confidence_rhand"], length)
        score = (conf_l + conf_r) / 2.0
        chosen = np.argsort(score)[:count]
    elif gate == "high_error_oracle":
        score = np.abs(base_hand - ref_hand).mean(axis=1)
        chosen = np.argsort(score)[-count:]
    elif gate == "positive_gain_oracle":
        base_err = np.abs(base_hand - ref_hand).mean(axis=1)
        pred_err = np.abs(pred_hand - ref_hand).mean(axis=1)
        gain = base_err - pred_err
        positive = np.where(gain > 0)[0]
        if len(positive) == 0:
            chosen = np.array([], dtype=np.int64)
        else:
            chosen = positive[np.argsort(gain[positive])[-min(count, len(positive)):]]
    else:
        raise ValueError(f"Unknown gate: {gate}")
    mask = np.zeros((length,), dtype=bool)
    mask[chosen] = True
    return mask


def speed(x):
    if len(x) < 2:
        return 0.0
    return float(np.abs(np.diff(x, axis=0)).mean())


def evaluate_gate(names, payloads, model, tokenizer, model_kind, budgets, device, seed):
    gates = ["none", "all", "low_conf", "random", "high_error_oracle", "positive_gain_oracle"]
    rows = {
        (gate, budget): {
            "base_mae": [],
            "pred_mae": [],
            "base_speed": [],
            "pred_speed": [],
            "base_tstd": [],
            "pred_tstd": [],
            "win": [],
            "applied": [],
        }
        for gate in gates
        for budget in budgets
        if not (gate in {"none", "all"} and budget != budgets[0])
    }
    rng = np.random.default_rng(seed)
    for start in range(0, len(names), 32):
        batch_payloads = payloads[start:start + 32]
        bases, residuals, lengths = predict_residual_batch(
            model, tokenizer, batch_payloads, model_kind, device
        )
        for payload, base, residual in zip(batch_payloads, bases, residuals):
            ref = np.asarray(payload["feats_ref"], dtype=np.float32)
            ref_resampled = resample_time(ref, len(base))
            base_hand = base[:, HAND_START:HAND_END]
            pred_hand_full = base_hand + residual[: len(base)]
            ref_hand = ref_resampled[:, HAND_START:HAND_END]
            base_err_frames = np.abs(base_hand - ref_hand).mean(axis=1)

            for budget in budgets:
                for gate in gates:
                    if gate == "none" and budget != budgets[0]:
                        continue
                    if gate == "all" and budget != budgets[0]:
                        continue
                    effective_budget = 1.0 if gate == "all" else budget
                    mask = select_mask(
                        gate,
                        effective_budget,
                        base_hand,
                        pred_hand_full,
                        ref_hand,
                        payload,
                        rng,
                    )
                    pred_hand = base_hand.copy()
                    pred_hand[mask] = pred_hand_full[mask]
                    pred_err_frames = np.abs(pred_hand - ref_hand).mean(axis=1)
                    stats = sequence_stats(pred_hand, ref_hand)
                    base_stats = sequence_stats(base_hand, ref_hand)
                    key = (gate, budget)
                    rows[key]["base_mae"].append(float(base_err_frames.mean()))
                    rows[key]["pred_mae"].append(float(pred_err_frames.mean()))
                    rows[key]["base_speed"].append(base_stats["speed_ratio"])
                    rows[key]["pred_speed"].append(stats["speed_ratio"])
                    rows[key]["base_tstd"].append(base_stats["tstd_ratio"])
                    rows[key]["pred_tstd"].append(stats["tstd_ratio"])
                    rows[key]["win"].append(float(pred_err_frames.mean() < base_err_frames.mean()))
                    rows[key]["applied"].append(float(mask.mean()))
    summary = []
    for (gate, budget), values in rows.items():
        base_mae = float(np.mean(values["base_mae"]))
        pred_mae = float(np.mean(values["pred_mae"]))
        summary.append(
            {
                "gate": gate,
                "budget": float(1.0 if gate == "all" else 0.0 if gate == "none" else budget),
                "applied": float(np.mean(values["applied"])),
                "base_mae": base_mae,
                "pred_mae": pred_mae,
                "mae_delta_pct": 100.0 * (pred_mae - base_mae) / max(base_mae, 1e-9),
                "base_speed": float(np.mean(values["base_speed"])),
                "pred_speed": float(np.mean(values["pred_speed"])),
                "base_tstd": float(np.mean(values["base_tstd"])),
                "pred_tstd": float(np.mean(values["pred_tstd"])),
                "win_rate": float(np.mean(values["win"])),
            }
        )
    return sorted(summary, key=lambda item: (item["gate"], item["budget"]))


def write_report(report, output_dir, dataset):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{dataset}_feature_gate_sweep.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    rows = report["rows"]
    md = [
        f"# T5-2 gated residual feature sweep - {dataset}",
        "",
        f"Samples: **{report['samples']}**",
        "",
        "| Gate | Budget | Applied | Δ MAE | Pred MAE ↓ | Pred speed/GT | Pred tstd/GT | Win-rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            "| {gate} | {budget:.2f} | {applied:.2f} | {mae_delta_pct:+.2f}% | "
            "{pred_mae:.4f} | {pred_speed:.3f} | {pred_tstd:.3f} | {win_rate:.3f} |".format(**row)
        )
    md.append("")
    (output_dir / f"{dataset}_feature_gate_sweep.md").write_text(
        "\n".join(md),
        encoding="utf-8",
    )
    print("\n".join(md))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    transformer_state = torch.load(args.transformer_ckpt, map_location="cpu")
    transformer_args = transformer_state.get("args", {})
    tokenizer_ckpt = args.tokenizer_ckpt or Path(transformer_args["tokenizer_ckpt"])
    tokenizer = load_tokenizer(tokenizer_ckpt, device)
    model, loaded_args = load_transformer(args.transformer_ckpt, tokenizer, args.model_kind, device)
    budgets = [float(item) for item in args.budgets.split(",") if item.strip()]
    datasets = ["csl", "phoenix"] if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        names = dataset_names(args.cache_dir, dataset, args.max_samples)
        payloads = [load_payload(args.cache_dir / f"{name}.pkl") for name in names]
        rows = evaluate_gate(names, payloads, model, tokenizer, args.model_kind, budgets, device, args.seed)
        report = {
            "dataset": dataset,
            "samples": len(names),
            "model_kind": args.model_kind,
            "transformer_checkpoint": str(args.transformer_ckpt),
            "tokenizer_checkpoint": str(tokenizer_ckpt),
            "transformer_args": loaded_args,
            "cache_dir": str(args.cache_dir),
            "rows": rows,
        }
        write_report(report, args.output_dir, dataset)


if __name__ == "__main__":
    main()
