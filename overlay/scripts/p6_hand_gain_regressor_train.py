#!/usr/bin/env python3
"""Train P6-D continuous gain regressor for P6-B hand-token candidates."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from mGPT.models.utils.p6_hand_gain_regressor import P6HandGainRegressor
from mGPT.models.utils.p6_hand_token_editor import P6HandTokenEditor
from scripts.p6_hand_gain_gate_train import predict_candidates
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_hand_token_editor_paired_eval import token_gain
from scripts.p6_token_edit_oracle_ceiling import (
    decode_hand_tokens,
    load_vq_pack,
    normalize_token_shape,
    pad_or_crop_tokens,
)
from scripts.t5_residual_tokenizer_oracle import resample_time


LH_START = 30
LH_END = 75
RH_START = 75
RH_END = 120


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--val-cache-dir", type=Path, default=None)
    parser.add_argument("--p6b-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/soke.yaml", type=Path)
    parser.add_argument("--default-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--vae-ckpt", default="deps/tokenizer_ckpt/tokenizer.ckpt", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-features", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-scale", type=float, default=10.0)
    parser.add_argument("--positive-weight", type=float, default=1.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


class P6GainRegressionDataset(Dataset):
    def __init__(self, cache_dir, editor, pack, device, max_samples=0, target_scale=10.0):
        self.samples = []
        self.target_scale = float(target_scale)
        paths = sorted(cache_dir.glob("*.pkl"))
        if max_samples:
            paths = paths[:max_samples]
        for path in paths:
            payload = load_payload(path)
            base = np.asarray(payload["feats_rst"], dtype=np.float32)
            ref = np.asarray(payload["feats_ref"], dtype=np.float32)
            if len(base) < 2 or len(ref) < 2:
                continue
            token_len = min(
                len(normalize_token_shape(payload["tokens_lhand"])),
                len(normalize_token_shape(payload["tokens_rhand"])),
            )
            if token_len < 2:
                continue
            body, lhand, rhand, l_new, r_new, conf, l_model_conf, r_model_conf = predict_candidates(
                editor, payload, token_len, device
            )
            ref_aligned = resample_time(ref, len(base)).astype(np.float32)
            l_candidate = decode_hand_tokens(pack.hand_vae, l_new, len(base), device)
            r_candidate = decode_hand_tokens(pack.rhand_vae, r_new, len(base), device)
            l_gain = token_gain(base[:, LH_START:LH_END], l_candidate, ref_aligned[:, LH_START:LH_END], token_len)
            r_gain = token_gain(base[:, RH_START:RH_END], r_candidate, ref_aligned[:, RH_START:RH_END], token_len)
            progress = np.linspace(0.0, 1.0, token_len, dtype=np.float32)
            meta = np.concatenate(
                [
                    conf,
                    l_model_conf[:, None],
                    r_model_conf[:, None],
                    (l_new != lhand).astype(np.float32)[:, None],
                    (r_new != rhand).astype(np.float32)[:, None],
                ],
                axis=-1,
            ).astype(np.float32)
            meta[:, 0] = 0.5 * meta[:, 0] + 0.5 * progress
            self.samples.append(
                {
                    "name": path.stem,
                    "text": str(payload.get("text", "")),
                    "body": body,
                    "lhand": lhand,
                    "rhand": rhand,
                    "cand_lhand": l_new,
                    "cand_rhand": r_new,
                    "meta": meta,
                    "target_lhand": (l_gain * self.target_scale).astype(np.float32),
                    "target_rhand": (r_gain * self.target_scale).astype(np.float32),
                    "raw_lhand_gain": l_gain.astype(np.float32),
                    "raw_rhand_gain": r_gain.astype(np.float32),
                }
            )
        if not self.samples:
            raise RuntimeError("No P6-D samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(batch):
    max_len = max(len(item["body"]) for item in batch)
    out = {
        "body": np.zeros((len(batch), max_len), dtype=np.int64),
        "lhand": np.zeros((len(batch), max_len), dtype=np.int64),
        "rhand": np.zeros((len(batch), max_len), dtype=np.int64),
        "cand_lhand": np.zeros((len(batch), max_len), dtype=np.int64),
        "cand_rhand": np.zeros((len(batch), max_len), dtype=np.int64),
        "meta": np.zeros((len(batch), max_len, 7), dtype=np.float32),
        "target_lhand": np.zeros((len(batch), max_len), dtype=np.float32),
        "target_rhand": np.zeros((len(batch), max_len), dtype=np.float32),
        "raw_lhand_gain": np.zeros((len(batch), max_len), dtype=np.float32),
        "raw_rhand_gain": np.zeros((len(batch), max_len), dtype=np.float32),
        "mask": np.zeros((len(batch), max_len), dtype=np.float32),
    }
    texts = []
    names = []
    for index, item in enumerate(batch):
        length = len(item["body"])
        for key in ["body", "lhand", "rhand", "cand_lhand", "cand_rhand"]:
            out[key][index, :length] = item[key]
            if length < max_len:
                out[key][index, length:] = item[key][-1]
        for key in ["meta", "target_lhand", "target_rhand", "raw_lhand_gain", "raw_rhand_gain"]:
            out[key][index, :length] = item[key]
            if length < max_len:
                out[key][index, length:] = item[key][-1]
        out["mask"][index, :length] = 1.0
        texts.append(item["text"])
        names.append(item["name"])
    batch_out = {key: torch.from_numpy(value) for key, value in out.items()}
    batch_out["texts"] = texts
    batch_out["names"] = names
    return batch_out


def masked_smooth_l1(pred, target, raw_gain, mask, positive_weight):
    weight = torch.ones_like(target)
    weight = torch.where(raw_gain > 0.0, weight * float(positive_weight), weight)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * weight * mask
    return loss.sum() / (weight * mask).sum().clamp_min(1.0)


def masked_corr(pred, target, mask):
    pred = pred[mask.bool()]
    target = target[mask.bool()]
    if pred.numel() < 2:
        return 0.0
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float((pred * target).sum().div(denom).item())


def selected_gain_at_budget(score, raw_gain, mask, budget):
    values = []
    score_np = score.detach().cpu().numpy()
    gain_np = raw_gain.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    for row_score, row_gain, row_mask in zip(score_np, gain_np, mask_np):
        valid_score = row_score[row_mask]
        valid_gain = row_gain[row_mask]
        if len(valid_score) == 0:
            continue
        count = max(1, int(round(len(valid_score) * budget)))
        order = np.argsort(-valid_score)[:count]
        values.append(float(valid_gain[order].mean()))
    return float(np.mean(values)) if values else 0.0


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values))}


def evaluate(model, loader, device, target_scale, positive_weight):
    model.eval()
    losses = []
    l_corr = []
    r_corr = []
    l_top10 = []
    r_top10 = []
    l_top20 = []
    r_top20 = []
    with torch.no_grad():
        for batch in loader:
            body = batch["body"].to(device)
            lhand = batch["lhand"].to(device)
            rhand = batch["rhand"].to(device)
            cand_lhand = batch["cand_lhand"].to(device)
            cand_rhand = batch["cand_rhand"].to(device)
            meta = batch["meta"].to(device)
            target_lhand = batch["target_lhand"].to(device)
            target_rhand = batch["target_rhand"].to(device)
            raw_lhand = batch["raw_lhand_gain"].to(device)
            raw_rhand = batch["raw_rhand_gain"].to(device)
            mask = batch["mask"].to(device)
            text = P6HandGainRegressor.hash_text(batch["texts"], 4096, 96, device)
            l_score, r_score = model(body, lhand, rhand, cand_lhand, cand_rhand, meta, text, mask)
            l_loss = masked_smooth_l1(l_score, target_lhand, raw_lhand, mask, positive_weight)
            r_loss = masked_smooth_l1(r_score, target_rhand, raw_rhand, mask, positive_weight)
            losses.append(float((l_loss + r_loss).item()))
            l_corr.append(masked_corr(l_score, target_lhand, mask))
            r_corr.append(masked_corr(r_score, target_rhand, mask))
            l_top10.append(selected_gain_at_budget(l_score / target_scale, raw_lhand, mask, 0.10))
            r_top10.append(selected_gain_at_budget(r_score / target_scale, raw_rhand, mask, 0.10))
            l_top20.append(selected_gain_at_budget(l_score / target_scale, raw_lhand, mask, 0.20))
            r_top20.append(selected_gain_at_budget(r_score / target_scale, raw_rhand, mask, 0.20))
    return {
        "loss": float(np.mean(losses)),
        "lhand_corr": summarize(l_corr),
        "rhand_corr": summarize(r_corr),
        "lhand_top10_gain": summarize(l_top10),
        "rhand_top10_gain": summarize(r_top10),
        "lhand_top20_gain": summarize(l_top20),
        "rhand_top20_gain": summarize(r_top20),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    pack = load_vq_pack(args, device)
    editor, _editor_args = load_p6_editor(args.p6b_checkpoint, device)
    editor.eval()
    dataset = P6GainRegressionDataset(
        args.cache_dir,
        editor,
        pack,
        device,
        max_samples=args.max_samples,
        target_scale=args.target_scale,
    )
    val_dataset = None
    if args.val_cache_dir is not None:
        val_dataset = P6GainRegressionDataset(
            args.val_cache_dir,
            editor,
            pack,
            device,
            max_samples=args.max_samples,
            target_scale=args.target_scale,
        )
        train_indices = list(range(len(dataset)))
        val_indices = list(range(len(val_dataset)))
        split_source = "separate_cache"
    else:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        val_count = max(1, int(round(len(indices) * args.val_ratio)))
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        split_source = "random_sample_split"
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        Subset(val_dataset if val_dataset is not None else dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = P6HandGainRegressor(
        hidden_features=args.hidden_features,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_score = None
    best_path = args.output_dir / "best_p6_hand_gain_regressor.pt"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            body = batch["body"].to(device)
            lhand = batch["lhand"].to(device)
            rhand = batch["rhand"].to(device)
            cand_lhand = batch["cand_lhand"].to(device)
            cand_rhand = batch["cand_rhand"].to(device)
            meta = batch["meta"].to(device)
            target_lhand = batch["target_lhand"].to(device)
            target_rhand = batch["target_rhand"].to(device)
            raw_lhand = batch["raw_lhand_gain"].to(device)
            raw_rhand = batch["raw_rhand_gain"].to(device)
            mask = batch["mask"].to(device)
            text = P6HandGainRegressor.hash_text(batch["texts"], 4096, 96, device)
            l_score, r_score = model(body, lhand, rhand, cand_lhand, cand_rhand, meta, text, mask)
            loss = masked_smooth_l1(l_score, target_lhand, raw_lhand, mask, args.positive_weight)
            loss = loss + masked_smooth_l1(r_score, target_rhand, raw_rhand, mask, args.positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate(model, val_loader, device, args.target_scale, args.positive_weight)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val_metrics}
        history.append(entry)
        print(json.dumps(entry, indent=2))
        score = val_metrics["lhand_top20_gain"]["mean"] + val_metrics["rhand_top20_gain"]["mean"]
        if best_score is None or score > best_score:
            best_score = score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": {
                        "hidden_features": args.hidden_features,
                        "layers": args.layers,
                        "heads": args.heads,
                        "dropout": args.dropout,
                        "target_scale": args.target_scale,
                        "positive_weight": args.positive_weight,
                        "p6b_checkpoint": str(args.p6b_checkpoint),
                        "cache_dir": str(args.cache_dir),
                        "val_cache_dir": str(args.val_cache_dir) if args.val_cache_dir else None,
                        "split_source": split_source,
                    },
                    "epoch": epoch,
                    "val": val_metrics,
                },
                best_path,
            )
    train_metrics = evaluate(model, train_loader, device, args.target_scale, args.positive_weight)
    val_metrics = evaluate(model, val_loader, device, args.target_scale, args.positive_weight)
    report = {
        "samples": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "val_cache_samples": len(val_dataset) if val_dataset is not None else None,
        "split_source": split_source,
        "history": history,
        "final_train": train_metrics,
        "final_val": val_metrics,
        "best_checkpoint": str(best_path),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# P6-D hand gain regressor train",
        "",
        f"Samples: **{len(dataset)}** train={len(train_indices)} val={len(val_indices)}",
        f"Split source: **{split_source}**",
        f"Best checkpoint: `{best_path}`",
        "",
        "| Split | Loss ↓ | LH corr ↑ | RH corr ↑ | LH top10 gain ↑ | RH top10 gain ↑ | LH top20 gain ↑ | RH top20 gain ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
        md.append(
            f"| {split} | {metrics['loss']:.4f} | "
            f"{metrics['lhand_corr']['mean']:.3f} | {metrics['rhand_corr']['mean']:.3f} | "
            f"{metrics['lhand_top10_gain']['mean']:.5f} | {metrics['rhand_top10_gain']['mean']:.5f} | "
            f"{metrics['lhand_top20_gain']['mean']:.5f} | {metrics['rhand_top20_gain']['mean']:.5f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
