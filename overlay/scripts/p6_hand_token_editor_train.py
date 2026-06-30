#!/usr/bin/env python3
"""Train P6-B learned hand-token editor on top of the P5 cache."""

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

from mGPT.models.utils.p6_hand_token_editor import P6HandTokenEditor
from scripts.p6_token_edit_oracle_ceiling import (
    encode_hand_tokens,
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/soke.yaml", type=Path)
    parser.add_argument("--default-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--vae-ckpt", default="deps/tokenizer_ckpt/tokenizer.ckpt", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--low-conf-weight", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def pad_or_crop_float(values, target_len):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) == target_len:
        return values
    if len(values) > target_len:
        return values[:target_len]
    if len(values) == 0:
        return np.zeros((target_len, values.shape[-1]), dtype=np.float32)
    pad = np.repeat(values[-1:], target_len - len(values), axis=0)
    return np.concatenate([values, pad], axis=0).astype(np.float32)


class P6HandTokenDataset(Dataset):
    def __init__(self, cache_dir, pack, device, max_samples=0):
        self.samples = []
        paths = sorted(cache_dir.glob("*.pkl"))
        if max_samples:
            paths = paths[:max_samples]
        for path in paths:
            payload = load_payload(path)
            base = np.asarray(payload["feats_rst"], dtype=np.float32)
            ref = np.asarray(payload["feats_ref"], dtype=np.float32)
            if len(base) < 2 or len(ref) < 2:
                continue
            l_pred = normalize_token_shape(payload["tokens_lhand"])
            r_pred = normalize_token_shape(payload["tokens_rhand"])
            token_len = min(len(l_pred), len(r_pred))
            if token_len < 2:
                continue
            l_pred = pad_or_crop_tokens(l_pred, token_len)
            r_pred = pad_or_crop_tokens(r_pred, token_len)
            body = pad_or_crop_tokens(payload["tokens_body"], token_len)
            ref_aligned = resample_time(ref, len(base)).astype(np.float32)
            l_gt = encode_hand_tokens(pack.hand_vae, ref_aligned[:, LH_START:LH_END], device)
            r_gt = encode_hand_tokens(pack.rhand_vae, ref_aligned[:, RH_START:RH_END], device)
            l_gt = pad_or_crop_tokens(l_gt, token_len)
            r_gt = pad_or_crop_tokens(r_gt, token_len)
            confidence = np.concatenate(
                [
                    pad_or_crop_float(payload["confidence_body"], token_len),
                    pad_or_crop_float(payload["confidence_lhand"], token_len),
                    pad_or_crop_float(payload["confidence_rhand"], token_len),
                ],
                axis=-1,
            )
            self.samples.append(
                {
                    "name": path.stem,
                    "text": str(payload.get("text", "")),
                    "body": body,
                    "lhand": l_pred,
                    "rhand": r_pred,
                    "confidence": confidence,
                    "target_lhand": l_gt,
                    "target_rhand": r_gt,
                }
            )
        if not self.samples:
            raise RuntimeError("No P6 samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(batch):
    max_len = max(len(item["body"]) for item in batch)
    body = np.zeros((len(batch), max_len), dtype=np.int64)
    lhand = np.zeros((len(batch), max_len), dtype=np.int64)
    rhand = np.zeros((len(batch), max_len), dtype=np.int64)
    target_lhand = np.zeros((len(batch), max_len), dtype=np.int64)
    target_rhand = np.zeros((len(batch), max_len), dtype=np.int64)
    confidence = np.zeros((len(batch), max_len, 3), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=np.float32)
    texts = []
    names = []
    for index, item in enumerate(batch):
        length = len(item["body"])
        body[index, :length] = item["body"]
        lhand[index, :length] = item["lhand"]
        rhand[index, :length] = item["rhand"]
        target_lhand[index, :length] = item["target_lhand"]
        target_rhand[index, :length] = item["target_rhand"]
        confidence[index, :length] = item["confidence"]
        if length < max_len:
            body[index, length:] = item["body"][-1]
            lhand[index, length:] = item["lhand"][-1]
            rhand[index, length:] = item["rhand"][-1]
            target_lhand[index, length:] = item["target_lhand"][-1]
            target_rhand[index, length:] = item["target_rhand"][-1]
            confidence[index, length:] = item["confidence"][-1]
        mask[index, :length] = 1.0
        texts.append(item["text"])
        names.append(item["name"])
    return {
        "body": torch.from_numpy(body),
        "lhand": torch.from_numpy(lhand),
        "rhand": torch.from_numpy(rhand),
        "target_lhand": torch.from_numpy(target_lhand),
        "target_rhand": torch.from_numpy(target_rhand),
        "confidence": torch.from_numpy(confidence),
        "mask": torch.from_numpy(mask),
        "texts": texts,
        "names": names,
    }


def masked_token_ce(logits, target, mask, confidence, low_conf_weight):
    ce = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    weights = 1.0 + float(low_conf_weight) * (1.0 - confidence).clamp(0.0, 1.0)
    weighted = ce * weights * mask
    return weighted.sum() / (weights * mask).sum().clamp_min(1.0)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def evaluate(model, loader, device, low_conf_weight):
    model.eval()
    losses = []
    l_acc = []
    r_acc = []
    l_keep = []
    r_keep = []
    with torch.no_grad():
        for batch in loader:
            body = batch["body"].to(device)
            lhand = batch["lhand"].to(device)
            rhand = batch["rhand"].to(device)
            target_lhand = batch["target_lhand"].to(device)
            target_rhand = batch["target_rhand"].to(device)
            confidence = batch["confidence"].to(device)
            mask = batch["mask"].to(device)
            text = P6HandTokenEditor.hash_text(batch["texts"], 4096, 96, device)
            l_logits, r_logits = model(body, lhand, rhand, confidence, text, mask)
            l_loss = masked_token_ce(l_logits, target_lhand, mask, confidence[..., 1], low_conf_weight)
            r_loss = masked_token_ce(r_logits, target_rhand, mask, confidence[..., 2], low_conf_weight)
            losses.append(float((l_loss + r_loss).item()))
            l_pred = l_logits.argmax(dim=-1)
            r_pred = r_logits.argmax(dim=-1)
            denom = mask.sum().clamp_min(1.0)
            l_acc.append(float((((l_pred == target_lhand).float() * mask).sum() / denom).item()))
            r_acc.append(float((((r_pred == target_rhand).float() * mask).sum() / denom).item()))
            l_keep.append(float((((l_pred == lhand).float() * mask).sum() / denom).item()))
            r_keep.append(float((((r_pred == rhand).float() * mask).sum() / denom).item()))
    return {
        "loss": float(np.mean(losses)),
        "lhand_acc": summarize(l_acc),
        "rhand_acc": summarize(r_acc),
        "lhand_keep_rate": summarize(l_keep),
        "rhand_keep_rate": summarize(r_keep),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    pack = load_vq_pack(args, device)
    dataset = P6HandTokenDataset(args.cache_dir, pack, device, max_samples=args.max_samples)
    val_dataset = None
    if args.val_cache_dir is not None:
        val_dataset = P6HandTokenDataset(args.val_cache_dir, pack, device, max_samples=args.max_samples)
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
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        Subset(val_dataset if val_dataset is not None else dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = P6HandTokenEditor(
        hidden_features=args.hidden_features,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_val = None
    best_path = args.output_dir / "best_p6_hand_token_editor.pt"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            body = batch["body"].to(device)
            lhand = batch["lhand"].to(device)
            rhand = batch["rhand"].to(device)
            target_lhand = batch["target_lhand"].to(device)
            target_rhand = batch["target_rhand"].to(device)
            confidence = batch["confidence"].to(device)
            mask = batch["mask"].to(device)
            text = P6HandTokenEditor.hash_text(batch["texts"], 4096, 96, device)
            l_logits, r_logits = model(body, lhand, rhand, confidence, text, mask)
            l_loss = masked_token_ce(l_logits, target_lhand, mask, confidence[..., 1], args.low_conf_weight)
            r_loss = masked_token_ce(r_logits, target_rhand, mask, confidence[..., 2], args.low_conf_weight)
            loss = l_loss + r_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate(model, val_loader, device, args.low_conf_weight)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val_metrics}
        history.append(entry)
        print(json.dumps(entry, indent=2))
        if best_val is None or val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": {
                        "hidden_features": args.hidden_features,
                        "layers": args.layers,
                        "heads": args.heads,
                        "dropout": args.dropout,
                        "low_conf_weight": args.low_conf_weight,
                        "cache_dir": str(args.cache_dir),
                        "val_cache_dir": str(args.val_cache_dir) if args.val_cache_dir else None,
                        "split_source": split_source,
                        "vae_ckpt": str(args.vae_ckpt),
                    },
                    "epoch": epoch,
                    "val": val_metrics,
                },
                best_path,
            )

    train_metrics = evaluate(model, train_loader, device, args.low_conf_weight)
    val_metrics = evaluate(model, val_loader, device, args.low_conf_weight)
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
        "args": vars(args),
    }
    report["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in report["args"].items()}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# P6-B learned hand-token editor train",
        "",
        f"Samples: **{len(dataset)}** train={len(train_indices)} val={len(val_indices)}",
        f"Split source: **{split_source}**",
        f"Best checkpoint: `{best_path}`",
        "",
        "| Split | Loss ↓ | LH acc ↑ | RH acc ↑ | LH keep | RH keep |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
        md.append(
            f"| {split} | {metrics['loss']:.4f} | "
            f"{metrics['lhand_acc']['mean']:.3f} | {metrics['rhand_acc']['mean']:.3f} | "
            f"{metrics['lhand_keep_rate']['mean']:.3f} | {metrics['rhand_keep_rate']['mean']:.3f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
