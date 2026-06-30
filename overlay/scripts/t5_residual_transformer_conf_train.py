#!/usr/bin/env python3
"""Train/evaluate T5-1D residual tokens from P5 features + tokens + confidence."""

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

from mGPT.models.utils.t5_residual_transformer_conf import T5ResidualConfidenceTransformer
from mGPT.models.utils.t5_residual_vq import T5ResidualVQVAE
from scripts.t5_residual_tokenizer_oracle import HAND_END, HAND_START, resample_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-ckpt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
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


def tokens_to_frames(tokens, length):
    tokens = np.asarray(tokens, dtype=np.int64)
    if len(tokens) == 0:
        return np.zeros((length,), dtype=np.int64)
    frame_tokens = np.repeat(tokens, 4)
    if len(frame_tokens) < length:
        pad = np.full((length - len(frame_tokens),), frame_tokens[-1], dtype=np.int64)
        frame_tokens = np.concatenate([frame_tokens, pad], axis=0)
    return frame_tokens[:length].astype(np.int64)


def confidence_to_frames(confidence, length):
    confidence = np.asarray(confidence, dtype=np.float32)
    if len(confidence) == 0:
        return np.zeros((length,), dtype=np.float32)
    frame_confidence = np.repeat(confidence, 4)
    if len(frame_confidence) < length:
        pad = np.full(
            (length - len(frame_confidence),),
            frame_confidence[-1],
            dtype=np.float32,
        )
        frame_confidence = np.concatenate([frame_confidence, pad], axis=0)
    return frame_confidence[:length].astype(np.float32)


class T5ResidualTokenCacheDataset(Dataset):
    def __init__(self, cache_dir, max_samples=0):
        self.samples = []
        for path in sorted(cache_dir.glob("*.pkl")):
            payload = load_payload(path)
            base = np.asarray(payload["feats_rst"], dtype=np.float32)
            ref = np.asarray(payload["feats_ref"], dtype=np.float32)
            if len(base) < 2 or len(ref) < 2:
                continue
            ref_resampled = resample_time(ref, len(base))
            residual = ref_resampled[:, HAND_START:HAND_END] - base[:, HAND_START:HAND_END]
            self.samples.append(
                {
                    "name": path.stem,
                    "text": str(payload.get("text", "")),
                    "base_full": base.astype(np.float32),
                    "base_hand": base[:, HAND_START:HAND_END].astype(np.float32),
                    "ref_hand": ref_resampled[:, HAND_START:HAND_END].astype(np.float32),
                    "residual": residual.astype(np.float32),
                    "body_tokens": tokens_to_frames(payload["tokens_body"], len(base)),
                    "lhand_tokens": tokens_to_frames(payload["tokens_lhand"], len(base)),
                    "rhand_tokens": tokens_to_frames(payload["tokens_rhand"], len(base)),
                    "confidence": np.stack(
                        [
                            confidence_to_frames(payload["confidence_body"], len(base)),
                            confidence_to_frames(payload["confidence_lhand"], len(base)),
                            confidence_to_frames(payload["confidence_rhand"], len(base)),
                        ],
                        axis=-1,
                    ).astype(np.float32),
                }
            )
        if max_samples:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError("No samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(batch):
    max_len = max(item["base_full"].shape[0] for item in batch)
    base_full = np.zeros((len(batch), max_len, 133), dtype=np.float32)
    base_hand = np.zeros((len(batch), max_len, 90), dtype=np.float32)
    ref_hand = np.zeros((len(batch), max_len, 90), dtype=np.float32)
    residual = np.zeros((len(batch), max_len, 90), dtype=np.float32)
    body_tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    lhand_tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    rhand_tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    confidence = np.zeros((len(batch), max_len, 3), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=np.float32)
    texts = []
    names = []
    for index, item in enumerate(batch):
        length = item["base_full"].shape[0]
        base_full[index, :length] = item["base_full"]
        base_hand[index, :length] = item["base_hand"]
        ref_hand[index, :length] = item["ref_hand"]
        residual[index, :length] = item["residual"]
        body_tokens[index, :length] = item["body_tokens"]
        lhand_tokens[index, :length] = item["lhand_tokens"]
        rhand_tokens[index, :length] = item["rhand_tokens"]
        confidence[index, :length] = item["confidence"]
        if length < max_len:
            base_full[index, length:] = item["base_full"][-1]
            base_hand[index, length:] = item["base_hand"][-1]
            ref_hand[index, length:] = item["ref_hand"][-1]
            residual[index, length:] = item["residual"][-1]
            body_tokens[index, length:] = item["body_tokens"][-1]
            lhand_tokens[index, length:] = item["lhand_tokens"][-1]
            rhand_tokens[index, length:] = item["rhand_tokens"][-1]
            confidence[index, length:] = item["confidence"][-1]
        mask[index, :length] = 1.0
        texts.append(item["text"])
        names.append(item["name"])
    return {
        "base_full": torch.from_numpy(base_full),
        "base_hand": torch.from_numpy(base_hand),
        "ref_hand": torch.from_numpy(ref_hand),
        "residual": torch.from_numpy(residual),
        "body_tokens": torch.from_numpy(body_tokens),
        "lhand_tokens": torch.from_numpy(lhand_tokens),
        "rhand_tokens": torch.from_numpy(rhand_tokens),
        "confidence": torch.from_numpy(confidence),
        "mask": torch.from_numpy(mask),
        "texts": texts,
        "names": names,
    }


def masked_ce(logits, targets, mask):
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values))}


def ratios(pred_hand, ref_hand, base_hand, mask):
    pred = pred_hand.detach().cpu().numpy()
    ref = ref_hand.detach().cpu().numpy()
    base = base_hand.detach().cpu().numpy()
    valid = mask.bool().cpu().numpy()
    rows = []
    for idx in range(pred.shape[0]):
        length = int(valid[idx].sum())
        if length < 2:
            continue
        p = pred[idx, :length]
        r = ref[idx, :length]
        b = base[idx, :length]

        def speed(x):
            return np.abs(np.diff(x, axis=0)).mean()

        rows.append(
            {
                "base_mae": float(np.abs(b - r).mean()),
                "pred_mae": float(np.abs(p - r).mean()),
                "base_speed": float(speed(b) / max(speed(r), 1e-9)),
                "pred_speed": float(speed(p) / max(speed(r), 1e-9)),
                "base_tstd": float(b.std(axis=0).mean() / max(r.std(axis=0).mean(), 1e-9)),
                "pred_tstd": float(p.std(axis=0).mean() / max(r.std(axis=0).mean(), 1e-9)),
            }
        )
    return rows


def evaluate(model, tokenizer, loader, device):
    model.eval()
    rows = []
    ce_values = []
    token_hits = []
    with torch.no_grad():
        for batch in loader:
            base_full = batch["base_full"].to(device)
            base_hand = batch["base_hand"].to(device)
            ref_hand = batch["ref_hand"].to(device)
            residual = batch["residual"].to(device)
            body_tokens = batch["body_tokens"].to(device)
            lhand_tokens = batch["lhand_tokens"].to(device)
            rhand_tokens = batch["rhand_tokens"].to(device)
            confidence = batch["confidence"].to(device)
            mask = batch["mask"].to(device)
            text_tokens = T5ResidualConfidenceTransformer.hash_text(batch["texts"], 4096, 96, device)
            target_tokens = tokenizer.encode(residual)
            logits = model(base_full, body_tokens, lhand_tokens, rhand_tokens, confidence, text_tokens, mask)
            pred_tokens = logits.argmax(dim=-1)
            decoded = tokenizer.decode(pred_tokens)
            pred_hand = base_hand + decoded
            ce_values.append(float(masked_ce(logits, target_tokens, mask).item()))
            token_hits.append(float(((pred_tokens == target_tokens).float() * mask).sum().item() / mask.sum().item()))
            rows.extend(ratios(pred_hand, ref_hand, base_hand, mask))
    metrics = {
        "ce": float(np.mean(ce_values)),
        "token_acc": float(np.mean(token_hits)),
        "samples": len(rows),
    }
    for key in ["base_mae", "pred_mae", "base_speed", "pred_speed", "base_tstd", "pred_tstd"]:
        metrics[key] = summarize([row[key] for row in rows])
    metrics["mae_delta_pct"] = 100.0 * (
        metrics["pred_mae"]["mean"] - metrics["base_mae"]["mean"]
    ) / max(metrics["base_mae"]["mean"], 1e-9)
    return metrics


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.tokenizer_ckpt, device)
    dataset = T5ResidualTokenCacheDataset(args.cache_dir, max_samples=args.max_samples)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_count = max(1, int(round(len(indices) * args.val_ratio)))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = T5ResidualConfidenceTransformer(
        codebook_size=tokenizer.quantizer.codebook_size,
        hidden_features=args.hidden_features,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_val = None
    best_path = args.output_dir / "best_t5_residual_conf_transformer.pt"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            base_full = batch["base_full"].to(device)
            residual = batch["residual"].to(device)
            body_tokens = batch["body_tokens"].to(device)
            lhand_tokens = batch["lhand_tokens"].to(device)
            rhand_tokens = batch["rhand_tokens"].to(device)
            confidence = batch["confidence"].to(device)
            mask = batch["mask"].to(device)
            text_tokens = T5ResidualConfidenceTransformer.hash_text(batch["texts"], 4096, 96, device)
            with torch.no_grad():
                target_tokens = tokenizer.encode(residual)
            logits = model(base_full, body_tokens, lhand_tokens, rhand_tokens, confidence, text_tokens, mask)
            loss = masked_ce(logits, target_tokens, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate(model, tokenizer, val_loader, device)
        entry = {"epoch": epoch, "train_ce": float(np.mean(losses)), "val": val_metrics}
        history.append(entry)
        print(json.dumps(entry, indent=2))
        if best_val is None or val_metrics["ce"] < best_val:
            best_val = val_metrics["ce"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": {
                        "hidden_features": args.hidden_features,
                        "layers": args.layers,
                        "heads": args.heads,
                        "tokenizer_ckpt": str(args.tokenizer_ckpt),
                    },
                    "epoch": epoch,
                    "val": val_metrics,
                },
                best_path,
            )

    train_metrics = evaluate(model, tokenizer, train_loader, device)
    val_metrics = evaluate(model, tokenizer, val_loader, device)
    report = {
        "samples": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "history": history,
        "final_train": train_metrics,
        "final_val": val_metrics,
        "best_checkpoint": str(best_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# T5-1D residual confidence-aware transformer",
        "",
        f"Samples: **{len(dataset)}** train={len(train_indices)} val={len(val_indices)}",
        "",
        "| Split | Token acc ↑ | Base MAE ↓ | T5-1D MAE ↓ | Δ MAE | Base speed/GT | T5-1D speed/GT | Base tstd/GT | T5-1D tstd/GT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
        md.append(
            f"| {split} | {metrics['token_acc']:.3f} | "
            f"{metrics['base_mae']['mean']:.4f} | {metrics['pred_mae']['mean']:.4f} | "
            f"{metrics['mae_delta_pct']:+.2f}% | "
            f"{metrics['base_speed']['mean']:.3f} | {metrics['pred_speed']['mean']:.3f} | "
            f"{metrics['base_tstd']['mean']:.3f} | {metrics['pred_tstd']['mean']:.3f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
