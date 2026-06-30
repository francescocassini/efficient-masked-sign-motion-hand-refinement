#!/usr/bin/env python3
"""T5-0 residual tokenizer oracle.

This standalone gate trains a second residual codebook on hand deltas computed
from fixed P5 predictions. It intentionally does not touch P3/P5/T4 code or
checkpoints. The goal is to verify whether an RVQ-style residual level can
represent useful hand detail before training any residual transformer.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from mGPT.models.utils.t5_residual_vq import T5ResidualVQVAE


HAND_START = 30
HAND_END = 120


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p5-dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--code-dim", type=int, default=256)
    parser.add_argument("--max-residual", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--final-weight", type=float, default=1.0)
    parser.add_argument("--vq-weight", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def resample_time(sequence: np.ndarray, length: int) -> np.ndarray:
    """Linear-resample [T,C] to target length."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) == length:
        return sequence
    if len(sequence) == 0:
        return np.zeros((length, sequence.shape[-1]), dtype=np.float32)
    if len(sequence) == 1:
        return np.repeat(sequence, length, axis=0)
    old_x = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    columns = [np.interp(new_x, old_x, sequence[:, dim]) for dim in range(sequence.shape[-1])]
    return np.stack(columns, axis=-1).astype(np.float32)


@dataclass
class SampleStats:
    name: str
    source_dir: str
    length_pred: int
    length_ref: int
    residual_abs_mean: float
    base_hand_mae: float


class ResidualHandDataset(Dataset):
    def __init__(self, p5_dirs: list[Path], max_samples: int = 0):
        self.samples = []
        for p5_dir in p5_dirs:
            for path in sorted(p5_dir.glob("*.pkl")):
                if path.name == "test_scores.json":
                    continue
                payload = load_payload(path)
                base = np.asarray(payload["feats_rst"], dtype=np.float32)
                ref = np.asarray(payload["feats_ref"], dtype=np.float32)
                if len(base) < 2 or len(ref) < 2:
                    continue
                ref_resampled = resample_time(ref, len(base))
                base_hand = base[:, HAND_START:HAND_END]
                ref_hand = ref_resampled[:, HAND_START:HAND_END]
                residual = ref_hand - base_hand
                self.samples.append(
                    {
                        "name": path.stem,
                        "source_dir": p5_dir.name,
                        "base_hand": base_hand.astype(np.float32),
                        "ref_hand": ref_hand.astype(np.float32),
                        "residual": residual.astype(np.float32),
                        "stats": SampleStats(
                            name=path.stem,
                            source_dir=p5_dir.name,
                            length_pred=len(base),
                            length_ref=len(ref),
                            residual_abs_mean=float(np.abs(residual).mean()),
                            base_hand_mae=float(np.abs(base_hand - ref_hand).mean()),
                        ),
                    }
                )
        if max_samples:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError("No usable P5 pkl samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(batch):
    max_len = max(item["residual"].shape[0] for item in batch)
    residual = np.zeros((len(batch), max_len, 90), dtype=np.float32)
    base = np.zeros((len(batch), max_len, 90), dtype=np.float32)
    ref = np.zeros((len(batch), max_len, 90), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=np.float32)
    names = []
    stats = []
    for index, item in enumerate(batch):
        length = item["residual"].shape[0]
        residual[index, :length] = item["residual"]
        base[index, :length] = item["base_hand"]
        ref[index, :length] = item["ref_hand"]
        if length < max_len:
            residual[index, length:] = item["residual"][-1]
            base[index, length:] = item["base_hand"][-1]
            ref[index, length:] = item["ref_hand"][-1]
        mask[index, :length] = 1.0
        names.append(item["name"])
        stats.append(item["stats"])
    return {
        "residual": torch.from_numpy(residual),
        "base": torch.from_numpy(base),
        "ref": torch.from_numpy(ref),
        "mask": torch.from_numpy(mask),
        "names": names,
        "stats": stats,
    }


def masked_l1(pred, target, mask):
    loss = (pred - target).abs().mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def velocity_loss(pred, target, mask):
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    pred_v = pred[:, 1:] - pred[:, :-1]
    target_v = target[:, 1:] - target[:, :-1]
    valid = mask[:, 1:] * mask[:, :-1]
    return masked_l1(pred_v, target_v, valid)


def sequence_ratios(pred, ref, base, mask):
    valid = mask.bool().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    ref_np = ref.detach().cpu().numpy()
    base_np = base.detach().cpu().numpy()
    rows = []
    for idx in range(pred_np.shape[0]):
        length = int(valid[idx].sum())
        if length < 2:
            continue
        p = pred_np[idx, :length]
        r = ref_np[idx, :length]
        b = base_np[idx, :length]
        def speed(x):
            return np.abs(np.diff(x, axis=0)).mean()
        rows.append(
            {
                "pred_mae": float(np.abs(p - r).mean()),
                "base_mae": float(np.abs(b - r).mean()),
                "pred_speed_ratio": float(speed(p) / max(speed(r), 1e-9)),
                "base_speed_ratio": float(speed(b) / max(speed(r), 1e-9)),
                "pred_tstd_ratio": float(p.std(axis=0).mean() / max(r.std(axis=0).mean(), 1e-9)),
                "base_tstd_ratio": float(b.std(axis=0).mean() / max(r.std(axis=0).mean(), 1e-9)),
            }
        )
    return rows


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def serializable_args(args):
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(item) if isinstance(item, Path) else item for item in value]
        elif isinstance(value, torch.device):
            payload[key] = str(value)
    return payload


def evaluate(model, loader, device):
    model.eval()
    rows = []
    token_hist = torch.zeros(model.quantizer.codebook_size, dtype=torch.long)
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            residual = batch["residual"].to(device)
            base = batch["base"].to(device)
            ref = batch["ref"].to(device)
            mask = batch["mask"].to(device)
            out = model(residual)
            recon = out["reconstruction"]
            final = base + recon
            loss = masked_l1(recon, residual, mask)
            total_loss += float(loss.item())
            total_batches += 1
            rows.extend(sequence_ratios(final, ref, base, mask))
            tokens = out["tokens"].detach().cpu().reshape(-1)
            token_hist += torch.bincount(tokens, minlength=model.quantizer.codebook_size)
    used = int((token_hist > 0).sum().item())
    total = int(token_hist.sum().item())
    probs = token_hist.float() / max(total, 1)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item())
    metrics = {
        "loss_reconstruction": total_loss / max(total_batches, 1),
        "samples": len(rows),
        "codebook_used": used,
        "codebook_size": model.quantizer.codebook_size,
        "codebook_usage_pct": 100.0 * used / model.quantizer.codebook_size,
        "codebook_entropy": entropy,
    }
    for key in [
        "base_mae",
        "pred_mae",
        "base_speed_ratio",
        "pred_speed_ratio",
        "base_tstd_ratio",
        "pred_tstd_ratio",
    ]:
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

    dataset = ResidualHandDataset(args.p5_dirs, max_samples=args.max_samples)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_count = max(1, int(round(len(indices) * args.val_ratio)))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    if not train_indices:
        raise RuntimeError("Not enough samples after validation split")

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    device = torch.device(args.device)
    model = T5ResidualVQVAE(
        hidden_features=args.hidden_features,
        codebook_size=args.codebook_size,
        code_dim=args.code_dim,
        max_residual=args.max_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    stats_payload = {
        "samples": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "p5_dirs": [str(path) for path in args.p5_dirs],
        "sample_stats_head": [asdict(item["stats"]) for item in dataset.samples[:20]],
    }
    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(stats_payload, indent=2),
        encoding="utf-8",
    )

    history = []
    best_val = None
    best_path = args.output_dir / "best_t5_residual_tokenizer.pt"
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            residual = batch["residual"].to(device)
            base = batch["base"].to(device)
            ref = batch["ref"].to(device)
            mask = batch["mask"].to(device)
            out = model(residual)
            recon = out["reconstruction"]
            final = base + recon
            loss_recon = masked_l1(recon, residual, mask)
            loss_final = masked_l1(final, ref, mask)
            loss_vel = velocity_loss(final, ref, mask)
            loss = (
                loss_recon
                + args.final_weight * loss_final
                + args.velocity_weight * loss_vel
                + args.vq_weight * out["vq_loss"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val": val_metrics,
        }
        history.append(entry)
        print(json.dumps(entry, indent=2))
        current = val_metrics["pred_mae"]["mean"]
        if best_val is None or current < best_val:
            best_val = current
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": serializable_args(args),
                    "epoch": epoch,
                    "val": val_metrics,
                },
                best_path,
            )

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    report = {
        "args": serializable_args(args),
        "dataset": stats_payload,
        "history": history,
        "final_train": train_metrics,
        "final_val": val_metrics,
        "best_checkpoint": str(best_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# T5-0 residual tokenizer oracle",
        "",
        f"Samples: **{len(dataset)}** train={len(train_indices)} val={len(val_indices)}",
        "",
        "| Split | Base MAE ↓ | T5 oracle MAE ↓ | Δ MAE | Base speed/GT | T5 speed/GT | Base tstd/GT | T5 tstd/GT | Codebook used |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
        lines.append(
            f"| {split} | {metrics['base_mae']['mean']:.4f} | "
            f"{metrics['pred_mae']['mean']:.4f} | {metrics['mae_delta_pct']:+.2f}% | "
            f"{metrics['base_speed_ratio']['mean']:.3f} | "
            f"{metrics['pred_speed_ratio']['mean']:.3f} | "
            f"{metrics['base_tstd_ratio']['mean']:.3f} | "
            f"{metrics['pred_tstd_ratio']['mean']:.3f} | "
            f"{metrics['codebook_used']}/{metrics['codebook_size']} |"
        )
    lines.append("")
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
