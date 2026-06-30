#!/usr/bin/env python3
"""Train P6-C learned KEEP/REPLACE gain gate for P6-B token candidates."""

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

from mGPT.models.utils.p6_hand_gain_gate import P6HandGainGate
from mGPT.models.utils.p6_hand_token_editor import P6HandTokenEditor
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_hand_token_editor_paired_eval import token_gain
from scripts.p6_hand_token_editor_train import pad_or_crop_float
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
    parser.add_argument("--p6b-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/soke.yaml", type=Path)
    parser.add_argument("--default-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--vae-ckpt", default="deps/tokenizer_ckpt/tokenizer.ckpt", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-features", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_candidates(editor, payload, token_len, device):
    body = pad_or_crop_tokens(payload["tokens_body"], token_len)
    lhand = pad_or_crop_tokens(payload["tokens_lhand"], token_len)
    rhand = pad_or_crop_tokens(payload["tokens_rhand"], token_len)
    conf = np.concatenate(
        [
            pad_or_crop_float(payload["confidence_body"], token_len),
            pad_or_crop_float(payload["confidence_lhand"], token_len),
            pad_or_crop_float(payload["confidence_rhand"], token_len),
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
        l_new = l_prob.argmax(dim=-1).detach().cpu().numpy()[0].astype(np.int64)
        r_new = r_prob.argmax(dim=-1).detach().cpu().numpy()[0].astype(np.int64)
        l_model_conf = l_prob.max(dim=-1).values.detach().cpu().numpy()[0].astype(np.float32)
        r_model_conf = r_prob.max(dim=-1).values.detach().cpu().numpy()[0].astype(np.float32)
    return body, lhand, rhand, l_new, r_new, conf, l_model_conf, r_model_conf


class P6GainGateDataset(Dataset):
    def __init__(self, cache_dir, editor, pack, device, max_samples=0):
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
            # Replace confidence body with normalized temporal position in an
            # extra channel without changing model shape: append by overwriting
            # the least useful meta column would be brittle, so keep progress as
            # an additive signal in confidence body scaling.
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
                    "target_lhand": (l_gain > 0.0).astype(np.float32),
                    "target_rhand": (r_gain > 0.0).astype(np.float32),
                }
            )
        if not self.samples:
            raise RuntimeError("No P6-C samples found")

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
        for key in ["meta", "target_lhand", "target_rhand"]:
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


def masked_bce(logits, target, mask, pos_weight):
    weight = torch.ones_like(target)
    weight = torch.where(target > 0.5, weight * float(pos_weight), weight)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * weight * mask
    return loss.sum() / (weight * mask).sum().clamp_min(1.0)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values))}


def evaluate(model, loader, device, pos_weight):
    model.eval()
    losses = []
    l_acc = []
    r_acc = []
    l_pos_rate = []
    r_pos_rate = []
    l_pred_rate = []
    r_pred_rate = []
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
            mask = batch["mask"].to(device)
            text = P6HandGainGate.hash_text(batch["texts"], 4096, 96, device)
            l_logits, r_logits = model(body, lhand, rhand, cand_lhand, cand_rhand, meta, text, mask)
            l_loss = masked_bce(l_logits, target_lhand, mask, pos_weight)
            r_loss = masked_bce(r_logits, target_rhand, mask, pos_weight)
            losses.append(float((l_loss + r_loss).item()))
            l_pred = torch.sigmoid(l_logits) > 0.5
            r_pred = torch.sigmoid(r_logits) > 0.5
            denom = mask.sum().clamp_min(1.0)
            l_acc.append(float((((l_pred == (target_lhand > 0.5)).float() * mask).sum() / denom).item()))
            r_acc.append(float((((r_pred == (target_rhand > 0.5)).float() * mask).sum() / denom).item()))
            l_pos_rate.append(float(((target_lhand * mask).sum() / denom).item()))
            r_pos_rate.append(float(((target_rhand * mask).sum() / denom).item()))
            l_pred_rate.append(float(((l_pred.float() * mask).sum() / denom).item()))
            r_pred_rate.append(float(((r_pred.float() * mask).sum() / denom).item()))
    return {
        "loss": float(np.mean(losses)),
        "lhand_acc": summarize(l_acc),
        "rhand_acc": summarize(r_acc),
        "lhand_pos_rate": summarize(l_pos_rate),
        "rhand_pos_rate": summarize(r_pos_rate),
        "lhand_pred_rate": summarize(l_pred_rate),
        "rhand_pred_rate": summarize(r_pred_rate),
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
    dataset = P6GainGateDataset(args.cache_dir, editor, pack, device, max_samples=args.max_samples)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_count = max(1, int(round(len(indices) * args.val_ratio)))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = P6HandGainGate(
        hidden_features=args.hidden_features,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_val = None
    best_path = args.output_dir / "best_p6_hand_gain_gate.pt"
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
            mask = batch["mask"].to(device)
            text = P6HandGainGate.hash_text(batch["texts"], 4096, 96, device)
            l_logits, r_logits = model(body, lhand, rhand, cand_lhand, cand_rhand, meta, text, mask)
            loss = masked_bce(l_logits, target_lhand, mask, args.pos_weight)
            loss = loss + masked_bce(r_logits, target_rhand, mask, args.pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate(model, val_loader, device, args.pos_weight)
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
                        "pos_weight": args.pos_weight,
                        "p6b_checkpoint": str(args.p6b_checkpoint),
                        "cache_dir": str(args.cache_dir),
                    },
                    "epoch": epoch,
                    "val": val_metrics,
                },
                best_path,
            )
    train_metrics = evaluate(model, train_loader, device, args.pos_weight)
    val_metrics = evaluate(model, val_loader, device, args.pos_weight)
    report = {
        "samples": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "history": history,
        "final_train": train_metrics,
        "final_val": val_metrics,
        "best_checkpoint": str(best_path),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# P6-C hand gain gate train",
        "",
        f"Samples: **{len(dataset)}** train={len(train_indices)} val={len(val_indices)}",
        f"Best checkpoint: `{best_path}`",
        "",
        "| Split | Loss ↓ | LH acc ↑ | RH acc ↑ | LH pos | RH pos | LH pred | RH pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
        md.append(
            f"| {split} | {metrics['loss']:.4f} | "
            f"{metrics['lhand_acc']['mean']:.3f} | {metrics['rhand_acc']['mean']:.3f} | "
            f"{metrics['lhand_pos_rate']['mean']:.3f} | {metrics['rhand_pos_rate']['mean']:.3f} | "
            f"{metrics['lhand_pred_rate']['mean']:.3f} | {metrics['rhand_pred_rate']['mean']:.3f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
