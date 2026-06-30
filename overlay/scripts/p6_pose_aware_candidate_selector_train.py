#!/usr/bin/env python3
"""Train P6-K pose-aware selector with online P6-G oracle labels."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from mGPT.models.utils.p6_topk_candidate_selector import P6TopKCandidateSelector
from scripts.p6_geometric_candidate_selector_eval import candidate_features, mean_abs_velocity
from scripts.p6_hand_gain_regressor_budget_sweep import select_top_budget
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_token_edit_oracle_ceiling import load_vq_pack, normalize_token_shape
from scripts.p6_topk_candidate_selector_train import (
    candidate_gain,
    load_regressor,
    p6d_scores,
    topk_with_probs,
)
from scripts.t5_residual_tokenizer_oracle import resample_time


LH_START = 30
LH_END = 75
RH_START = 75
RH_END = 120
META_DIM = 16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--val-cache-dir", type=Path, default=None)
    parser.add_argument("--p6b-checkpoint", required=True, type=Path)
    parser.add_argument("--regressor-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/soke.yaml", type=Path)
    parser.add_argument("--default-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--vae-ckpt", default="deps/tokenizer_ckpt/tokenizer.ckpt", type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-scale", type=float, default=10.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=0.20)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_meta(conf, prob, score, progress, token, orig, rank, topk, geom, base_vel):
    return np.asarray(
        [
            conf[0],
            conf[1],
            conf[2],
            float(prob),
            float(score),
            float(progress),
            1.0 if int(token) == int(orig) else 0.0,
            1.0 if int(rank) == 0 else 0.0,
            float(rank) / max(1, topk),
            float(token != orig),
            float(geom["logprob"]),
            float(geom["jump"]),
            float(geom["accel"]),
            float(geom["change"]),
            float(geom["vitality"]),
            float(base_vel),
        ],
        dtype=np.float32,
    )


def local_base_velocity(base_hand, index):
    start = int(index) * 4
    end = min((int(index) + 1) * 4, len(base_hand))
    if end <= start:
        return 0.0
    return mean_abs_velocity(base_hand[start:end])


def build_group(side, index, body, lhand, rhand, conf, top_tokens, top_probs, scores, base, ref, pack, device, target_scale):
    if side == 0:
        vae = pack.hand_vae
        orig = lhand
        other = rhand
        base_hand = base[:, LH_START:LH_END]
        ref_hand = ref[:, LH_START:LH_END]
    else:
        vae = pack.rhand_vae
        orig = rhand
        other = lhand
        base_hand = base[:, RH_START:RH_END]
        ref_hand = ref[:, RH_START:RH_END]
    progress = index / max(1, len(orig) - 1)
    candidates = list(top_tokens[index]) + [int(orig[index])]
    probs = list(top_probs[index]) + [1.0]
    group = []
    base_vel = local_base_velocity(base_hand, index)
    for rank, (token, prob) in enumerate(zip(candidates, probs)):
        gain = candidate_gain(vae, orig, int(token), index, base_hand, ref_hand, len(base), device)
        geom = candidate_features(vae, orig, int(token), float(prob), index, base_hand, len(base), device)
        group.append(
            {
                "body": int(body[index]),
                "orig": int(orig[index]),
                "other": int(other[index]),
                "candidate": int(token),
                "side": int(side),
                "rank": int(rank),
                "meta": build_meta(conf[index], prob, scores[index], progress, token, orig[index], rank, len(top_tokens[index]), geom, base_vel),
                "target": float(gain * target_scale),
                "raw_gain": float(gain),
            }
        )
    target = np.asarray([item["target"] for item in group], dtype=np.float32)
    return {
        "body": np.asarray([item["body"] for item in group], dtype=np.int64),
        "orig": np.asarray([item["orig"] for item in group], dtype=np.int64),
        "other": np.asarray([item["other"] for item in group], dtype=np.int64),
        "candidate": np.asarray([item["candidate"] for item in group], dtype=np.int64),
        "side": np.asarray([item["side"] for item in group], dtype=np.int64),
        "rank": np.asarray([item["rank"] for item in group], dtype=np.int64),
        "meta": np.stack([item["meta"] for item in group]).astype(np.float32),
        "target": target,
        "raw_gain": np.asarray([item["raw_gain"] for item in group], dtype=np.float32),
        "best": int(np.argmax(target)),
    }


class P6PoseAwareSelectorDataset(Dataset):
    def __init__(self, cache_dir, editor, regressor, pack, args, device, label="train"):
        self.groups = []
        self.sample_names = []
        paths = sorted(cache_dir.glob("*.pkl"))
        if args.max_samples:
            paths = paths[: args.max_samples]
        total_paths = len(paths)
        start_time = time.time()
        progress_every = max(0, int(args.progress_every))
        print(
            f"[P6-K dataset:{label}] start cache={cache_dir} samples={total_paths} "
            f"topk={args.topk} budget={args.budget}",
            flush=True,
        )
        for path_index, path in enumerate(paths, start=1):
            sample_name = path.stem
            payload = load_payload(path)
            base = np.asarray(payload["feats_rst"], dtype=np.float32)
            ref = np.asarray(payload["feats_ref"], dtype=np.float32)
            if len(base) < 2 or len(ref) < 2:
                if progress_every and path_index % progress_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[P6-K dataset:{label}] {path_index}/{total_paths} samples "
                        f"groups={len(self.groups)} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                continue
            token_len = min(
                len(normalize_token_shape(payload["tokens_lhand"])),
                len(normalize_token_shape(payload["tokens_rhand"])),
            )
            if token_len < 2:
                if progress_every and path_index % progress_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[P6-K dataset:{label}] {path_index}/{total_paths} samples "
                        f"groups={len(self.groups)} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                continue
            ref_aligned = resample_time(ref, len(base)).astype(np.float32)
            body, lhand, rhand, conf, l_top, r_top, l_probs, r_probs = topk_with_probs(
                editor, payload, token_len, args.topk, device
            )
            _body, _lhand, _rhand, _conf, _cl, _cr, _lc, _rc, l_score, r_score = p6d_scores(
                editor, regressor, payload, token_len, device
            )
            for side, selected, top_tokens, top_probs, scores in [
                (0, select_top_budget(l_score, args.budget), l_top, l_probs, l_score),
                (1, select_top_budget(r_score, args.budget), r_top, r_probs, r_score),
            ]:
                for index in np.where(selected)[0]:
                    group = build_group(
                        side,
                        int(index),
                        body,
                        lhand,
                        rhand,
                        conf,
                        top_tokens,
                        top_probs,
                        scores,
                        base,
                        ref_aligned,
                        pack,
                        device,
                        args.target_scale,
                    )
                    group["sample_name"] = sample_name
                    group["token_index"] = int(index)
                    self.groups.append(group)
                    self.sample_names.append(sample_name)
            if progress_every and (path_index % progress_every == 0 or path_index == total_paths):
                elapsed = time.time() - start_time
                rate = path_index / max(elapsed, 1e-9)
                eta = (total_paths - path_index) / max(rate, 1e-9)
                print(
                    f"[P6-K dataset:{label}] {path_index}/{total_paths} samples "
                    f"groups={len(self.groups)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if not self.groups:
            raise RuntimeError("No P6-K groups found")
        elapsed = time.time() - start_time
        print(
            f"[P6-K dataset:{label}] done samples={total_paths} groups={len(self.groups)} "
            f"unique_samples={len(set(self.sample_names))} elapsed={elapsed:.1f}s",
            flush=True,
        )

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return self.groups[index]


def collate(batch):
    return {
        "body": torch.tensor(np.stack([item["body"] for item in batch]), dtype=torch.long),
        "orig": torch.tensor(np.stack([item["orig"] for item in batch]), dtype=torch.long),
        "other": torch.tensor(np.stack([item["other"] for item in batch]), dtype=torch.long),
        "candidate": torch.tensor(np.stack([item["candidate"] for item in batch]), dtype=torch.long),
        "side": torch.tensor(np.stack([item["side"] for item in batch]), dtype=torch.long),
        "rank": torch.tensor(np.stack([item["rank"] for item in batch]), dtype=torch.long),
        "meta": torch.tensor(np.stack([item["meta"] for item in batch]), dtype=torch.float32),
        "target": torch.tensor(np.stack([item["target"] for item in batch]), dtype=torch.float32),
        "raw_gain": torch.tensor(np.stack([item["raw_gain"] for item in batch]), dtype=torch.float32),
        "best": torch.tensor([item["best"] for item in batch], dtype=torch.long),
    }


def forward_group(model, batch):
    batch_size, group_size = batch["body"].shape
    scores = model(
        batch["body"].reshape(-1),
        batch["orig"].reshape(-1),
        batch["other"].reshape(-1),
        batch["candidate"].reshape(-1),
        batch["side"].reshape(-1),
        batch["rank"].reshape(-1),
        batch["meta"].reshape(batch_size * group_size, -1),
    )
    return scores.reshape(batch_size, group_size)


def group_loss(scores, target, raw_gain, best, ce_weight, reg_weight, positive_weight):
    ce = F.cross_entropy(scores, best)
    weight = torch.where(raw_gain > 0.0, torch.full_like(raw_gain, float(positive_weight)), torch.ones_like(raw_gain))
    reg = F.smooth_l1_loss(scores, target, reduction="none")
    reg = (reg * weight).sum() / weight.sum().clamp_min(1.0)
    return float(ce_weight) * ce + float(reg_weight) * reg


def evaluate(model, loader, device, args):
    model.eval()
    losses = []
    accs = []
    regrets = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = forward_group(model, batch)
            loss = group_loss(scores, batch["target"], batch["raw_gain"], batch["best"], args.ce_weight, args.reg_weight, args.positive_weight)
            pred = scores.argmax(dim=-1)
            best_gain = batch["target"].gather(1, batch["best"].unsqueeze(1)).squeeze(1)
            pred_gain = batch["target"].gather(1, pred.unsqueeze(1)).squeeze(1)
            losses.append(float(loss.item()))
            accs.append(float((pred == batch["best"]).float().mean().item()))
            regrets.append(float((best_gain - pred_gain).mean().item()))
    return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs)), "regret": float(np.mean(regrets))}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    pack = load_vq_pack(args, device)
    editor, editor_args = load_p6_editor(args.p6b_checkpoint, device)
    regressor = load_regressor(args.regressor_checkpoint, device)
    dataset = P6PoseAwareSelectorDataset(args.cache_dir, editor, regressor, pack, args, device, label="train")
    val_dataset = None
    if args.val_cache_dir is not None:
        val_dataset = P6PoseAwareSelectorDataset(args.val_cache_dir, editor, regressor, pack, args, device, label="val")
        train_names = set(dataset.sample_names)
        val_names = set(val_dataset.sample_names)
        train_indices = list(range(len(dataset)))
        val_indices = list(range(len(val_dataset)))
        split_source = "separate_cache"
    else:
        unique_names = sorted(set(dataset.sample_names))
        random.shuffle(unique_names)
        val_sample_count = max(1, int(round(len(unique_names) * args.val_ratio)))
        val_names = set(unique_names[:val_sample_count])
        train_names = set(unique_names[val_sample_count:])
        train_indices = [index for index, name in enumerate(dataset.sample_names) if name in train_names]
        val_indices = [index for index, name in enumerate(dataset.sample_names) if name in val_names]
        split_source = "sample_split_within_cache"
    if not train_indices or not val_indices:
        raise RuntimeError("Sample-level split produced an empty train or val set")
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        Subset(val_dataset if val_dataset is not None else dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = P6TopKCandidateSelector(
        hidden_features=args.hidden_features,
        dropout=args.dropout,
        meta_dim=META_DIM,
        max_rank=args.topk,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score = None
    best_path = args.output_dir / "best_p6_pose_aware_candidate_selector.pt"
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = forward_group(model, batch)
            loss = group_loss(scores, batch["target"], batch["raw_gain"], batch["best"], args.ce_weight, args.reg_weight, args.positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val = evaluate(model, val_loader, device, args)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val}
        history.append(entry)
        print(json.dumps(entry, indent=2))
        score = -val["regret"]
        if best_score is None or score > best_score:
            best_score = score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": {
                        "hidden_features": args.hidden_features,
                        "dropout": args.dropout,
                        "meta_dim": META_DIM,
                        "topk": args.topk,
                        "budget": args.budget,
                        "target_scale": args.target_scale,
                        "ce_weight": args.ce_weight,
                        "reg_weight": args.reg_weight,
                        "positive_weight": args.positive_weight,
                        "p6b_checkpoint": str(args.p6b_checkpoint),
                        "regressor_checkpoint": str(args.regressor_checkpoint),
                        "cache_dir": str(args.cache_dir),
                        "val_cache_dir": str(args.val_cache_dir) if args.val_cache_dir else None,
                        "split_source": split_source,
                        "split_unit": "sample",
                        "train_samples": len(train_names),
                        "val_samples": len(val_names),
                    },
                    "epoch": epoch,
                    "val": val,
                    "editor_args": editor_args,
                },
                best_path,
            )
    train_metrics = evaluate(model, train_loader, device, args)
    val_metrics = evaluate(model, val_loader, device, args)
    report = {
        "groups": len(dataset),
        "train_groups": len(train_indices),
        "val_groups": len(val_indices),
        "samples": len(set(dataset.sample_names)),
        "train_samples": len(train_names),
        "val_samples": len(val_names),
        "val_cache_groups": len(val_dataset) if val_dataset is not None else None,
        "val_cache_samples": len(set(val_dataset.sample_names)) if val_dataset is not None else None,
        "split_source": split_source,
        "split_unit": "sample",
        "train_val_sample_overlap": len(train_names & val_names),
        "history": history,
        "final_train": train_metrics,
        "final_val": val_metrics,
        "best_checkpoint": str(best_path),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# P6-K pose-aware candidate selector train",
        "",
        f"Groups: **{len(dataset)}** train={len(train_indices)} val={len(val_indices)}",
        f"Samples: **{len(set(dataset.sample_names))}** train={len(train_names)} val={len(val_names)} overlap={len(train_names & val_names)}",
        f"Split source: **{split_source}**",
        "Split unit: **sample/frase**",
        f"Best checkpoint: `{best_path}`",
        "",
        "| Split | Loss ↓ | Acc ↑ | Regret ↓ |",
        "|---|---:|---:|---:|",
        f"| train | {train_metrics['loss']:.4f} | {train_metrics['acc']:.3f} | {train_metrics['regret']:.4f} |",
        f"| val | {val_metrics['loss']:.4f} | {val_metrics['acc']:.3f} | {val_metrics['regret']:.4f} |",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
