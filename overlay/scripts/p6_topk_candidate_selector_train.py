#!/usr/bin/env python3
"""Train P6-H selector for P6-B top-k hand-token candidates."""

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
from mGPT.models.utils.p6_topk_candidate_selector import P6TopKCandidateSelector
from scripts.p6_hand_gain_gate_train import predict_candidates
from scripts.p6_hand_gain_regressor_budget_sweep import select_top_budget
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_hand_topk_candidate_oracle import editor_topk, local_error
from scripts.p6_token_edit_oracle_ceiling import (
    decode_hand_tokens,
    load_vq_pack,
    normalize_token_shape,
)
from scripts.t5_residual_tokenizer_oracle import resample_time


LH_START = 30
LH_END = 75
RH_START = 75
RH_END = 120
META_DIM = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--p6b-checkpoint", required=True, type=Path)
    parser.add_argument("--regressor-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/soke.yaml", type=Path)
    parser.add_argument("--default-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--vae-ckpt", default="deps/tokenizer_ckpt/tokenizer.ckpt", type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-features", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-scale", type=float, default=10.0)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
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
    return model


def p6d_scores(editor, regressor, payload, token_len, device):
    body, lhand, rhand, cand_lhand, cand_rhand, conf, l_model_conf, r_model_conf = predict_candidates(
        editor, payload, token_len, device
    )
    progress = np.linspace(0.0, 1.0, token_len, dtype=np.float32)
    meta = np.concatenate(
        [
            conf,
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
    return (
        body,
        lhand,
        rhand,
        conf,
        cand_lhand,
        cand_rhand,
        l_model_conf,
        r_model_conf,
        l_score.detach().cpu().numpy()[0],
        r_score.detach().cpu().numpy()[0],
    )


def topk_with_probs(editor, payload, token_len, topk, device):
    body, lhand, rhand, conf, l_top, r_top = editor_topk(editor, payload, token_len, topk, device)
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
        l_values = torch.topk(l_prob, k=topk, dim=-1).values.detach().cpu().numpy()[0].astype(np.float32)
        r_values = torch.topk(r_prob, k=topk, dim=-1).values.detach().cpu().numpy()[0].astype(np.float32)
    return body, lhand, rhand, conf, l_top, r_top, l_values, r_values


def candidate_gain(vae, orig_tokens, candidate_token, index, base_hand, ref_hand, target_frames, device):
    base_err = local_error(base_hand, ref_hand, index)
    if int(candidate_token) == int(orig_tokens[index]):
        return 0.0
    trial = orig_tokens.copy()
    trial[index] = candidate_token
    decoded = decode_hand_tokens(vae, trial, target_frames, device)
    return base_err - local_error(decoded, ref_hand, index)


def add_rows(rows, side, selected, body, lhand, rhand, conf, top_tokens, top_probs, scores, base, ref, pack, device, target_scale):
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

    token_len = len(orig)
    for index in np.where(selected)[0]:
        progress = index / max(1, token_len - 1)
        candidates = list(top_tokens[index]) + [int(orig[index])]
        probs = list(top_probs[index]) + [1.0]
        ranks = list(range(len(top_tokens[index]))) + [len(top_tokens[index])]
        for rank, token, prob in zip(ranks, candidates, probs):
            gain = candidate_gain(vae, orig, int(token), index, base_hand, ref_hand, len(base), device)
            meta = np.asarray(
                [
                    conf[index, 0],
                    conf[index, 1],
                    conf[index, 2],
                    float(prob),
                    float(scores[index]),
                    progress,
                    1.0 if int(token) == int(orig[index]) else 0.0,
                    1.0 if int(rank) == 0 else 0.0,
                    float(rank) / max(1, len(top_tokens[index])),
                    float(token != orig[index]),
                ],
                dtype=np.float32,
            )
            rows.append(
                {
                    "body": int(body[index]),
                    "orig": int(orig[index]),
                    "other": int(other[index]),
                    "candidate": int(token),
                    "side": int(side),
                    "rank": int(rank),
                    "meta": meta,
                    "target": float(gain * target_scale),
                    "raw_gain": float(gain),
                }
            )


class P6TopKSelectorDataset(Dataset):
    def __init__(self, cache_dir, editor, regressor, pack, args, device):
        self.rows = []
        paths = sorted(cache_dir.glob("*.pkl"))
        if args.max_samples:
            paths = paths[: args.max_samples]
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
            ref_aligned = resample_time(ref, len(base)).astype(np.float32)
            body, lhand, rhand, conf, l_top, r_top, l_probs, r_probs = topk_with_probs(
                editor, payload, token_len, args.topk, device
            )
            _body, _lhand, _rhand, _conf, _cl, _cr, _lc, _rc, l_score, r_score = p6d_scores(
                editor, regressor, payload, token_len, device
            )
            l_sel = select_top_budget(l_score, args.budget)
            r_sel = select_top_budget(r_score, args.budget)
            add_rows(
                self.rows,
                0,
                l_sel,
                body,
                lhand,
                rhand,
                conf,
                l_top,
                l_probs,
                l_score,
                base,
                ref_aligned,
                pack,
                device,
                args.target_scale,
            )
            add_rows(
                self.rows,
                1,
                r_sel,
                body,
                lhand,
                rhand,
                conf,
                r_top,
                r_probs,
                r_score,
                base,
                ref_aligned,
                pack,
                device,
                args.target_scale,
            )
        if not self.rows:
            raise RuntimeError("No P6-H selector rows found")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(batch):
    out = {
        "body": torch.tensor([item["body"] for item in batch], dtype=torch.long),
        "orig": torch.tensor([item["orig"] for item in batch], dtype=torch.long),
        "other": torch.tensor([item["other"] for item in batch], dtype=torch.long),
        "candidate": torch.tensor([item["candidate"] for item in batch], dtype=torch.long),
        "side": torch.tensor([item["side"] for item in batch], dtype=torch.long),
        "rank": torch.tensor([item["rank"] for item in batch], dtype=torch.long),
        "meta": torch.tensor(np.stack([item["meta"] for item in batch]), dtype=torch.float32),
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
        "raw_gain": torch.tensor([item["raw_gain"] for item in batch], dtype=torch.float32),
    }
    return out


def weighted_smooth_l1(pred, target, raw_gain, positive_weight):
    weight = torch.ones_like(target)
    weight = torch.where(raw_gain > 0.0, weight * float(positive_weight), weight)
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weight
    return loss.sum() / weight.sum().clamp_min(1.0)


def corr(pred, target):
    if pred.numel() < 2:
        return 0.0
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float((pred * target).sum().div(denom).item())


def evaluate(model, loader, device, positive_weight):
    model.eval()
    losses = []
    corrs = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = model(batch["body"], batch["orig"], batch["other"], batch["candidate"], batch["side"], batch["rank"], batch["meta"])
            losses.append(float(weighted_smooth_l1(pred, batch["target"], batch["raw_gain"], positive_weight).item()))
            corrs.append(corr(pred, batch["target"]))
    return {"loss": float(np.mean(losses)), "corr": float(np.mean(corrs))}


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
    dataset = P6TopKSelectorDataset(args.cache_dir, editor, regressor, pack, args, device)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_count = max(1, int(round(len(indices) * args.val_ratio)))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = P6TopKCandidateSelector(
        hidden_features=args.hidden_features,
        dropout=args.dropout,
        meta_dim=META_DIM,
        max_rank=args.topk,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score = None
    best_path = args.output_dir / "best_p6_topk_candidate_selector.pt"
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = model(batch["body"], batch["orig"], batch["other"], batch["candidate"], batch["side"], batch["rank"], batch["meta"])
            loss = weighted_smooth_l1(pred, batch["target"], batch["raw_gain"], args.positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val = evaluate(model, val_loader, device, args.positive_weight)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val}
        history.append(entry)
        print(json.dumps(entry, indent=2))
        if best_score is None or val["corr"] > best_score:
            best_score = val["corr"]
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
                        "positive_weight": args.positive_weight,
                        "p6b_checkpoint": str(args.p6b_checkpoint),
                        "regressor_checkpoint": str(args.regressor_checkpoint),
                        "cache_dir": str(args.cache_dir),
                    },
                    "epoch": epoch,
                    "val": val,
                    "editor_args": editor_args,
                },
                best_path,
            )
    train_metrics = evaluate(model, train_loader, device, args.positive_weight)
    val_metrics = evaluate(model, val_loader, device, args.positive_weight)
    report = {
        "rows": len(dataset),
        "train_rows": len(train_indices),
        "val_rows": len(val_indices),
        "history": history,
        "final_train": train_metrics,
        "final_val": val_metrics,
        "best_checkpoint": str(best_path),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# P6-H top-k candidate selector train",
        "",
        f"Rows: **{len(dataset)}** train={len(train_indices)} val={len(val_indices)}",
        f"Best checkpoint: `{best_path}`",
        "",
        "| Split | Loss ↓ | Corr ↑ |",
        "|---|---:|---:|",
        f"| train | {train_metrics['loss']:.4f} | {train_metrics['corr']:.3f} |",
        f"| val | {val_metrics['loss']:.4f} | {val_metrics['corr']:.3f} |",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
