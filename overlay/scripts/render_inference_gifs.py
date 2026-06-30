#!/usr/bin/env python3
"""Render best/worst skeleton GIFs (GT | Pred) from a test_swissft_*.yaml run.

For each dataset (phoenix, csl), reads:
  results/mgpt/<NAME>/test_rank_0/test_scores.json   (per-sample metrics)
  results/mgpt/<NAME>/test_rank_0/<sample>.pkl       (feats_rst, feats_ref, text)
Sorts by *_DTW_PA_JPE_body (low=best), then dumps:
  visualize/<NAME>/<rank2>_<TAG>_PA{body:.2f}_<sample>.gif
where TAG = BEST or WORST.

Runs inside the Docker container (has torch, smplx, mGPT modules).
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

# silence pyrender etc
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from mGPT.render.matplot.plot_3d_global import plot_3d_motion
from mGPT.utils.human_models import get_coord


SHAPE_PARAM = torch.tensor(
    [[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
      0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]],
    dtype=torch.float32,
)


def load_mean_std(mean_path: str, std_path: str, device):
    mean = torch.load(mean_path, map_location=device).float()
    std = torch.load(std_path, map_location=device).float()
    # Slice in stesso modo di preview_test_sample.py / vis_mesh.py
    mean = mean[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std = std[(3 + 3 * 11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
    return mean.to(device), std.to(device)


def feats_to_joints22(feats_np: np.ndarray, mean, std, device):
    feats = torch.from_numpy(feats_np).float().to(device)
    T = feats.shape[0]
    feats = feats * std.unsqueeze(0) + mean.unsqueeze(0)

    zero_pose = torch.zeros((T, 36), device=device)
    shape = SHAPE_PARAM.to(device).repeat(T, 1)
    full = torch.cat([zero_pose, feats], dim=-1)  # [T, 169]

    _, joints = get_coord(
        root_pose=full[..., 0:3],
        body_pose=full[..., 3:66],
        lhand_pose=full[..., 66:111],
        rhand_pose=full[..., 111:156],
        jaw_pose=full[..., 156:159],
        shape=shape,
        expr=full[..., 159:169],
    )
    return joints[:, :22, :].detach().cpu().numpy()


def render_skeleton_gif(joints: np.ndarray, title: str, out_path: str, fps: int):
    frames_rgba = plot_3d_motion((joints, None, title), fps=fps)
    frames = frames_rgba[..., :3].cpu().numpy().astype(np.uint8)
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(out_path, list(frames), duration=duration)
    return frames


def concat_side_by_side(frames_a: np.ndarray, frames_b: np.ndarray):
    T = max(len(frames_a), len(frames_b))
    if len(frames_a) < T:
        pad = np.repeat(frames_a[-1][None], T - len(frames_a), axis=0)
        frames_a = np.concatenate([frames_a, pad], axis=0)
    if len(frames_b) < T:
        pad = np.repeat(frames_b[-1][None], T - len(frames_b), axis=0)
        frames_b = np.concatenate([frames_b, pad], axis=0)
    return np.concatenate([frames_a, frames_b], axis=2)


def pick_metric_key(dataset: str) -> str:
    return f"{dataset}_DTW_PA_JPE_body"


def render_one(pkl_path: Path, mean, std, device, fps: int, label: str):
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)
    feats_rst = pkl["feats_rst"]
    feats_ref = pkl["feats_ref"]
    text = pkl.get("text", "")
    title_pred = f"PRED | {label} | {text[:60]}"
    title_gt = f"GT   | {label} | {text[:60]}"
    pred_joints = feats_to_joints22(feats_rst, mean, std, device)
    ref_joints = feats_to_joints22(feats_ref, mean, std, device)
    # Render separately to PNG-RGB tensors, then concat
    pred_frames_rgba = plot_3d_motion((pred_joints, None, title_pred), fps=fps)
    gt_frames_rgba = plot_3d_motion((ref_joints, None, title_gt), fps=fps)
    pred_frames = pred_frames_rgba[..., :3].cpu().numpy().astype(np.uint8)
    gt_frames = gt_frames_rgba[..., :3].cpu().numpy().astype(np.uint8)
    combined = concat_side_by_side(gt_frames, pred_frames)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True,
                        help="e.g. SOKE_INFER_SWISSFT_PHX50")
    parser.add_argument("--dataset", required=True, choices=["phoenix", "csl", "swissl", "how2sign"])
    parser.add_argument("--results_root", default="results/mgpt")
    parser.add_argument("--mean_path", default="datasets/CSL-Daily/mean.pt")
    parser.add_argument("--std_path", default="datasets/CSL-Daily/std.pt")
    parser.add_argument("--out_root", default="visualize/inference_gifs")
    parser.add_argument("--top_k", type=int, default=10,
                        help="how many best and how many worst to render")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    mean, std = load_mean_std(args.mean_path, args.std_path, device)

    pred_dir = Path(args.results_root) / args.run_name / "test_rank_0"
    scores_path = pred_dir / "test_scores.json"
    if not scores_path.exists():
        sys.exit(f"Missing scores: {scores_path}")

    scores = json.loads(scores_path.read_text())
    metric_key = pick_metric_key(args.dataset)

    rows = []
    for name, m in scores.items():
        v = m.get(metric_key)
        if v is None:
            continue
        # sample name in scores has prefix like "test/...", strip
        leaf = name.split("/")[-1]
        pkl = pred_dir / f"{leaf}.pkl"
        if not pkl.exists():
            print(f"[WARN] missing pkl for {leaf}, skipping")
            continue
        # also harvest hand PA for context
        pa_hand_l = m.get(f"{args.dataset}_DTW_PA_JPE_lhand", float("nan"))
        pa_hand_r = m.get(f"{args.dataset}_DTW_PA_JPE_rhand", float("nan"))
        rows.append((leaf, float(v), float(pa_hand_l), float(pa_hand_r), pkl))

    if not rows:
        sys.exit(f"No samples with metric {metric_key} found")

    rows.sort(key=lambda r: r[1])  # ascending: smaller=better
    n = len(rows)
    k = min(args.top_k, n // 2)
    best = rows[:k]
    worst = rows[-k:][::-1]

    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a small markdown ranking summary
    md = ["# Ranking — " + args.run_name, "",
          f"Metric: **{metric_key}** (mm, lower=better).",
          f"Samples evaluated: **{n}**", "",
          "| Rank | Sample | PA-body | PA-lhand | PA-rhand |",
          "|---:|---|---:|---:|---:|"]
    for i, (leaf, body, hl, hr, _pkl) in enumerate(rows):
        md.append(f"| {i+1} | {leaf} | {body:.2f} | {hl:.2f} | {hr:.2f} |")
    (out_dir / "ranking.md").write_text("\n".join(md))

    print(f"[INFO] {args.run_name}: n={n}, will render best {k} and worst {k}")
    print(f"[INFO] Output dir: {out_dir}")

    def render_group(group, tag):
        for rank0, (leaf, body, hl, hr, pkl) in enumerate(group):
            label = f"{tag} #{rank0+1} | PA-body {body:.2f}mm"
            out_path = out_dir / f"{tag.lower()}_{rank0+1:02d}_{leaf}_PAb{body:.2f}.gif"
            try:
                with open(pkl, "rb") as f:
                    payload = pickle.load(f)
                text = payload.get("text", "")
                feats_rst = payload["feats_rst"]
                feats_ref = payload["feats_ref"]
                pred_joints = feats_to_joints22(feats_rst, mean, std, device)
                ref_joints = feats_to_joints22(feats_ref, mean, std, device)
                title_pred = f"PRED {tag} #{rank0+1} | PA-body {body:.2f}mm | {text[:50]}"
                title_gt = f"GT {tag} #{rank0+1} | {text[:50]}"
                pred_frames_rgba = plot_3d_motion((pred_joints, None, title_pred), fps=args.fps)
                gt_frames_rgba = plot_3d_motion((ref_joints, None, title_gt), fps=args.fps)
                pred_frames = pred_frames_rgba[..., :3].cpu().numpy().astype(np.uint8)
                gt_frames = gt_frames_rgba[..., :3].cpu().numpy().astype(np.uint8)
                combined = concat_side_by_side(gt_frames, pred_frames)
                duration = 1.0 / max(args.fps, 1)
                imageio.mimsave(str(out_path), list(combined), duration=duration)
                print(f"[OK] {tag} #{rank0+1:02d} PA-body={body:.2f} text={text[:30]!r} → {out_path.name}")
            except Exception as e:
                print(f"[ERR] {tag} #{rank0+1}: {e}")

    render_group(best, "BEST")
    render_group(worst, "WORST")
    print(f"[DONE] {args.run_name}")


if __name__ == "__main__":
    main()
