#!/usr/bin/env python3
"""Audit motion vitality and diversity for a saved prediction run."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from mGPT.utils.human_models import get_coord, smpl_x


SHAPE_PARAM = torch.tensor(
    [[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
      0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]],
    dtype=torch.float32,
)
GROUPS = {
    "torso/legs/head": list(range(0, 16)),
    "arms": [16, 17, 18, 19, 20, 21],
    "lhand fingers": list(range(25, 40)),
    "rhand fingers": list(range(40, 55)),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-name", default="CANDIDATE")
    parser.add_argument("--mean", default="datasets/CSL-Daily/mean.pt", type=Path)
    parser.add_argument("--std", default="datasets/CSL-Daily/std.pt", type=Path)
    parser.add_argument("--diversity-samples", default=200, type=int)
    return parser.parse_args()


def load_mean_std(mean_path, std_path, device):
    mean = torch.load(mean_path, map_location=device).float()
    std = torch.load(std_path, map_location=device).float()
    mean = mean[(3 + 3 * 11):]
    std = std[(3 + 3 * 11):]
    return (
        torch.cat([mean[:-20], mean[-10:]], dim=0).to(device),
        torch.cat([std[:-20], std[-10:]], dim=0).to(device),
    )


def feats_to_joints(feats, mean, std, device):
    feats = torch.from_numpy(np.ascontiguousarray(feats)).float().to(device)
    feats = feats * std.unsqueeze(0) + mean.unsqueeze(0)
    frames = feats.shape[0]
    full = torch.cat([torch.zeros((frames, 36), device=device), feats], dim=-1)
    shape = SHAPE_PARAM.to(device).repeat(frames, 1)
    _, joints = get_coord(
        root_pose=full[..., 0:3],
        body_pose=full[..., 3:66],
        lhand_pose=full[..., 66:111],
        rhand_pose=full[..., 111:156],
        jaw_pose=full[..., 156:159],
        shape=shape,
        expr=full[..., 159:169],
    )
    return joints.detach().cpu().numpy()


def tstd(joints, indices):
    return float(joints[:, indices, :].std(axis=0).mean())


def speed(joints, indices):
    if len(joints) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(joints[:, indices, :], axis=0), axis=-1).mean())


def diversity(means):
    values = np.stack(means)
    normalized = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)
    distances = []
    cosines = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            distances.append(np.linalg.norm(values[i] - values[j]))
            cosines.append(float(normalized[i] @ normalized[j]))
    return float(np.mean(distances)), float(np.mean(cosines))


def describe_ratios(title, values):
    values = np.asarray(values)
    print(f"\n{title}")
    print(
        f"mean={values.mean():.3f} median={np.median(values):.3f} "
        f"p10={np.percentile(values, 10):.3f} p90={np.percentile(values, 90):.3f} "
        f"<0.30={(values < 0.30).mean():.1%} <0.10={(values < 0.10).mean():.1%}"
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_mean_std(args.mean, args.std, device)
    names = sorted(path.name for path in args.candidate.glob("*.pkl"))
    if args.reference:
        names = [name for name in names if (args.reference / name).exists()]
    labels = ["GT"]
    if args.reference:
        labels.append("SOKE")
    labels.append(args.candidate_name)
    group_tstd = {label: {group: [] for group in GROUPS} for label in labels}
    group_speed = {label: {group: [] for group in GROUPS} for label in labels}
    means = {label: [] for label in labels}
    feature_tstd_ratios = []
    feature_speed_ratios = []
    length_ratios = []

    for name in names:
        with open(args.candidate / name, "rb") as handle:
            candidate = pickle.load(handle)
        reference = None
        if args.reference:
            with open(args.reference / name, "rb") as handle:
                reference = pickle.load(handle)
        joints = {
            "GT": feats_to_joints(candidate["feats_ref"], mean, std, device),
            args.candidate_name: feats_to_joints(candidate["feats_rst"], mean, std, device),
        }
        if reference:
            joints["SOKE"] = feats_to_joints(reference["feats_rst"], mean, std, device)
        gt_feats = np.asarray(candidate["feats_ref"])
        pred_feats = np.asarray(candidate["feats_rst"])
        feature_tstd_ratios.append(
            pred_feats.std(axis=0).mean() / (gt_feats.std(axis=0).mean() + 1e-9)
        )
        feature_speed_ratios.append(
            np.abs(np.diff(pred_feats, axis=0)).mean()
            / (np.abs(np.diff(gt_feats, axis=0)).mean() + 1e-9)
        )
        length_ratios.append(len(pred_feats) / len(gt_feats))
        for label, motion in joints.items():
            upper_body = smpl_x.joint_part2idx["upper_body"]
            means[label].append(motion[:, upper_body, :].mean(axis=0).reshape(-1))
            for group, indices in GROUPS.items():
                group_tstd[label][group].append(tstd(motion, indices))
                group_speed[label][group].append(speed(motion, indices))

    print(f"samples: {len(names)}")
    print("\nVitality tstd (% GT)")
    model_labels = [label for label in labels if label != "GT"]
    print(f"{'part':<18}" + "".join(f"{label:>14}" for label in model_labels))
    for group in GROUPS:
        gt = np.mean(group_tstd["GT"][group])
        values = [100 * np.mean(group_tstd[label][group]) / gt for label in model_labels]
        print(f"{group:<18}" + "".join(f"{value:>13.1f}%" for value in values))

    print("\nMotion speed (% GT)")
    print(f"{'part':<18}" + "".join(f"{label:>14}" for label in model_labels))
    for group in GROUPS:
        gt = np.mean(group_speed["GT"][group])
        values = [100 * np.mean(group_speed[label][group]) / gt for label in model_labels]
        print(f"{group:<18}" + "".join(f"{value:>13.1f}%" for value in values))

    describe_ratios("Per-sequence feature tstd ratio / GT", feature_tstd_ratios)
    describe_ratios("Per-sequence feature speed ratio / GT", feature_speed_ratios)
    length_ratios = np.asarray(length_ratios)
    print("\nLength ratio predicted / GT")
    print(
        f"mean={length_ratios.mean():.3f} median={np.median(length_ratios):.3f} "
        f"p10={np.percentile(length_ratios, 10):.3f} p90={np.percentile(length_ratios, 90):.3f} "
        f"<0.75={(length_ratios < 0.75).mean():.1%} >1.25={(length_ratios > 1.25).mean():.1%}"
    )

    limit = min(args.diversity_samples, len(names))
    gt_div, gt_cos = diversity(means["GT"][:limit])
    print(f"\nCross-sample diversity (first {limit})")
    print(f"{'model':<18} {'div %GT':>10} {'cosine':>10}")
    for label in model_labels:
        div, cosine = diversity(means[label][:limit])
        print(f"{label:<18} {100 * div / gt_div:>9.1f}% {cosine:>10.4f}")
    print(f"{'GT':<18} {100.0:>9.1f}% {gt_cos:>10.4f}")


if __name__ == "__main__":
    main()
