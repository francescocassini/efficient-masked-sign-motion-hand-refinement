#!/usr/bin/env python3
"""Build the compact qualitative paper figure from matched prediction tensors.

The audit GIF for this sample is useful for quick inspection, but it mixes a
P5 row and a P6-K row whose bodies are not matched. For the paper figure we
render the relevant tensors directly:

- GT from the matched HandPolish cache.
- SOKE-AR from the original SOKE run.
- HandPolish from cache replica 0.
- PoseSelect from the final deployable prediction.

For this sample the final PoseSelect prediction has exactly the same body
feature slice as cache replica 0; only hand features change.
"""

from __future__ import annotations

import base64
import html
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Some result pickles were written with NumPy 2 and refer to numpy._core.
# The osx environment used for SMPL-X rendering still has NumPy 1.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


ROOT = Path(__file__).resolve().parents[1]
SOKE_ROOT = Path(os.environ.get("SOKENAR_ROOT", ROOT))
sys.path.insert(0, str(SOKE_ROOT))
from mGPT.utils.human_models import get_coord, smpl_x  # noqa: E402

OUT_DIR = Path(os.environ.get("PAPER_OUT_DIR", SOKE_ROOT / "assets/figures"))
OUT_SVG = OUT_DIR / "paper_qualitative_contrastive_compact.svg"
OUT_PNG = OUT_DIR / "paper_qualitative_contrastive_compact.png"
OUT_PNG_2X = OUT_DIR / "paper_qualitative_contrastive_compact_2x.png"

SAMPLE = "13October_2009_Tuesday_tagesschau-1659"
DATA_ROOT = Path(os.environ.get("SOKE_DATA_ROOT", SOKE_ROOT / "data"))
MEAN_PATH = Path(os.environ.get("PAPER_MEAN_PATH", DATA_ROOT / "CSL-Daily/mean.pt"))
STD_PATH = Path(os.environ.get("PAPER_STD_PATH", DATA_ROOT / "CSL-Daily/std.pt"))
SOKE_PKL = Path(
    os.environ.get(
        "PAPER_SOKE_PKL",
        SOKE_ROOT / "artifacts/soke_ar/COLLAPSE_FULL_SOKE_E69_PHX_SAVE/test_rank_0" / f"{SAMPLE}.pkl",
    )
)
HANDPOLISH_PKL = (
    Path(os.environ["PAPER_HANDPOLISH_PKL"])
    if "PAPER_HANDPOLISH_PKL" in os.environ
    else SOKE_ROOT / "artifacts/handpolish_cache/test_rank_0_rep0" / f"{SAMPLE}.pkl"
)
POSESELECT_NPY = Path(
    os.environ.get(
        "PAPER_POSESELECT_NPY",
        SOKE_ROOT / "artifacts/poseselect/predictions/phoenix" / f"{SAMPLE}.npy",
    )
)

METHODS = [
    ("GT", "reference motion", "#1f7a3f"),
    ("SOKE-AR", "autoregressive baseline", "#3459d1"),
    ("+ HandPolish", "matched cache; refined hands", "#d97a00"),
    ("+ PoseSelect", "same body; selected hands", "#b22cc8"),
]
ROWS = [
    ("Body + fingers", "global pose"),
    ("Left hand zoom", "manual detail"),
    ("Right hand zoom", "manual detail"),
]
SHAPE_PARAM = torch.tensor(
    [[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
      0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]],
    dtype=torch.float32,
)
BODY_CHAINS = (
    (0, 1, 3, 5, 16),
    (0, 2, 4, 6, 19),
    (0, 7),
    (7, 8, 10, 12),
    (7, 9, 11, 13),
    (7, 20), (7, 21), (7, 22), (7, 23), (7, 24),
)
LEFT_FINGERS = tuple(tuple([12] + list(range(25 + 4 * index, 25 + 4 * index + 4))) for index in range(5))
RIGHT_FINGERS = tuple(tuple([13] + list(range(45 + 4 * index, 45 + 4 * index + 4))) for index in range(5))


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_mean_std(mean_path: Path, std_path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.load(mean_path, map_location=device).float()
    std = torch.load(std_path, map_location=device).float()
    mean = mean[(3 + 3 * 11):]
    std = std[(3 + 3 * 11):]
    return (
        torch.cat([mean[:-20], mean[-10:]], dim=0).to(device),
        torch.cat([std[:-20], std[-10:]], dim=0).to(device),
    )


def feats_to_joints65(feats_np: np.ndarray, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> np.ndarray:
    feats = torch.from_numpy(np.ascontiguousarray(feats_np)).float().to(device)
    frames = feats.shape[0]
    feats = feats * std.unsqueeze(0) + mean.unsqueeze(0)
    zero_pose = torch.zeros((frames, 36), device=device)
    shape = SHAPE_PARAM.to(device).repeat(frames, 1)
    full = torch.cat([zero_pose, feats], dim=-1)
    vertices, raw_joints = get_coord(
        root_pose=full[..., 0:3],
        body_pose=full[..., 3:66],
        lhand_pose=full[..., 66:111],
        rhand_pose=full[..., 111:156],
        jaw_pose=full[..., 156:159],
        shape=shape,
        expr=full[..., 159:169],
    )
    body_idx = torch.as_tensor(smpl_x.joint_idx[:25], device=device, dtype=torch.long)
    body = raw_joints.index_select(1, body_idx)
    left_reg = smpl_x.orig_hand_regressor["left"].to(device)
    right_reg = smpl_x.orig_hand_regressor["right"].to(device)
    left = torch.einsum("jv,tvc->tjc", left_reg, vertices)[:, 1:]
    right = torch.einsum("jv,tvc->tjc", right_reg, vertices)[:, 1:]
    return torch.cat([body, left, right], dim=1).detach().cpu().numpy()


def project(points: np.ndarray) -> np.ndarray:
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([x + 0.18 * z, -y + 0.08 * z], axis=-1)


def bounds_for(method_joints: dict[str, np.ndarray], indices: list[int], margin: float = 0.12):
    all_points = [project(joints[:, indices]).reshape(-1, 2) for joints in method_joints.values()]
    points = np.concatenate(all_points, axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    return lo - span * margin, hi + span * margin


def hand_bounds_for(method_joints: dict[str, np.ndarray], wrist_idx: int, hand_start: int, margin: float = 0.18):
    all_points = []
    for joints in method_joints.values():
        wrist = joints[:, wrist_idx:wrist_idx + 1]
        points = np.concatenate([wrist, joints[:, hand_start:hand_start + 20]], axis=1) - wrist
        all_points.append(project(points).reshape(-1, 2))
    points = np.concatenate(all_points, axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    return lo - span * margin, hi + span * margin


def map_to_box(points: np.ndarray, bounds, box: tuple[int, int, int, int]) -> np.ndarray:
    lo, hi = bounds
    x0, y0, x1, y1 = box
    span = np.maximum(hi - lo, 1e-6)
    scale = min((x1 - x0) / span[0], (y1 - y0) / span[1])
    centered = (points - (lo + hi) / 2.0) * scale
    return centered + np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])


def draw_chains(draw: ImageDraw.ImageDraw, points: np.ndarray, chains, color, width: int) -> None:
    for chain in chains:
        coords = [tuple(points[index]) for index in chain]
        draw.line(coords, fill=color, width=width, joint="curve")
        for coord in coords:
            radius = max(1, width // 2)
            draw.ellipse((coord[0] - radius, coord[1] - radius, coord[0] + radius, coord[1] + radius), fill=color)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, color: str, bold: bool = False) -> None:
    draw.text(xy, value, fill=hex_to_rgb(color), font=font(size, bold))


def draw_panel_border(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rectangle(box, outline=(212, 212, 212), width=1)


def draw_body_panel(draw, frame, bounds, box, color, scale):
    x0, y0, x1, y1 = box
    points = map_to_box(
        project(frame),
        bounds,
        (x0 + 18 * scale, y0 + 18 * scale, x1 - 18 * scale, y1 - 12 * scale),
    )
    draw_chains(draw, points, BODY_CHAINS, color, 3 * scale)
    draw_chains(draw, points, LEFT_FINGERS + RIGHT_FINGERS, color, 2 * scale)


def draw_hand_panel(draw, frame, bounds, box, color, scale, *, side: str):
    x0, y0, x1, y1 = box
    if side == "left":
        points_3d = np.concatenate([frame[12:13], frame[25:45]], axis=0) - frame[12:13]
    else:
        points_3d = np.concatenate([frame[13:14], frame[45:65]], axis=0) - frame[13:14]
    chains = tuple(tuple([0] + list(range(1 + 4 * index, 1 + 4 * index + 4))) for index in range(5))
    points = map_to_box(
        project(points_3d),
        bounds,
        (x0 + 34 * scale, y0 + 18 * scale, x1 - 34 * scale, y1 - 14 * scale),
    )
    draw_chains(draw, points, chains, color, 4 * scale)


def load_method_joints():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_mean_std(MEAN_PATH, STD_PATH, device)

    soke = load_pickle(SOKE_PKL)
    handpolish = load_pickle(HANDPOLISH_PKL)
    poseselect = np.load(POSESELECT_NPY).astype(np.float32)

    body_delta = float(np.abs(np.asarray(handpolish["feats_rst"])[:, :30] - poseselect[:, :30]).max())
    if body_delta > 1e-7:
        raise RuntimeError(f"PoseSelect is not body-locked to the selected HandPolish cache: max delta {body_delta}")

    feats = {
        "GT": np.asarray(handpolish["feats_ref"], dtype=np.float32),
        "SOKE-AR": np.asarray(soke["feats_rst"], dtype=np.float32),
        "+ HandPolish": np.asarray(handpolish["feats_rst"], dtype=np.float32),
        "+ PoseSelect": poseselect,
    }
    joints = {name: feats_to_joints65(value, mean, std, device) for name, value in feats.items()}
    return joints, body_delta


def draw_figure(scale: int = 1) -> Image.Image:
    joints, _body_delta = load_method_joints()

    width, height = 1500 * scale, 690 * scale
    margin = 26 * scale
    label_w = 185 * scale
    header_h = 72 * scale
    col_w = 310 * scale
    col_gap = 9 * scale
    row_gap = 8 * scale
    body_h = 215 * scale
    hand_h = 150 * scale
    x0 = margin + label_w
    y0 = margin + header_h

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for col_idx, (name, desc, color_hex) in enumerate(METHODS):
        x = x0 + col_idx * (col_w + col_gap)
        draw.rectangle((x, margin, x + 7 * scale, margin + header_h - 10 * scale), fill=hex_to_rgb(color_hex))
        text(draw, (x + 18 * scale, margin + 9 * scale), name, 23 * scale, color_hex, bold=True)
        text(draw, (x + 18 * scale, margin + 40 * scale), desc, 14 * scale, "#555555")

    full_bounds = bounds_for(joints, list(range(65)))
    left_bounds = hand_bounds_for(joints, wrist_idx=12, hand_start=25)
    right_bounds = hand_bounds_for(joints, wrist_idx=13, hand_start=45)

    frame_idx = 0
    current_y = y0
    for row_index, (row_title, row_desc) in enumerate(ROWS):
        row_h = body_h if row_index == 0 else hand_h
        text(draw, (margin + 4 * scale, current_y + 21 * scale), row_title, 17 * scale, "#333333", bold=True)
        text(draw, (margin + 4 * scale, current_y + 51 * scale), row_desc, 14 * scale, "#777777")

        for col_idx, (method, _desc, color_hex) in enumerate(METHODS):
            x = x0 + col_idx * (col_w + col_gap)
            box = (x, current_y, x + col_w, current_y + row_h)
            draw.rectangle(box, fill=(248, 248, 248))
            frame = joints[method][min(frame_idx, len(joints[method]) - 1)]
            color = hex_to_rgb(color_hex)
            if row_index == 0:
                draw_body_panel(draw, frame, full_bounds, box, color, scale)
            elif row_index == 1:
                draw_hand_panel(draw, frame, left_bounds, box, color, scale, side="left")
            else:
                draw_hand_panel(draw, frame, right_bounds, box, color, scale, side="right")
            draw_panel_border(draw, box)
        current_y += row_h + row_gap

    return image


def write_svg_wrapper(png_path: Path, svg_path: Path) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="690" viewBox="0 0 1500 690">\n'
        f'  <image href="data:image/png;base64,{html.escape(encoded)}" x="0" y="0" width="1500" height="690"/>\n'
        "</svg>\n"
    )
    svg_path.write_text(svg, encoding="utf-8")


def main() -> None:
    for path in (MEAN_PATH, STD_PATH, SOKE_PKL, HANDPOLISH_PKL, POSESELECT_NPY):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_figure(scale=1).save(OUT_PNG)
    draw_figure(scale=2).save(OUT_PNG_2X)
    write_svg_wrapper(OUT_PNG, OUT_SVG)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PNG_2X}")
    print(f"wrote {OUT_SVG}")
    print("PoseSelect body is locked to HandPolish cache replica 0 (max body feature delta = 0.0).")


if __name__ == "__main__":
    main()
