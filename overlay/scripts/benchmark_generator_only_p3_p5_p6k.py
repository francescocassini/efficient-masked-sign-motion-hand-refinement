#!/usr/bin/env python3
"""Generator-only timing benchmark for the clean P6-K stack.

This script starts from the existing P5/T5C cache and measures the deployable
post-P5 generator stack: P6-B candidate generation, P6-D gain scoring/budget
selection and P6-K pose-aware top-k selection. The cached P5 output itself is
reported as a cache-read baseline, not as P3/P5 generation time.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mGPT.models.utils.p6_hand_gain_regressor import P6HandGainRegressor
from mGPT.models.utils.p6_hand_token_editor import P6HandTokenEditor
from mGPT.models.utils.p6_topk_candidate_selector import P6TopKCandidateSelector
from scripts.p6_geometric_candidate_selector_eval import candidate_features, mean_abs_velocity
from scripts.p6_hand_gain_regressor_budget_sweep import select_top_budget
from scripts.p6_hand_topk_candidate_oracle import load_regressor
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_pose_aware_candidate_selector_train import META_DIM, build_meta
from scripts.p6_token_edit_oracle_ceiling import (
    load_vq_pack,
    normalize_token_shape,
    pad_or_crop_np,
    pad_or_crop_tokens,
)
from scripts.t5_gated_residual_feature_sweep import dataset_names


LH_START = 30
LH_END = 75
RH_START = 75
RH_END = 120

DEFAULT_CACHE = Path("results/mgpt_t5c_extended_cache/T5C_P5_CONF_CACHE_FULL/test_rank_0")
DEFAULT_P6B = Path("results/p6_hand_token_editor/clean_trainval_e20_h256_l4/best_p6_hand_token_editor.pt")
DEFAULT_P6D = Path("results/p6_hand_gain_regressor/clean_trainval_e10_h192_l3/best_p6_hand_gain_regressor.pt")
DEFAULT_P6K = Path(
    "results/p6_pose_aware_candidate_selector_clean/"
    "clean_trainval_e12_top5_b020_h256/best_p6_pose_aware_candidate_selector.pt"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both", choices=["csl", "phoenix", "both"])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, type=Path)
    parser.add_argument("--p6b-checkpoint", default=DEFAULT_P6B, type=Path)
    parser.add_argument("--regressor-checkpoint", default=DEFAULT_P6D, type=Path)
    parser.add_argument("--selector-checkpoint", default=DEFAULT_P6K, type=Path)
    parser.add_argument("--output-dir", default=Path("results/benchmarks/generator_only_p3_p5_p6k"), type=Path)
    parser.add_argument("--config", default=Path("configs/soke.yaml"), type=Path)
    parser.add_argument("--default-config", default=Path("configs/default.yaml"), type=Path)
    parser.add_argument("--vae-ckpt", default=Path("deps/tokenizer_ckpt/tokenizer.ckpt"), type=Path)
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--warmup-samples", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_payload(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_selector(checkpoint: Path, device):
    state = torch.load(checkpoint, map_location="cpu")
    args = state.get("args", {})
    model = P6TopKCandidateSelector(
        hidden_features=int(args.get("hidden_features", 256)),
        dropout=float(args.get("dropout", 0.1)),
        meta_dim=int(args.get("meta_dim", META_DIM)),
        max_rank=int(args.get("topk", 5)),
    )
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return model, args


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device):
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def time_call(device, fn):
    synchronize(device)
    start = time.perf_counter()
    value = fn()
    synchronize(device)
    return value, time.perf_counter() - start


def empty_timings():
    return {
        "p5_cache_read": [],
        "p6b_candidate_generation": [],
        "p6d_gain_scoring": [],
        "p6d_budget_selection": [],
        "p6k_pose_aware_selection": [],
        "post_p5_total": [],
    }


def summarize_times(values, samples):
    if not values:
        return {
            "samples": 0,
            "latency_mean_s": None,
            "latency_median_s": None,
            "latency_p90_s": None,
            "throughput_samples_s": None,
            "elapsed_total_s": 0.0,
        }
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(np.ceil(0.90 * len(ordered))) - 1)
    total = float(sum(values))
    return {
        "samples": int(samples),
        "latency_mean_s": float(statistics.mean(values)),
        "latency_median_s": float(statistics.median(values)),
        "latency_p90_s": float(ordered[p90_index]),
        "throughput_samples_s": float(samples / max(total, 1e-12)),
        "elapsed_total_s": total,
    }


def editor_topk_with_probs(editor, payload, token_len, topk, device):
    body = pad_or_crop_tokens(payload["tokens_body"], token_len)
    lhand = pad_or_crop_tokens(payload["tokens_lhand"], token_len)
    rhand = pad_or_crop_tokens(payload["tokens_rhand"], token_len)
    conf = np.concatenate(
        [
            pad_or_crop_np(np.asarray(payload["confidence_body"], dtype=np.float32)[:, None], token_len),
            pad_or_crop_np(np.asarray(payload["confidence_lhand"], dtype=np.float32)[:, None], token_len),
            pad_or_crop_np(np.asarray(payload["confidence_rhand"], dtype=np.float32)[:, None], token_len),
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
        l_values, l_indices = torch.topk(l_prob, k=topk, dim=-1)
        r_values, r_indices = torch.topk(r_prob, k=topk, dim=-1)
    return (
        body,
        lhand,
        rhand,
        conf,
        l_indices.detach().cpu().numpy()[0].astype(np.int64),
        r_indices.detach().cpu().numpy()[0].astype(np.int64),
        l_values.detach().cpu().numpy()[0].astype(np.float32),
        r_values.detach().cpu().numpy()[0].astype(np.float32),
    )


def score_top1_from_cached_p6b(regressor, body, lhand, rhand, conf, l_top, r_top, l_probs, r_probs, payload, device):
    cand_lhand = l_top[:, 0]
    cand_rhand = r_top[:, 0]
    progress = np.linspace(0.0, 1.0, len(lhand), dtype=np.float32)
    meta = np.concatenate(
        [
            conf,
            l_probs[:, :1],
            r_probs[:, :1],
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
        mask = torch.ones((1, len(lhand)), dtype=torch.float32, device=device)
        text = P6HandGainRegressor.hash_text([str(payload.get("text", ""))], 4096, 96, device)
        l_score, r_score = regressor(body_t, l_t, r_t, cand_l_t, cand_r_t, meta_t, text, mask)
    return cand_lhand, cand_rhand, l_score.detach().cpu().numpy()[0], r_score.detach().cpu().numpy()[0]


def local_base_velocity(base_hand, index):
    start = int(index) * 4
    end = min((int(index) + 1) * 4, len(base_hand))
    if end <= start:
        return 0.0
    return mean_abs_velocity(base_hand[start:end])


def selector_scores(selector, vae, side, body, lhand, rhand, conf, top_tokens, top_probs, p6d_scores, index, base_hand, device):
    orig = lhand if side == 0 else rhand
    other = rhand if side == 0 else lhand
    progress = index / max(1, len(orig) - 1)
    candidates = list(top_tokens[index]) + [int(orig[index])]
    probs = list(top_probs[index]) + [1.0]
    base_vel = local_base_velocity(base_hand, index)
    rows = []
    for rank, (token, prob) in enumerate(zip(candidates, probs)):
        geom = candidate_features(vae, orig, int(token), float(prob), index, base_hand, len(base_hand), device)
        rows.append(
            {
                "body": int(body[index]),
                "orig": int(orig[index]),
                "other": int(other[index]),
                "candidate": int(token),
                "side": int(side),
                "rank": int(rank),
                "meta": build_meta(conf[index], prob, p6d_scores[index], progress, token, orig[index], rank, len(top_tokens[index]), geom, base_vel),
            }
        )
    with torch.no_grad():
        score = selector(
            torch.tensor([row["body"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["orig"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["other"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["candidate"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["side"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["rank"] for row in rows], dtype=torch.long, device=device),
            torch.tensor(np.stack([row["meta"] for row in rows]), dtype=torch.float32, device=device),
        ).detach().cpu().numpy()
    best = int(np.argmax(score))
    return rows[best]["candidate"], best, float(score[best])


def choose_selector_tokens(selector, side, selected, body, lhand, rhand, conf, top_tokens, top_probs, p6d_scores, base_hand, pack, device):
    vae = pack.hand_vae if side == 0 else pack.rhand_vae
    edited = lhand.copy() if side == 0 else rhand.copy()
    chosen_rank = []
    chosen_score = []
    for index in np.where(selected)[0]:
        token, rank, score = selector_scores(
            selector,
            vae,
            side,
            body,
            lhand,
            rhand,
            conf,
            top_tokens,
            top_probs,
            p6d_scores,
            int(index),
            base_hand,
            device,
        )
        edited[index] = token
        chosen_rank.append(rank)
        chosen_score.append(score)
    return edited, {
        "selected": int(selected.sum()),
        "mean_rank": float(np.mean(chosen_rank)) if chosen_rank else -1.0,
        "mean_score": float(np.mean(chosen_score)) if chosen_score else 0.0,
    }


def run_one_payload(payload, editor, regressor, selector, pack, args, device, measure=True):
    timings = empty_timings()

    def timed(name, fn):
        value, elapsed = time_call(device, fn)
        if measure:
            timings[name].append(elapsed)
        return value

    total_start = time.perf_counter()
    base = timed("p5_cache_read", lambda: np.asarray(payload["feats_rst"], dtype=np.float32))
    token_len = min(
        len(normalize_token_shape(payload["tokens_lhand"])),
        len(normalize_token_shape(payload["tokens_rhand"])),
    )
    body, lhand, rhand, conf, l_top, r_top, l_probs, r_probs = timed(
        "p6b_candidate_generation",
        lambda: editor_topk_with_probs(editor, payload, token_len, args.topk, device),
    )
    _l_top1, _r_top1, l_score, r_score = timed(
        "p6d_gain_scoring",
        lambda: score_top1_from_cached_p6b(
            regressor, body, lhand, rhand, conf, l_top, r_top, l_probs, r_probs, payload, device
        ),
    )
    l_sel, r_sel = timed(
        "p6d_budget_selection",
        lambda: (select_top_budget(l_score, args.budget), select_top_budget(r_score, args.budget)),
    )

    def select_p6k():
        l_edit, l_stats = choose_selector_tokens(
            selector, 0, l_sel, body, lhand, rhand, conf, l_top, l_probs, l_score, base[:, LH_START:LH_END], pack, device
        )
        r_edit, r_stats = choose_selector_tokens(
            selector, 1, r_sel, body, lhand, rhand, conf, r_top, r_probs, r_score, base[:, RH_START:RH_END], pack, device
        )
        return l_edit, r_edit, l_stats, r_stats

    _l_edit, _r_edit, l_stats, r_stats = timed("p6k_pose_aware_selection", select_p6k)
    synchronize(device)
    if measure:
        timings["post_p5_total"].append(time.perf_counter() - total_start)
    return timings, {
        "token_len": int(token_len),
        "frames": int(len(base)),
        "lh_selected": int(l_stats["selected"]),
        "rh_selected": int(r_stats["selected"]),
        "lh_selector_rank": float(l_stats["mean_rank"]),
        "rh_selector_rank": float(r_stats["mean_rank"]),
    }


def merge_timings(target, source):
    for key, values in source.items():
        target[key].extend(values)


def selected_names(cache_dir, dataset, max_samples):
    datasets = ["csl", "phoenix"] if dataset == "both" else [dataset]
    names = []
    per_dataset = max_samples if dataset != "both" else 0
    for item in datasets:
        names.extend(dataset_names(cache_dir, item, per_dataset))
    if dataset == "both" and max_samples:
        names = names[:max_samples]
    return names


def markdown_report(report):
    rows = [
        f"# P6-K generator-only benchmark",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Samples: **{report['samples']}** measured, **{report['warmup_samples']}** warmup",
        f"Device: `{report['device']}`",
        "",
        "Scope: starts from the P5/T5C cache. `p5_cache_read` is a cache-read baseline; P3 and P5 generator time are not measured by this cache benchmark.",
        "",
        "| Block | Mean ms/sample | Median | P90 | Samples/s | Peak VRAM MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    peak = report.get("peak_vram_mb")
    peak_text = "n/a" if peak is None else f"{peak:.1f}"
    for name, row in report["summary"].items():
        if row["latency_mean_s"] is None:
            rows.append(f"| {name} | n/a | n/a | n/a | n/a | {peak_text} |")
        else:
            rows.append(
                f"| {name} | {row['latency_mean_s'] * 1000:.3f} | "
                f"{row['latency_median_s'] * 1000:.3f} | {row['latency_p90_s'] * 1000:.3f} | "
                f"{row['throughput_samples_s']:.2f} | {peak_text} |"
            )
    rows.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Uses the clean P6-B, P6-D and P6-K checkpoints.",
            "- Uses `torch.cuda.synchronize()` around every timed block on CUDA.",
            "- Excludes full motion VQ decode, metric computation, rendering and prediction saving.",
            "- P6-K pose-aware selection includes its internal hand-token candidate feature computation.",
            "- No ground-truth motion is read by the measured inference path.",
        ]
    )
    return "\n".join(rows) + "\n"


def manifest_text(report):
    lines = [
        "# Manifest - P6-K Generator-Only Benchmark",
        "",
        f"Created by: `scripts/benchmark_generator_only_p3_p5_p6k.py`",
        f"Dataset: `{report['dataset']}`",
        f"Samples: `{report['samples']}`",
        f"Warmup samples: `{report['warmup_samples']}`",
        f"Device: `{report['device']}`",
        "",
        "## Inputs",
        "",
        f"- Cache: `{report['cache_dir']}`",
        f"- P6-B checkpoint: `{report['checkpoints']['p6b']}`",
        f"- P6-D checkpoint: `{report['checkpoints']['p6d']}`",
        f"- P6-K checkpoint: `{report['checkpoints']['p6k']}`",
        f"- top-k: `{report['topk']}`",
        f"- budget: `{report['budget']}`",
        "",
        "## Scope",
        "",
        "This run measures the post-P5 cached generator stack: P6-B, P6-D and P6-K. "
        "It does not measure fresh P3/P5 generation because the benchmark starts from cached P5 outputs.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    names = selected_names(args.cache_dir, args.dataset, args.max_samples)
    if not names:
        raise RuntimeError(f"No samples found in {args.cache_dir} for dataset {args.dataset}")
    payloads = [load_payload(args.cache_dir / f"{name}.pkl") for name in names]
    warmup_payloads = payloads[: max(0, args.warmup_samples)]
    measured_payloads = payloads[max(0, args.warmup_samples):]
    if not measured_payloads:
        raise RuntimeError("No measured samples left after warmup; reduce --warmup-samples or raise --max-samples")

    pack_args = SimpleNamespace(config=args.config, default_config=args.default_config, vae_ckpt=args.vae_ckpt)
    pack = load_vq_pack(pack_args, device)
    editor, editor_args = load_p6_editor(args.p6b_checkpoint, device)
    regressor, regressor_args = load_regressor(args.regressor_checkpoint, device)
    selector, selector_args = load_selector(args.selector_checkpoint, device)

    for payload in warmup_payloads:
        run_one_payload(payload, editor, regressor, selector, pack, args, device, measure=False)
    synchronize(device)
    reset_peak(device)

    timings = empty_timings()
    sample_stats = []
    wall_start = time.perf_counter()
    for index, payload in enumerate(measured_payloads, start=1):
        item_timings, item_stats = run_one_payload(payload, editor, regressor, selector, pack, args, device, measure=True)
        merge_timings(timings, item_timings)
        sample_stats.append(item_stats)
        if args.progress_every and index % args.progress_every == 0:
            print(f"Measured {index}/{len(measured_payloads)} samples", flush=True)
    total_wall = time.perf_counter() - wall_start

    summary = {name: summarize_times(values, len(measured_payloads)) for name, values in timings.items()}
    post = summary["post_p5_total"]["elapsed_total_s"]
    for name, row in summary.items():
        row["share_of_post_p5_total"] = None if post <= 0 else float(row["elapsed_total_s"] / post)

    report = {
        "benchmark": "generator_only_p3_p5_p6k_from_p5_cache",
        "scope": "post_p5_cached_generator_stack",
        "dataset": args.dataset,
        "samples": len(measured_payloads),
        "warmup_samples": len(warmup_payloads),
        "cache_dir": str(args.cache_dir),
        "device": str(device),
        "topk": args.topk,
        "budget": args.budget,
        "seed": args.seed,
        "wall_time_s": total_wall,
        "peak_vram_mb": peak_mb(device),
        "checkpoints": {
            "p6b": str(args.p6b_checkpoint),
            "p6d": str(args.regressor_checkpoint),
            "p6k": str(args.selector_checkpoint),
        },
        "checkpoint_args": {
            "p6b": editor_args,
            "p6d": regressor_args,
            "p6k": selector_args,
        },
        "p3_p5_generation": {
            "status": "not_measured_from_cache",
            "reason": "This script starts from cached P5 outputs in feats_rst; fresh P3/P5 generation requires a separate runner/config benchmark.",
        },
        "summary": summary,
        "selection_stats": {
            "token_len_mean": float(np.mean([item["token_len"] for item in sample_stats])),
            "frames_mean": float(np.mean([item["frames"] for item in sample_stats])),
            "lh_selected_mean": float(np.mean([item["lh_selected"] for item in sample_stats])),
            "rh_selected_mean": float(np.mean([item["rh_selected"] for item in sample_stats])),
            "lh_selector_rank_mean": float(np.mean([item["lh_selector_rank"] for item in sample_stats if item["lh_selector_rank"] >= 0]))
            if any(item["lh_selector_rank"] >= 0 for item in sample_stats)
            else -1.0,
            "rh_selector_rank_mean": float(np.mean([item["rh_selector_rank"] for item in sample_stats if item["rh_selector_rank"] >= 0]))
            if any(item["rh_selector_rank"] >= 0 for item in sample_stats)
            else -1.0,
        },
        "method_notes": [
            "torch.cuda.synchronize is used before and after every timed block on CUDA.",
            "Full VQ decode, metrics, rendering and prediction saving are excluded.",
            "The P6-K block includes internal hand-token candidate feature computation.",
            "No ground-truth motion is used in the measured inference path.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown_report(report), encoding="utf-8")
    (args.output_dir / "MANIFEST.md").write_text(manifest_text(report), encoding="utf-8")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
