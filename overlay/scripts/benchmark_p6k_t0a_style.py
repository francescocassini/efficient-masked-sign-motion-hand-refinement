#!/usr/bin/env python3
"""T0-A style benchmark for P5-cache, clean P6-D and clean P6-K.

The benchmark starts from the validated P5/T5C cache, like the clean P6-K
validation. It measures deployable post-cache inference variants and samples
GPU power with nvidia-smi when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.benchmark_generator_only_p3_p5_p6k import (
    DEFAULT_CACHE,
    DEFAULT_P6B,
    DEFAULT_P6D,
    DEFAULT_P6K,
    LH_END,
    LH_START,
    RH_END,
    RH_START,
    choose_selector_tokens,
    editor_topk_with_probs,
    load_payload,
    load_selector,
    score_top1_from_cached_p6b,
    selected_names,
    synchronize,
)
from scripts.p6_hand_gain_regressor_budget_sweep import select_top_budget
from scripts.p6_hand_topk_candidate_oracle import load_regressor
from scripts.p6_hand_token_editor_paired_eval import load_model as load_p6_editor
from scripts.p6_token_edit_oracle_ceiling import decode_hand_tokens, load_vq_pack, normalize_token_shape


QUALITY_REFERENCE = {
    "csl": {
        "p5_cache": {"pa_body": 8.3074, "pa_hand": 1.8057, "speed_gt": 0.773, "tstd_gt": 0.611},
        "p6d_clean": {"pa_body": 8.2913, "pa_hand": 1.6918, "speed_gt": 0.735, "tstd_gt": 0.587},
        "p6k_clean": {"pa_body": 8.2880, "pa_hand": 1.6877, "speed_gt": 0.768, "tstd_gt": 0.636},
    },
    "phoenix": {
        "p5_cache": {"pa_body": 6.7821, "pa_hand": 1.3497, "speed_gt": 0.544, "tstd_gt": 0.587},
        "p6d_clean": {"pa_body": 6.7746, "pa_hand": 1.3109, "speed_gt": 0.454, "tstd_gt": 0.508},
        "p6k_clean": {"pa_body": 6.7711, "pa_hand": 1.3104, "speed_gt": 0.520, "tstd_gt": 0.574},
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both", choices=["csl", "phoenix", "both"])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, type=Path)
    parser.add_argument("--p6b-checkpoint", default=DEFAULT_P6B, type=Path)
    parser.add_argument("--regressor-checkpoint", default=DEFAULT_P6D, type=Path)
    parser.add_argument("--selector-checkpoint", default=DEFAULT_P6K, type=Path)
    parser.add_argument("--output-dir", default=Path("results/benchmarks/p6k_t0a_style"), type=Path)
    parser.add_argument("--config", default=Path("configs/soke.yaml"), type=Path)
    parser.add_argument("--default-config", default=Path("configs/default.yaml"), type=Path)
    parser.add_argument("--vae-ckpt", default=Path("deps/tokenizer_ckpt/tokenizer.ckpt"), type=Path)
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--warmup-samples", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--monitor-interval", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_mb(device):
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def gpu_sample(gpu_index):
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=timestamp,memory.used,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    query = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().split(", ")
    return {
        "wall_time": time.time(),
        "timestamp": query[0],
        "memory_mib": float(query[1]),
        "power_w": float(query[2]),
        "utilization_pct": float(query[3]),
    }


class GPUMonitor:
    def __init__(self, gpu_index, interval):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self.error = None
        self._stop = threading.Event()
        self._ready = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(gpu_sample(self.gpu_index))
                if self._ready.empty():
                    self._ready.put(True)
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
                self.error = str(exc)
                if self._ready.empty():
                    self._ready.put(False)
                break
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        try:
            self._ready.get(timeout=2.0)
        except queue.Empty:
            self.error = "nvidia-smi monitor did not produce an initial sample"
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def summary(self, duration_s, samples_count):
        if len(self.samples) < 2:
            return {
                "gpu_monitor_available": False,
                "gpu_monitor_error": self.error,
                "gpu_samples": len(self.samples),
                "peak_memory_mib_nvidia_smi": None,
                "mean_power_w": None,
                "peak_power_w": None,
                "gpu_energy_j": None,
                "gpu_energy_j_per_sample": None,
                "mean_utilization_pct": None,
            }
        energy_j = sum(
            0.5 * (left["power_w"] + right["power_w"]) * (right["wall_time"] - left["wall_time"])
            for left, right in zip(self.samples, self.samples[1:])
        )
        return {
            "gpu_monitor_available": True,
            "gpu_monitor_error": None,
            "gpu_samples": len(self.samples),
            "peak_memory_mib_nvidia_smi": max(sample["memory_mib"] for sample in self.samples),
            "mean_power_w": sum(sample["power_w"] for sample in self.samples) / len(self.samples),
            "peak_power_w": max(sample["power_w"] for sample in self.samples),
            "gpu_energy_j": energy_j,
            "gpu_energy_j_per_sample": energy_j / max(samples_count, 1),
            "mean_utilization_pct": sum(sample["utilization_pct"] for sample in self.samples) / len(self.samples),
            "sampled_duration_s": duration_s,
        }


def p5_cache_variant(payload):
    return np.asarray(payload["feats_rst"], dtype=np.float32).copy()


def p6_common(payload, editor, regressor, args, device):
    base = np.asarray(payload["feats_rst"], dtype=np.float32)
    token_len = min(
        len(normalize_token_shape(payload["tokens_lhand"])),
        len(normalize_token_shape(payload["tokens_rhand"])),
    )
    body, lhand, rhand, conf, l_top, r_top, l_probs, r_probs = editor_topk_with_probs(
        editor, payload, token_len, args.topk, device
    )
    l_top1, r_top1, l_score, r_score = score_top1_from_cached_p6b(
        regressor, body, lhand, rhand, conf, l_top, r_top, l_probs, r_probs, payload, device
    )
    l_sel = select_top_budget(l_score, args.budget)
    r_sel = select_top_budget(r_score, args.budget)
    return base, body, lhand, rhand, conf, l_top, r_top, l_probs, r_probs, l_top1, r_top1, l_score, r_score, l_sel, r_sel


def p6d_variant(payload, editor, regressor, pack, args, device):
    base, _body, lhand, rhand, _conf, _l_top, _r_top, _l_probs, _r_probs, l_top1, r_top1, _l_score, _r_score, l_sel, r_sel = p6_common(
        payload, editor, regressor, args, device
    )
    pred = base.copy()
    l_edit = lhand.copy()
    r_edit = rhand.copy()
    l_edit[l_sel] = l_top1[l_sel]
    r_edit[r_sel] = r_top1[r_sel]
    if l_sel.any():
        pred[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_edit, len(base), device)
    if r_sel.any():
        pred[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_edit, len(base), device)
    return pred.astype(np.float32)


def p6k_variant(payload, editor, regressor, selector, pack, args, device):
    base, body, lhand, rhand, conf, l_top, r_top, l_probs, r_probs, _l_top1, _r_top1, l_score, r_score, l_sel, r_sel = p6_common(
        payload, editor, regressor, args, device
    )
    pred = base.copy()
    l_edit, _l_stats = choose_selector_tokens(
        selector, 0, l_sel, body, lhand, rhand, conf, l_top, l_probs, l_score, base[:, LH_START:LH_END], pack, device
    )
    r_edit, _r_stats = choose_selector_tokens(
        selector, 1, r_sel, body, lhand, rhand, conf, r_top, r_probs, r_score, base[:, RH_START:RH_END], pack, device
    )
    if l_sel.any():
        pred[:, LH_START:LH_END] = decode_hand_tokens(pack.hand_vae, l_edit, len(base), device)
    if r_sel.any():
        pred[:, RH_START:RH_END] = decode_hand_tokens(pack.rhand_vae, r_edit, len(base), device)
    return pred.astype(np.float32)


def run_variant(name, payloads, fn, args, device):
    reset_peak(device)
    synchronize(device)
    with GPUMonitor(args.gpu_index, args.monitor_interval) as monitor:
        start = time.perf_counter()
        frames = 0
        for index, payload in enumerate(payloads, start=1):
            pred = fn(payload)
            frames += int(len(pred))
            if args.progress_every and index % args.progress_every == 0:
                print(f"{name}: {index}/{len(payloads)} samples", flush=True)
        synchronize(device)
        duration_s = time.perf_counter() - start
    result = {
        "variant": name,
        "samples": len(payloads),
        "frames": frames,
        "duration_s": duration_s,
        "throughput_samples_s": len(payloads) / max(duration_s, 1e-12),
        "time_per_sample_s": duration_s / max(len(payloads), 1),
        "throughput_frames_s": frames / max(duration_s, 1e-12),
        "cuda_peak_memory_mib": cuda_peak_mb(device),
    }
    result.update(monitor.summary(duration_s, len(payloads)))
    return result, monitor.samples


def write_gpu_csv(path, samples):
    if not samples:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)


def markdown_report(report):
    rows = [
        "# P6-K T0-A style benchmark",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Samples: **{report['samples']}** measured, **{report['warmup_samples']}** warmup",
        f"Device: `{report['device']}`",
        "",
        "Scope: starts from the validated P5/T5C cache and measures deployable post-cache variants.",
        "",
        "| Variant | sec/sample | samples/s | peak CUDA MB | energy J/sample | nvidia-smi peak MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["variants"]:
        energy = item["gpu_energy_j_per_sample"]
        energy_text = "n/a" if energy is None else f"{energy:.3f}"
        smi_peak = item["peak_memory_mib_nvidia_smi"]
        smi_text = "n/a" if smi_peak is None else f"{smi_peak:.1f}"
        cuda_peak = item["cuda_peak_memory_mib"]
        cuda_text = "n/a" if cuda_peak is None else f"{cuda_peak:.1f}"
        rows.append(
            f"| {item['variant']} | {item['time_per_sample_s']:.6f} | "
            f"{item['throughput_samples_s']:.2f} | {cuda_text} | {energy_text} | {smi_text} |"
        )
    rows.extend(
        [
            "",
            "## Overhead",
            "",
            "| Comparison | Extra sec/sample | Extra ms/sample |",
            "|---|---:|---:|",
        ]
    )
    for key, value in report["overhead"].items():
        rows.append(f"| {key} | {value:.6f} | {value * 1000:.3f} |")
    rows.extend(
        [
            "",
            "## Quality Reference",
            "",
            "Quality values are copied from the clean P6 validation report, not recomputed by this timing benchmark.",
            "",
            "| Dataset | Variant | PA-body | PA-hand | speed/GT | tstd/GT |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset, variants in report["quality_reference"].items():
        for variant, metrics in variants.items():
            rows.append(
                f"| {dataset} | {variant} | {metrics['pa_body']:.4f} | {metrics['pa_hand']:.4f} | "
                f"{metrics['speed_gt']:.3f} | {metrics['tstd_gt']:.3f} |"
            )
    rows.extend(
        [
            "",
            "## Method Notes",
            "",
            "- P6-D and P6-K include final LH/RH VQ decode into feature space.",
            "- Rendering, DTW/PA-JPE metric computation and prediction saving are excluded.",
            "- nvidia-smi energy is reported when the monitor is available inside the runtime.",
            "- No ground-truth motion is used by the measured inference path.",
        ]
    )
    return "\n".join(rows) + "\n"


def manifest_text(report):
    lines = [
        "# Manifest - P6-K T0-A Style Benchmark",
        "",
        f"Script: `scripts/benchmark_p6k_t0a_style.py`",
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
        "Measures P5-cache, P6-D clean and P6-K clean from the validated P5/T5C cache. "
        "P6-D/P6-K include final hand VQ decode, but metric computation and rendering are excluded.",
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
        raise RuntimeError("No measured samples left after warmup")

    pack_args = SimpleNamespace(config=args.config, default_config=args.default_config, vae_ckpt=args.vae_ckpt)
    pack = load_vq_pack(pack_args, device)
    editor, editor_args = load_p6_editor(args.p6b_checkpoint, device)
    regressor, regressor_args = load_regressor(args.regressor_checkpoint, device)
    selector, selector_args = load_selector(args.selector_checkpoint, device)

    for payload in warmup_payloads:
        p5_cache_variant(payload)
        p6d_variant(payload, editor, regressor, pack, args, device)
        p6k_variant(payload, editor, regressor, selector, pack, args, device)
    synchronize(device)

    variants = []
    gpu_samples = {}
    variant_fns = {
        "p5_cache": lambda payload: p5_cache_variant(payload),
        "p6d_clean": lambda payload: p6d_variant(payload, editor, regressor, pack, args, device),
        "p6k_clean": lambda payload: p6k_variant(payload, editor, regressor, selector, pack, args, device),
    }
    for name, fn in variant_fns.items():
        result, samples = run_variant(name, measured_payloads, fn, args, device)
        variants.append(result)
        gpu_samples[name] = samples
        write_gpu_csv(args.output_dir / f"{name}_gpu.csv", samples)

    by_name = {item["variant"]: item for item in variants}
    overhead = {
        "p6d_minus_p5_cache": by_name["p6d_clean"]["time_per_sample_s"] - by_name["p5_cache"]["time_per_sample_s"],
        "p6k_minus_p5_cache": by_name["p6k_clean"]["time_per_sample_s"] - by_name["p5_cache"]["time_per_sample_s"],
        "p6k_minus_p6d": by_name["p6k_clean"]["time_per_sample_s"] - by_name["p6d_clean"]["time_per_sample_s"],
    }
    quality_reference = QUALITY_REFERENCE if args.dataset == "both" else {args.dataset: QUALITY_REFERENCE[args.dataset]}
    report = {
        "benchmark": "p6k_t0a_style_from_p5_cache",
        "scope": "post_cache_deployable_variants_with_gpu_monitor",
        "dataset": args.dataset,
        "samples": len(measured_payloads),
        "warmup_samples": len(warmup_payloads),
        "cache_dir": str(args.cache_dir),
        "device": str(device),
        "topk": args.topk,
        "budget": args.budget,
        "seed": args.seed,
        "gpu_index": args.gpu_index,
        "monitor_interval": args.monitor_interval,
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
        "variants": variants,
        "overhead": overhead,
        "quality_reference": quality_reference,
        "method_notes": [
            "Starts from validated P5/T5C cache.",
            "P6-D and P6-K include final LH/RH VQ decode.",
            "Metric computation, rendering and prediction saving are excluded.",
            "No ground-truth motion is used by the measured inference path.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown_report(report), encoding="utf-8")
    (args.output_dir / "MANIFEST.md").write_text(manifest_text(report), encoding="utf-8")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
