#!/usr/bin/env python3
"""Run the isolated T0-A AR/P3/P5 benchmark and sample GPU efficiency."""

import csv
import json
import queue
import re
import statistics
import subprocess
import threading
import time
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("SOKENAR_T0A_OUTPUT", ROOT / "artifacts/t0a_efficiency_results"))
SAMPLES = 200
RUNS = {
    "soke_ar": "configs/paper/table3_soke_ar_phoenix200.yaml",
    "p3": "configs/paper/table3_masked_nar_phoenix200.yaml",
    "p5": "configs/paper/table3_handpolish_phoenix200.yaml",
}
REPETITIONS = 3
DATASET = "phoenix"
DOCKER_IMAGE = os.environ.get("SOKENAR_DOCKER_IMAGE", "signgen/soke:local")
DATA_ROOT = Path(os.environ.get("SOKE_DATA_ROOT", ROOT / "data"))
GPU_INDEX = os.environ.get("SOKENAR_GPU_INDEX", "0")


def docker_command(config):
    return [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "--entrypoint", "python",
        "-v", f"{ROOT}:/workspace/SOKENAR",
        "-v", f"{DATA_ROOT}:/workspace/data",
        "-v", f"{DATA_ROOT}:/workspace/SOKE_DATA",
        "-w", "/workspace/SOKENAR", DOCKER_IMAGE,
        "test.py", "--cfg", config, "--use_gpus", "0",
    ]


def gpu_sample():
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i", GPU_INDEX,
            "--query-gpu=timestamp,memory.used,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().split(", ")
    return {
        "wall_time": time.time(),
        "timestamp": query[0],
        "memory_mib": float(query[1]),
        "power_w": float(query[2]),
        "utilization_pct": float(query[3]),
    }


def reader(stream, lines):
    for line in iter(stream.readline, ""):
        print(line, end="", flush=True)
        lines.put(line)


def run_one(name, config, repetition):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        docker_command(config),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = queue.Queue()
    thread = threading.Thread(target=reader, args=(process.stdout, lines), daemon=True)
    thread.start()
    active = False
    start = None
    end = None
    output_lines = []
    samples = []

    while process.poll() is None or not lines.empty():
        try:
            line = lines.get(timeout=0.05)
            output_lines.append(line)
            if "Inference started" in line:
                active = True
                start = time.time()
            if "Inference finished" in line:
                end = time.time()
                active = False
        except queue.Empty:
            pass
        if active:
            try:
                samples.append(gpu_sample())
            except subprocess.CalledProcessError:
                pass
            time.sleep(0.05)

    thread.join(timeout=1)
    if process.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {process.returncode}")
    if start is None or end is None or not samples:
        raise RuntimeError(f"{name} did not expose inference markers or GPU samples")

    duration = end - start
    energy_j = sum(
        0.5 * (left["power_w"] + right["power_w"])
        * (right["wall_time"] - left["wall_time"])
        for left, right in zip(samples, samples[1:])
    )
    joined = "".join(output_lines)
    metrics = re.compile(
        rf"'Metrics/{DATASET}_DTW_PA_JPE_body/mean': '([^']+)'.*"
        rf"'Metrics/{DATASET}_DTW_PA_JPE_hand/mean': '([^']+)'"
    )
    matches = metrics.findall(joined)
    body, hand = matches[-1] if matches else (None, None)
    result = {
        "model": name,
        "config": config,
        "samples": SAMPLES,
        "duration_s": duration,
        "throughput_samples_s": SAMPLES / duration,
        "time_per_sample_s": duration / SAMPLES,
        "peak_memory_mib": max(sample["memory_mib"] for sample in samples),
        "mean_power_w": sum(sample["power_w"] for sample in samples) / len(samples),
        "peak_power_w": max(sample["power_w"] for sample in samples),
        "gpu_energy_j": energy_j,
        "gpu_energy_j_per_sample": energy_j / SAMPLES,
        "mean_utilization_pct": sum(sample["utilization_pct"] for sample in samples) / len(samples),
        "pa_body": None if body is None else float(body),
        "pa_hand": None if hand is None else float(hand),
        "gpu_samples": len(samples),
    }
    artifact = f"{name}_r{repetition}"
    result["repetition"] = repetition
    (OUTPUT / f"{artifact}.log").write_text(joined, encoding="utf-8")
    with (OUTPUT / f"{artifact}_gpu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)
    (OUTPUT / f"{artifact}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    results = [
        run_one(name, config, repetition)
        for repetition in range(1, REPETITIONS + 1)
        for name, config in RUNS.items()
    ]
    ar_by_repetition = {
        result["repetition"]: result for result in results if result["model"] == "soke_ar"
    }
    for result in results:
        ar = ar_by_repetition[result["repetition"]]
        result["speedup_vs_ar"] = ar["duration_s"] / result["duration_s"]
        result["energy_ratio_vs_ar"] = result["gpu_energy_j"] / ar["gpu_energy_j"]
    fields = [
        "duration_s", "throughput_samples_s", "time_per_sample_s",
        "peak_memory_mib", "mean_power_w", "peak_power_w", "gpu_energy_j",
        "gpu_energy_j_per_sample", "mean_utilization_pct", "speedup_vs_ar",
        "energy_ratio_vs_ar",
    ]
    aggregate = []
    for name in RUNS:
        group = [result for result in results if result["model"] == name]
        row = {"model": name, "repetitions": REPETITIONS}
        for field in fields:
            values = [result[field] for result in group]
            row[f"{field}_mean"] = statistics.mean(values)
            row[f"{field}_std"] = statistics.stdev(values)
        row["pa_body"] = group[0]["pa_body"]
        row["pa_hand"] = group[0]["pa_hand"]
        aggregate.append(row)
    (OUTPUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUTPUT / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
