"""Serving benchmark — measures real data-loading I/O for inference pipelines."""

import os
import time
import random
import argparse
from typing import List
from pathlib import Path

from benchmarks_common.metadata import build_metadata
from benchmarks_common.outputs import write_csv, write_yaml
from benchmarks_common.stats import percentile, safe_mean

from .initialization import init_dist
from .data_generator import generate_dataset


def _discover_samples(data_root: str) -> List[str]:
    """Find all .bin sample files under *data_root*."""
    root = Path(data_root)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("*.bin"))


def _load_sample(path: str, buffer_kb: int = 256) -> float:
    """Read a single sample file and return elapsed time in seconds."""
    buf_size = buffer_kb * 1024
    start = time.perf_counter()
    with open(path, "rb") as f:
        while f.read(buf_size):
            pass
    end = time.perf_counter()
    return max(end - start, 1e-9)


def main():
    parser = argparse.ArgumentParser(
        description="Serving Benchmark — real I/O data loading for inference")
    parser.add_argument("--run-name", type=str, default="demo-run")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--gbs", type=int, default=64, help="global batch size")
    parser.add_argument("--mbs", type=int, default=8, help="micro batch size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="infer")
    parser.add_argument("--data-root", type=str, default="./data/serving",
                        help="Directory with .bin sample files for real I/O")
    parser.add_argument("--auto-generate", type=str, default="true",
                        help="Auto-generate sample data if --data-root is empty")
    parser.add_argument("--sample-count", type=int, default=500,
                        help="Number of samples to auto-generate")
    parser.add_argument("--sample-size-kb", type=int, default=256,
                        help="Size per sample in KiB for auto-generation")
    parser.add_argument("--read-buffer-kb", type=int, default=256,
                        help="Read buffer size in KiB for data loading")
    parser.add_argument("--compute-ms", type=float, default=10.0,
                        help="Simulated compute time per step in ms (0 to skip)")
    parser.add_argument("--outdir", type=str, default="metrics")
    args = parser.parse_args()

    dist = init_dist()
    rank = dist["RANK"]

    # Discover or generate sample files
    samples = _discover_samples(args.data_root)
    if not samples and args.auto_generate.lower() in {"true", "1", "yes"}:
        print(f"No samples found in {args.data_root}, auto-generating "
              f"{args.sample_count} × {args.sample_size_kb} KiB...")
        generate_dataset(args.data_root, args.sample_count,
                         args.sample_size_kb)
        samples = _discover_samples(args.data_root)

    if not samples:
        raise FileNotFoundError(
            f"No .bin files in {args.data_root}. Run data_generator.py first "
            f"or pass --auto-generate true")

    run_dir = os.path.join(args.outdir, args.run_name)
    per_mb_dir = os.path.join(run_dir, "per_microbatch_data_loading_time")
    per_step_dir = os.path.join(run_dir, "per_step_data_loading_time")

    per_mb_rows: List[List] = []
    per_step_rows: List[List] = []
    step_latencies: List[float] = []

    microbatches_per_step = max(1, args.gbs // max(1, args.mbs))

    for step in range(args.steps):
        step_start = time.perf_counter()

        # Real I/O: load random samples for each microbatch
        mb_times: List[float] = []
        for mb in range(microbatches_per_step):
            sample_path = random.choice(samples)
            t = _load_sample(sample_path, args.read_buffer_kb)
            mb_times.append(t)
            per_mb_rows.append([step, mb, f"{t:.6f}"])

        # Simulated compute phase (for pipelining benchmark realism)
        if args.compute_ms > 0:
            time.sleep(args.compute_ms / 1000.0)

        step_latency = time.perf_counter() - step_start
        step_latencies.append(step_latency)
        per_step_rows.append([step, f"{step_latency:.6f}"])

    # Write CSVs
    write_csv(os.path.join(per_mb_dir, f"{rank}.csv"), per_mb_rows)
    write_csv(os.path.join(per_step_dir, f"{rank}.csv"), per_step_rows)

    # Summaries
    mean_step = safe_mean(step_latencies)
    throughput = args.gbs / mean_step if mean_step > 0 else 0.0
    p95 = percentile(step_latencies, 0.95)
    p99 = percentile(step_latencies, 0.99)

    summary = {
        "throughput_samples_per_sec": round(throughput, 4),
        "p95_step_latency_sec": round(p95, 6),
        "p99_step_latency_sec": round(p99, 6),
        "mean_step_latency_sec": round(mean_step, 6),
        "steps": args.steps,
        "microbatches_per_step": microbatches_per_step,
        "samples_available": len(samples),
        "data_root": os.path.abspath(args.data_root),
    }

    write_yaml(os.path.join(run_dir, "metadata.yaml"), build_metadata(
        run_name=args.run_name,
        benchmark="serving",
        parameters={
            "mode": args.mode,
            "steps": args.steps,
            "gbs": args.gbs,
            "mbs": args.mbs,
            "num_workers": args.num_workers,
            "data_root": args.data_root,
            "read_buffer_kb": args.read_buffer_kb,
            "compute_ms": args.compute_ms,
        },
        summary=summary,
    ))


if __name__ == "__main__":
    main()