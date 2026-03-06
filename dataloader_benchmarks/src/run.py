"""Dataloader benchmark entry point."""

import argparse
import os
from typing import Any, Dict, List

from benchmarks_common.cli import load_yaml_config, parse_bool
from benchmarks_common.metadata import build_metadata
from benchmarks_common.outputs import write_csv, write_yaml
from benchmarks_common.stats import percentile, safe_mean, safe_median

from .dataset_gen import generate_dataset
from .loader import EpochRecord, LoaderParams, SampleRecord, run_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataloader Benchmark — streaming data-loading I/O for training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="dl-run")
    parser.add_argument("--data-root", type=str, default="./data/dataloader")
    parser.add_argument("--strategy", type=str, default="sequential",
                        choices=["sequential", "random", "mmap", "prefetch"],
                        help="Read strategy to benchmark")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--prefetch-depth", type=int, default=4,
                        help="Worker threads for prefetch strategy")
    parser.add_argument("--read-buffer-kb", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--compressed", type=str, default="false",
                        help="Expect gzip-compressed samples")
    parser.add_argument("--auto-generate", type=str, default="true")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--sample-size-kb", type=int, default=512)
    parser.add_argument("--outdir", type=str, default="metrics")
    args = parser.parse_args()

    if args.config:
        cfg = load_yaml_config(args.config)
        args.run_name = cfg.get("run", {}).get("name", args.run_name)
        args.data_root = cfg.get("data", {}).get("root", args.data_root)
        args.strategy = cfg.get("benchmark", {}).get("strategy", args.strategy)
        args.epochs = cfg.get("benchmark", {}).get("epochs", args.epochs)
        args.prefetch_depth = cfg.get("benchmark", {}).get("prefetch_depth", args.prefetch_depth)
        args.read_buffer_kb = cfg.get("benchmark", {}).get("read_buffer_kb", args.read_buffer_kb)
        args.batch_size = cfg.get("benchmark", {}).get("batch_size", args.batch_size)
        args.compressed = str(cfg.get("benchmark", {}).get("compressed", args.compressed))
        args.outdir = cfg.get("output", {}).get("dir", args.outdir)

    compressed = parse_bool(args.compressed)

    # Auto-generate data if needed
    from .loader import discover_samples
    samples = discover_samples(args.data_root, compressed)
    if not samples and parse_bool(args.auto_generate):
        print(f"Auto-generating {args.sample_count} × {args.sample_size_kb} KiB "
              f"samples in {args.data_root}...")
        generate_dataset(args.data_root, args.sample_count,
                         args.sample_size_kb, compress=compressed)

    params = LoaderParams(
        data_root=args.data_root,
        strategy=args.strategy,
        epochs=max(1, args.epochs),
        prefetch_depth=max(1, args.prefetch_depth),
        read_buffer_kb=max(1, args.read_buffer_kb),
        batch_size=max(1, args.batch_size),
        shuffle_seed=args.shuffle_seed,
        compressed=compressed,
    )

    sample_records, epoch_records = run_loader(params)

    # Output paths
    run_dir = os.path.join(args.outdir, args.run_name)
    samples_csv = os.path.join(run_dir, "loader_samples.csv")
    epochs_csv = os.path.join(run_dir, "loader_epochs.csv")
    summary_yaml = os.path.join(run_dir, "loader_summary.yaml")
    meta_yaml = os.path.join(run_dir, "metadata.yaml")

    write_csv(samples_csv, _sample_rows(sample_records))
    write_csv(epochs_csv, _epoch_rows(epoch_records))

    summary = _build_summary(sample_records, epoch_records, params)
    write_yaml(summary_yaml, summary)
    write_yaml(meta_yaml, build_metadata(
        run_name=args.run_name,
        benchmark="dataloader",
        parameters={
            "data_root": args.data_root,
            "strategy": params.strategy,
            "epochs": params.epochs,
            "prefetch_depth": params.prefetch_depth,
            "read_buffer_kb": params.read_buffer_kb,
            "batch_size": params.batch_size,
            "compressed": params.compressed,
        },
        summary=summary,
    ))


def _sample_rows(records: List[SampleRecord]) -> List[List[Any]]:
    header = ["epoch", "sample_idx", "path", "bytes_read",
              "duration_sec", "throughput_mb_s"]
    rows = [header]
    for r in records:
        rows.append([
            r.epoch, r.sample_idx, r.path, r.bytes_read,
            round(r.duration_sec, 6),
            round(r.throughput_mb_s, 2) if r.throughput_mb_s != float("inf") else "inf",
        ])
    return rows


def _epoch_rows(records: List[EpochRecord]) -> List[List[Any]]:
    header = ["epoch", "strategy", "samples", "total_bytes",
              "duration_sec", "throughput_mb_s", "ttfb_sec"]
    rows = [header]
    for r in records:
        rows.append([
            r.epoch, r.strategy, r.samples, r.total_bytes,
            round(r.duration_sec, 6),
            round(r.throughput_mb_s, 2) if r.throughput_mb_s != float("inf") else "inf",
            round(r.ttfb_sec, 6),
        ])
    return rows


def _build_summary(
    sample_records: List[SampleRecord],
    epoch_records: List[EpochRecord],
    params: LoaderParams,
) -> Dict[str, Any]:
    durations = [r.duration_sec for r in sample_records]
    throughputs = [r.throughput_mb_s for r in epoch_records]
    ttfbs = [r.ttfb_sec for r in epoch_records]

    return {
        "strategy": params.strategy,
        "epochs": params.epochs,
        "total_samples_read": len(sample_records),
        "sample_p50_sec": round(safe_median(durations), 6),
        "sample_p95_sec": round(percentile(durations, 0.95), 6),
        "sample_p99_sec": round(percentile(durations, 0.99), 6),
        "mean_epoch_throughput_mb_s": round(safe_mean(throughputs), 2),
        "mean_ttfb_sec": round(safe_mean(ttfbs), 6),
        "prefetch_depth": params.prefetch_depth,
        "read_buffer_kb": params.read_buffer_kb,
        "compressed": params.compressed,
    }


if __name__ == "__main__":
    main()
