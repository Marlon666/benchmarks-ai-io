import argparse
import os
from typing import Any, Dict, List

from benchmarks_common.cli import load_yaml_config, parse_bool
from benchmarks_common.metadata import build_metadata
from benchmarks_common.outputs import write_csv, write_yaml
from benchmarks_common.stats import percentile, safe_mean, safe_median

from .checkpoint_runner import (BenchmarkParams, CheckpointingBenchmark,
                                IterationRecord, ShardRecord)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="cpb-run")
    parser.add_argument("--storage-root", type=str, default="./data/checkpoints")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--shard-size-mb", type=float, default=16.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--mode", type=str, default="write-read")
    parser.add_argument("--retention", type=int, default=2)
    parser.add_argument("--fsync", type=str, default="false")
    parser.add_argument("--chunk-mb", type=float, default=4.0)
    parser.add_argument("--read-buffer-kb", type=int, default=1024)
    parser.add_argument("--cleanup-after", type=str, default="false")
    parser.add_argument("--io-engine", type=str, default="sync",
                        choices=["sync", "async"],
                        help="IO engine: sync (default) or async (aiofiles)")
    parser.add_argument("--outdir", type=str, default="metrics")
    args = parser.parse_args()

    if args.config:
        cfg = load_yaml_config(args.config)
        args.run_name = cfg.get("run", {}).get("name", args.run_name)
        args.storage_root = cfg.get("storage", {}).get("root", args.storage_root)
        args.shard_count = cfg.get("storage",
                                   {}).get("shard_count", args.shard_count)
        args.shard_size_mb = cfg.get("storage",
                                     {}).get("shard_size_mb", args.shard_size_mb)
        args.iterations = cfg.get("benchmark",
                                  {}).get("iterations", args.iterations)
        args.concurrency = cfg.get("benchmark",
                                   {}).get("concurrency", args.concurrency)
        args.mode = cfg.get("benchmark", {}).get("mode", args.mode)
        args.retention = cfg.get("benchmark",
                                 {}).get("retention", args.retention)
        args.fsync = cfg.get("benchmark", {}).get("fsync", args.fsync)
        args.chunk_mb = cfg.get("benchmark", {}).get("chunk_mb", args.chunk_mb)
        args.read_buffer_kb = cfg.get("benchmark",
                                      {}).get("read_buffer_kb", args.read_buffer_kb)
        args.cleanup_after = cfg.get("benchmark",
                                     {}).get("cleanup_after", args.cleanup_after)
        args.io_engine = cfg.get("benchmark",
                                 {}).get("io_engine", args.io_engine)
        args.outdir = cfg.get("output", {}).get("dir", args.outdir)

    params = BenchmarkParams(
        run_name=args.run_name,
        root=args.storage_root,
        iterations=max(1, int(args.iterations)),
        shard_count=max(1, int(args.shard_count)),
        shard_size_mb=float(args.shard_size_mb),
        concurrency=max(1, int(args.concurrency)),
        fsync=parse_bool(args.fsync),
        mode=str(args.mode),
        retention=max(0, int(args.retention)),
        chunk_mb=max(0.1, float(args.chunk_mb)),
        read_buffer_kb=max(1, int(args.read_buffer_kb)),
        cleanup_after=parse_bool(args.cleanup_after),
        io_engine=str(args.io_engine),
    )

    benchmark = CheckpointingBenchmark(params)
    shard_records, iteration_records = benchmark.run()

    shard_csv = _to_shard_rows(shard_records)
    iter_csv = _to_iteration_rows(iteration_records)

    run_dir = os.path.join(args.outdir, args.run_name)
    shards_path = os.path.join(run_dir, "checkpoint_shards.csv")
    iterations_path = os.path.join(run_dir, "checkpoint_iterations.csv")
    summary_path = os.path.join(run_dir, "checkpoint_summary.yaml")
    meta_path = os.path.join(run_dir, "metadata.yaml")

    write_csv(shards_path, shard_csv)
    write_csv(iterations_path, iter_csv)

    summary = _build_summary(iteration_records, shard_records, params)
    write_yaml(summary_path, summary)
    write_yaml(meta_path, build_metadata(
        run_name=args.run_name,
        benchmark="checkpointing",
        parameters={
            "storage_root": args.storage_root,
            "iterations": params.iterations,
            "shard_count": params.shard_count,
            "shard_size_mb": params.shard_size_mb,
            "concurrency": params.concurrency,
            "mode": params.mode,
            "retention": params.retention,
            "fsync": params.fsync,
            "chunk_mb": params.chunk_mb,
            "read_buffer_kb": params.read_buffer_kb,
            "cleanup_after": params.cleanup_after,
            "io_engine": params.io_engine,
        },
        summary=summary,
    ))


def _to_shard_rows(records: List[ShardRecord]) -> List[List[Any]]:
    rows = [[
        "iteration",
        "phase",
        "shard_id",
        "bytes",
        "duration_sec",
        "throughput_mb_s",
        "path",
    ]]
    for record in records:
        rows.append([
            record.iteration,
            record.phase,
            record.shard_id,
            record.bytes,
            round(record.duration_sec, 6),
            round(record.throughput_mb_s, 2)
            if record.throughput_mb_s != float("inf") else float("inf"),
            record.path,
        ])
    return rows


def _to_iteration_rows(records: List[IterationRecord]) -> List[List[Any]]:
    rows = [[
        "iteration",
        "phase",
        "duration_sec",
        "total_bytes",
        "throughput_mb_s",
    ]]
    for record in records:
        rows.append([
            record.iteration,
            record.phase,
            round(record.duration_sec, 6),
            record.total_bytes,
            round(record.throughput_mb_s, 2)
            if record.throughput_mb_s != float("inf") else float("inf"),
        ])
    return rows


def _build_summary(iteration_records: List[IterationRecord],
                   shard_records: List[ShardRecord],
                   params: BenchmarkParams) -> Dict[str, Any]:
    write_iters = [i for i in iteration_records if i.phase == "write"]
    read_iters = [i for i in iteration_records if i.phase == "read"]
    write_shards = [s for s in shard_records if s.phase == "write"]
    read_shards = [s for s in shard_records if s.phase == "read"]

    def _durations(items):
        return [itm.duration_sec for itm in items]

    summary: Dict[str, Any] = {
        "mode": params.mode,
        "io_engine": params.io_engine,
        "iterations_scheduled": params.iterations,
        "write_iterations": len(write_iters),
        "read_iterations": len(read_iters),
        "shards_per_iteration": params.shard_count,
        "bytes_per_shard": int(params.shard_size_mb * 1024 * 1024),
        "concurrency": params.concurrency,
        "fsync_enabled": params.fsync,
        "retention": params.retention,
    }

    if write_iters:
        durations = _durations(write_iters)
        throughputs = [it.throughput_mb_s for it in write_iters]
        summary.update({
            "total_bytes_written": int(sum(it.total_bytes for it in write_iters)),
            "write_p50_sec": round(safe_median(durations), 6),
            "write_p95_sec": round(percentile(durations, 0.95), 6),
            "write_p99_sec": round(percentile(durations, 0.99), 6),
            "write_avg_throughput_mb_s":
            round(safe_mean(throughputs), 2),
        })
    else:
        summary.update({
            "total_bytes_written": 0,
            "write_p50_sec": 0.0,
            "write_p95_sec": 0.0,
            "write_p99_sec": 0.0,
            "write_avg_throughput_mb_s": 0.0,
        })

    if read_iters:
        durations = _durations(read_iters)
        throughputs = [it.throughput_mb_s for it in read_iters]
        summary.update({
            "total_bytes_read": int(sum(it.total_bytes for it in read_iters)),
            "read_p50_sec": round(safe_median(durations), 6),
            "read_p95_sec": round(percentile(durations, 0.95), 6),
            "read_p99_sec": round(percentile(durations, 0.99), 6),
            "read_avg_throughput_mb_s":
            round(safe_mean(throughputs), 2),
        })
    else:
        summary.update({
            "total_bytes_read": 0,
            "read_p50_sec": 0.0,
            "read_p95_sec": 0.0,
            "read_p99_sec": 0.0,
            "read_avg_throughput_mb_s": 0.0,
        })

    if write_shards:
        shard_durations = [s.duration_sec for s in write_shards]
        summary.update({
            "write_shard_p95_sec": round(percentile(shard_durations, 0.95), 6),
            "write_shard_p99_sec": round(percentile(shard_durations, 0.99), 6),
        })
    else:
        summary.update({"write_shard_p95_sec": 0.0, "write_shard_p99_sec": 0.0})

    if read_shards:
        shard_durations = [s.duration_sec for s in read_shards]
        summary.update({
            "read_shard_p95_sec": round(percentile(shard_durations, 0.95), 6),
            "read_shard_p99_sec": round(percentile(shard_durations, 0.99), 6),
        })
    else:
        summary.update({"read_shard_p95_sec": 0.0, "read_shard_p99_sec": 0.0})

    return summary


if __name__ == "__main__":
    main()
