# Checkpointing Benchmarks — AI I/O & Storage (Platform-agnostic)

Vendor-neutral, reproducible workloads that isolate the storage paths exercised by **model checkpoint write and restore** phases. Use them to profile local SSDs, network filesystems, or object-store mounts before pointing large training jobs at an unfamiliar storage tier.

---

## Why checkpointing throughput and latency matter

- **Training safety nets:** Periodic snapshots protect multi-day runs from preemption or host failure. Slow writes lengthen step times and leave accelerators idle.
- **Restart latency:** Restoring hundreds of gigabytes determines how quickly distributed jobs recover after a failure or planned maintenance.
- **Optimizer and shard layout:** ZeRO, FSDP, and tensor-parallel checkpoints splinter states across many files; metadata thrash and small-file overhead can dominate.
- **Mixed storage tiers:** Modern clusters layer NVMe caches, network-attached POSIX stores, and object storage. Understanding hand-off penalties avoids costly regressions.

This benchmark emulates configurable checkpoint shards, concurrent writers/readers, retention policies, and fsync behavior so you can measure throughput, tail latencies, and amplification without standing up a cluster-scale training job.

---

## Quick start

### 1) Environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r checkpointing_benchmarks/requirements.txt
```

### 2) Run a local demo (write + read)
```bash
bash checkpointing_benchmarks/scripts/run_local_small.sh
```

or directly:
```bash
python -m checkpointing_benchmarks.src.run \
  --run-name cpb-demo \
  --storage-root ./data/checkpoints \
  --iterations 2 \
  --shard-count 8 \
  --shard-size-mb 16 \
  --concurrency 4 \
  --mode write-read
```

### 3) Outputs
```
./metrics/<run-name>/checkpoint_shards.csv       # per-shard timings, throughput, paths
./metrics/<run-name>/checkpoint_iterations.csv   # per-iteration totals (write/read)
./metrics/<run-name>/checkpoint_summary.yaml     # aggregated metrics (p50/p95/p99, throughput)
./metrics/<run-name>/metadata.yaml               # parameters + environment snapshot
```

---

## What this benchmark measures
- **Checkpoint write performance:** Concurrent shard writers with configurable fsync, chunk size, and retention, emulating optimizer state flushes.
- **Restore / read performance:** Sequential or interleaved restores with adjustable buffer sizes to mimic loader threads.
- **Amplification knobs:** Explore the effect of shard counts, shard size, and retention on storage metadata traffic.
- **Throughput vs. latency trade-offs:** Observe how concurrency scaling, cache warm-up, and flushing impact write/read tail latencies.

All outputs are plain CSV/YAML so downstream plotting and comparisons stay transparent.

---

## Configuration & CLI

YAML configs live in `checkpointing_benchmarks/config/`. Key sections:

- `run.name` — labels the output directory under `./metrics/`.
- `storage.root` — directory where synthetic checkpoints are placed (create tmpfs/NVMe mounts to test different tiers).
- `storage.shard_count` / `storage.shard_size_mb` — number and size of shards per checkpoint.
- `benchmark.iterations` — number of checkpoint cycles (write, read, or both).
- `benchmark.mode` — `write`, `read`, or `write-read`.
- `benchmark.concurrency` — threads used for each phase.
- `benchmark.retention` — how many checkpoints to keep on disk before pruning (0 disables pruning).
- `benchmark.fsync` — forces fsync after each shard write.
- `benchmark.chunk_mb` — write chunk size (simulates streaming vs. large buffered writes).
- `benchmark.read_buffer_kb` — buffer size for read loops.
- `benchmark.cleanup_after` — delete generated checkpoints after the run (useful for scratch disks).
- `output.dir` — base directory for metrics artifacts.

All options can also be provided as CLI flags (run `python -m checkpointing_benchmarks.src.run --help`).

---

## Sample report format (synthetic data)

### Scenario overview
| Run ID | Storage tier | Mode | Shards × Size | Concurrency | fsync | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| cpb-nvme-baseline | Local NVMe | write-read | 8 × 16 MiB | 4 | false | Warm cache |
| cpb-nvme-fsync | Local NVMe | write-read | 8 × 16 MiB | 4 | true | Explicit fsync |
| cpb-nfs-shard32 | NFS mount | write-read | 32 × 8 MiB | 8 | false | Metadata-heavy |
| cpb-object-cache | Object storage (cached) | read | 16 × 64 MiB | 6 | false | Manifest warmed |

### Metric snapshot (synthetic)
| Run ID | Write p95 (s) | Write mean MB/s | Read p95 (s) | Read mean MB/s | Bytes/checkpoint (GiB) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| cpb-nvme-baseline | 0.42 | 740 | 0.28 | 910 | 0.125 | Baseline |
| cpb-nvme-fsync | 0.88 | 360 | 0.31 | 885 | 0.125 | Flush amplification |
| cpb-nfs-shard32 | 1.96 | 210 | 1.44 | 260 | 0.256 | Metadata constrained |
| cpb-object-cache | — | — | 0.67 | 470 | 1.000 | Read-only benchmark |

### Chart sketches (replace with real figures)
````text
Write Throughput (MB/s)
cpb-nvme-baseline |████████████████████████████████████ 740
cpb-nvme-fsync    |█████████████████ 360
cpb-nfs-shard32   |██████████ 210

Read Throughput (MB/s)
cpb-nvme-baseline |██████████████████████████████████████ 910
cpb-nvme-fsync    |███████████████████████████████████ 885
cpb-nfs-shard32   |█████████████████ 260
cpb-object-cache  |██████████████████████████████ 470

Write Tail (p95 seconds – lower is better)
cpb-nvme-baseline |███ 0.42
cpb-nvme-fsync    |███████ 0.88
cpb-nfs-shard32   |████████████████████ 1.96
````

> Suggested figure export paths: `docs/figures/cpb_write_throughput.png`, `docs/figures/cpb_read_throughput.png`, `docs/figures/cpb_tail_latency.png`. Include axis units, sample counts, and environment summaries on each figure.

### Reporting checklist
- Attach `metrics/<run-id>/checkpoint_shards.csv`, `checkpoint_iterations.csv`, and `checkpoint_summary.yaml`.
- Capture storage mount options, kernel version, and cache state in `metadata.yaml` or an experiment log.
- Note retention policies, fsync toggles, and differences between cold/warm runs.
- If publishing, include cost and energy estimates derived from throughput deltas when possible.

---

## Typical workflow
1. **Baseline:** Run the default config against your target storage tier (cold and warm cache runs).
2. **Single knob experiments:** Adjust one dimension—shard count, concurrency, fsync, or chunk size—and re-run.
3. **Compare:** Plot throughput and tail latencies across runs to pinpoint bottlenecks or wins.
4. **Document:** Log commands, configs, and environment notes so teammates can reproduce the scenario.

Have ideas for additional checkpoint shapes or integrations? Open an issue or PR—community contributions keep the benchmark neutral and broadly useful.
