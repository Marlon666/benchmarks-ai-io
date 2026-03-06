# Dataloader Benchmarks — AI I/O & Storage

Vendor-neutral, reproducible benchmarks for **data-loading pipelines** in AI training, focused on isolating storage throughput and latency patterns that affect accelerator utilization.

---

## Why data loading I/O matters

- **Starvation:** Slow or jittery reads starve GPUs/TPUs, creating idle bubbles that waste compute and money.
- **Shuffle overhead:** Random-access reads during epoch shuffling can be orders of magnitude slower than sequential scans.
- **Prefetch depth:** Too little prefetch → device stalls; too much → memory pressure and cache thrash.
- **Compression trade-offs:** Decompression burns CPU but reduces storage bandwidth needs.

This benchmark measures four read strategies (sequential, random, mmap, prefetch) to quantify these trade-offs on your target storage.

---

## Quick start

### 1) Environment
```bash
pip install -e ".[dev]"  # from repo root
```

### 2) Run a local demo
```bash
bash dataloader_benchmarks/scripts/run_local.sh
```

Or directly:
```bash
python -m dataloader_benchmarks.src.run \
  --run-name dl-demo \
  --data-root ./data/dataloader \
  --strategy sequential \
  --epochs 2 \
  --auto-generate true \
  --sample-count 200 \
  --sample-size-kb 512
```

### 3) Outputs
```
./metrics/<run-name>/loader_samples.csv    # per-sample timings
./metrics/<run-name>/loader_epochs.csv     # per-epoch aggregates
./metrics/<run-name>/loader_summary.yaml   # p50/p95/p99, throughput, TTFB
./metrics/<run-name>/metadata.yaml         # parameters + environment
```

---

## Read strategies

| Strategy | Description | Models |
| --- | --- | --- |
| `sequential` | Read files in order | Baseline scan, streaming datasets |
| `random` | Shuffle then read (seeded) | Epoch shuffling, worst-case HDDs |
| `mmap` | Memory-mapped zero-copy reads | DataLoader with mmap, page cache |
| `prefetch` | ThreadPoolExecutor concurrent reads | PyTorch DataLoader workers, tf.data |

---

## Configuration & CLI

Configs live in `dataloader_benchmarks/config/`. Key CLI flags:

- `--strategy`: `sequential` | `random` | `mmap` | `prefetch`
- `--epochs`: Number of full passes over the dataset
- `--prefetch-depth`: Worker threads (for `prefetch` strategy)
- `--read-buffer-kb`: Read buffer size
- `--compressed`: Expect gzip-compressed `.bin.gz` samples
- `--auto-generate`: Create synthetic data if none exists
- `--sample-count` / `--sample-size-kb`: Synthetic dataset dimensions

---

## Key metrics

- **Samples/s** — per-epoch effective throughput
- **TTFB** — time to first batch (first sample loaded)
- **Per-sample p50/p95/p99** — tail latency distribution
- **Epoch throughput (MB/s)** — aggregate bandwidth

---

## Typical workflow

1. **Generate data**: Use `--auto-generate` or `dataset_gen.py` with target sample sizes.
2. **Baseline**: Run `--strategy sequential` to establish baseline throughput.
3. **Compare strategies**: Run each strategy and compare TTFB and tail latencies.
4. **Tune prefetch**: Vary `--prefetch-depth` to find the saturation point.
5. **Compression**: Compare uncompressed vs. `--compressed` to measure CPU/bandwidth trade-off.
