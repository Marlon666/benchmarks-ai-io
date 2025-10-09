# Serving Benchmarks — AI I/O & Storage (Platform-agnostic)

Vendor-neutral, reproducible benchmarks for **batch inference / serving** pipelines, focused on **data I/O and storage**. The goal is to quantify and reduce I/O bottlenecks so accelerators spend more time on useful compute and less time waiting for data.

**Core metrics**

- **Throughput** — requests/s or tokens/s  
- **Tail latency** — P95 / P99 of per-request or per-step latency  
- **GPU utilization** — if available  
- **$ / token** — estimated from public price sheets × measured runtime  
- **Energy per sample** — simple, documented estimation method

> This component is **public and vendor-neutral**: no proprietary code, no internal links, and all outputs write to the local `./metrics/` directory by default.

---

## Quick start

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run a small serving demo

Use the helper script:

```bash
bash serving_benchmarks/scripts/run_infer_local.sh
```

Or call the Python entry directly:

```bash
python serving_benchmarks/src/train.py \
  --run-name demo-infer \
  --steps 200 \
  --gbs 256 \
  --mbs 1 \
  --num-workers 4 \
  --mode infer
```

### 3) Outputs

After a run you should see:

- `./metrics/<run-name>/per_microbatch_data_loading_time/<rank>.csv`
- `./metrics/<run-name>/per_step_data_loading_time/<rank>.csv`
- `./metrics/<run-name>/metadata.yaml`

  - CSVs contain per-microbatch / per-step timings.
  - `metadata.yaml` records run name, timestamp, environment, summary metrics.

---

## What this benchmark measures (Serving focus)

This benchmark emulates (or integrates with) a serving pipeline to capture the end-to-end impact of data access on inference. It records:

- Fetch path behavior (object storage / local SSD)
- Sharding & prefetch effects on hotspots and read amplification
- Jitter / tail effects at steady load
- Overall efficiency (throughput, GPU utilization) and cost/energy estimates

Results are saved as portable CSV/YAML so third parties can regenerate plots without vendor SDKs.

---

## Configuration & CLI

Configs live in `serving_benchmarks/config/` (add your own). Most options are also CLI flags.

**Common flags:**

- `--run-name`: subdirectory under `./metrics/`
- `--steps`: number of iterations
- `--gbs` / `--mbs`: global / micro batch sizes (for request batching)
- `--num-workers`: data-loader workers
- `--mode`: infer (serving; default here). A training demo exists but is out of scope for this component.

**Environment overrides (optional):**

- `MASTER_ADDR`, `MASTER_PORT`, `DIST_BACKEND` — basic distributed init knobs (default single process)

---

## Methods (concise)

- Exclude warm-up steps from statistics (configurable).
- Record per-microbatch and per-step wall-clock times → compute P95/P99.
- Derive throughput from measured latencies and effective batch size.
- GPU utilization is recorded if available; otherwise omitted.
- $ / token comes from measured runtime × public price sheets (cite your source in notes).
- Energy per sample via on-node telemetry or documented conversion factors; always state the window/assumptions.

For a full protocol (definitions, plotting, black-&-white-friendly figure templates), keep or add a project-level `METHODS.md`.

---

## Typical workflow

1. Baseline — run the default config and export CSV/YAML + metadata.
2. Enable a single knob — e.g., prefetch or alternative sharding; re-run.
3. Compare — generate two black-and-white figures (throughput, P95/P99), each one per page with units, sample sizes, and environment table.
4. Document — note the change and command line in your experiment log (e.g., `CHANGELOG.md`).
---

## Sample report format

### Scenario overview
| Run ID | Hardware | Model size | Global batch | Loader workers | Storage tier | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| infer-baseline-cpu | 32-core CPU | 7B | 256 | 4 | NFS mount | Cold start, no caching |
| infer-cached-gpu | 2×A100 | 13B | 512 | 8 | NVMe cache + S3 | Weights preloaded |
| infer-high-tail | 2×A100 | 13B | 512 | 8 | S3 (no cache) | Simulated throttling |
| infer-optimized | 2×A100 | 13B | 512 | 8 | NVMe cache + S3 | Async prefetch + packed batches |

### Metric snapshot (synthetic)
| Run ID | Throughput (tokens/s) | P95 latency (ms) | P99 latency (ms) | GPU util (%) | $ / 1M tokens (est.) | Energy / 1K samples (kJ) |
| --- | --- | --- | --- | --- | --- | --- |
| infer-baseline-cpu | 18,200 | 152 | 207 | 0 (CPU only) | 9.10 | 4.8 |
| infer-cached-gpu | 142,000 | 48 | 61 | 82 | 2.45 | 1.6 |
| infer-high-tail | 97,500 | 141 | 213 | 63 | 3.62 | 2.1 |
| infer-optimized | 158,400 | 39 | 52 | 88 | 2.18 | 1.4 |

### Chart sketches (replace with real figures)
````text
Throughput (tokens/s)
infer-optimized    |██████████████████████████████████████ 158k
infer-cached-gpu   |███████████████████████████████ 142k
infer-high-tail    |█████████████████████ 97k
infer-baseline-cpu |████ 18k

Latency P95 (ms) – lower is better
infer-optimized    |███ 39
infer-cached-gpu   |████ 48
infer-high-tail    |██████████████████████ 141
infer-baseline-cpu |███████████████████████ 152

GPU Utilization (%)
infer-optimized    |███████████████████████████████████ 88
infer-cached-gpu   |█████████████████████████████ 82
infer-high-tail    |███████████████████ 63
infer-baseline-cpu | 0
````

### Reporting checklist
- Attach `metrics/<run-id>/per_microbatch_data_loading_time/*.csv`, `per_step_data_loading_time/*.csv`, and `metadata.yaml`.
- Document model card, tokenizer settings, and prompt mix used for the run.
- Capture any throttling, cache warm-up, or retry events in `EXPERIMENT_LOG.md`.
