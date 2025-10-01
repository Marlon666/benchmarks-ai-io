# Listing Emulated Benchmark (LEB)
*Platform-agnostic, reproducible benchmark for **listing/enumeration** in AI data pipelines — designed to quantify and reduce I/O stalls that degrade model performance in **training** and **batch inference**.*

---

## Why listing matters for model performance

AI pipelines repeatedly enumerate data: building epoch shards, discovering checkpoints/artifacts, scanning prefixes for serving jobs. Slow or jittery listing increases **time-to-first-batch (TTFB)**, depresses **GPU utilization**, inflates **tail latency (P95/P99)**, and ultimately hurts **throughput**, **$ / token**, and **energy per sample**.

LEB isolates the **listing/enumeration path** and measures how tuning (prefetch, page size, concurrency, caching, manifest-based enumeration) changes outcomes. It emits portable CSV/YAML so anyone can re-run, compare and plot without vendor SDKs.

### Core metrics

- **Entries/s** — effective listing throughput
- **TTFB** — time from start until the first usable batch of paths
- **Tail latency** — P95/P99 of list RPCs and end-to-end enumeration
- **Request amplification** — list calls per K entries (pagination/filters)
- **GPU util impact (proxy)** — expected idle reduction when TTFB shrinks
- **$ / token & Energy/sample (proxy)** — derived from runtime changes using public price sheets / documented method

> This benchmark is **public & vendor-neutral**: no proprietary code, no internal links; outputs are local by default.

---

## Quick start

### 1) Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run a small listing demo

Using helper scripts:

```bash
bash scripts/run_local_flat.sh
bash scripts/run_local_deep.sh
```

Or call the Python entry directly:

```bash
python -m src.listing_bench.run \
  --run-name leb-demo \
  --root ./data/emulated_tree \
  --entries-per-dir 2000 \
  --depth 3 \
  --concurrency 8 \
  --page-size 1000 \
  --warm-cache false
```

### 3) Outputs

```
./metrics/<run-name>/listing_raw.csv       # per-call records (start, end, count, path)
./metrics/<run-name>/listing_summary.yaml  # entries/s, P50/P95/P99, TTFB, amplification
./metrics/<run-name>/metadata.yaml         # environment + parameters + summary
```

---

## What LEB measures

- **Listing patterns:** flat vs. deep trees; recursive vs. manifest-based enumeration
- **Caching:** cold vs. warm, optional local metadata cache
- **Concurrency scaling:** threads/processes and saturation analysis
- **Pagination cost:** page size, prefetch distance (simulated for local FS)
- **Negative lookups:** nonexistent prefixes and throttling behavior

### Methods (concise)

- Exclude warm-up from steady-state stats; report TTFB separately.
- Record per-call wall-clock times, compute P95/P99, derive entries/s and amplification.
- GPU utilization/$ / token/energy are proxies derived from TTFB/throughput deltas (method in METHODS.md).
- Figures must be black/white friendly with units, sample size, time window, and an environment table.
- See `METHODS.md` for full protocol and plotting notes.

## CLI & Config

**Common flags:**

- `--run-name` : output subdir under `./metrics/`
- `--root` : root directory to enumerate (created if synthetic tree is enabled)
- `--entries-per-dir`, `--depth` : synthetic tree parameters
- `--concurrency` : lister threads/processes
- `--page-size` : emulated pagination size for iteration
- `--warm-cache` : true|false

YAML examples live in `configs/`:

- `flat_tree.yaml` — many files in few dirs
- `deep_tree.yaml` — deep hierarchy with small pages
- `default.yaml` — sensible defaults for local runs

---

## Interpreting results → model performance

- **Lower TTFB** → faster first batch → fewer idle accelerators during warm-up/epoch reset.
- **Higher entries/s and lower amplification** → fewer metadata calls and less contention → higher steady-state throughput.
- **Lower P95/P99** → fewer long-tail stalls → smoother step times and better GPU utilization.

Together these reduce $ / token and energy per sample.