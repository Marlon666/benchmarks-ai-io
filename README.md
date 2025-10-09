# benchmarks-ai-io
Open, reproducible benchmarks and practical recipes to surface storage and data-path bottlenecks in AI training and large-scale inference—whether you have a rack of GPUs or only CPUs and a laptop.

## Project vision
- Build a neutral, scriptable harness that emulates common AI workloads and isolates the storage layer.
- Produce portable CSV/YAML outputs so teams can compare runs across clouds, filesystems, and hardware.
- Provide “good, better, best” optimization playbooks that translate benchmark wins into model performance and $/token savings.

## How AI workloads stress storage
Modern models mix bursty metadata traffic with sustained throughput requirements. The balance shifts depending on whether you are training or serving, creating different stress profiles for the I/O stack.

### Training profile
- **Checkpointing & snapshots:** Periodic multi-GB writes (and reads for restarts) demand fast, durable storage; inconsistent latency leads to idle accelerators.
- **Dataset enumeration:** Epoch resets or shuffle stages have to list and shard millions of files; metadata latency inflates time-to-first-batch.
- **Streaming & prefetch:** Background workers pull data continuously; they rely on predictable bandwidth and low jitter to keep device queues full.
- **Gradient logging & artifacts:** Auxiliary outputs (metrics, tensor dumps) create write amplification that competes with primary data reads.

### Inference / bulk decoding profile
- **Request fan-out:** Serving pipelines fetch model weights, tokenizer state, and prompt/context files; cold starts punish metadata-heavy workloads.
- **Hot/cold splits:** Popular models or prefixes live in cache while long-tail requests round-trip to cloud storage; caching strategy defines tail latencies.
- **Batch assembly:** Micro-batching and sequence packing touch many small objects, which stresses metadata services differently than streaming large shards.

## Critical I/O behaviors to measure
- **Listing & metadata fan-out:** Determines time-to-first-batch and controls the number of RPCs the filesystem or object store must service. Small directories vs. deep trees show different scaling characteristics.
- **Checkpoint throughput & replay:** Drives how quickly training recovers from failures and whether mid-epoch checkpoints starve compute.
- **Data loading & prefetch depth:** Captures the end-to-end impact of loaders, compression, shuffling, and caching on steady-state throughput.
- **Intermediate artifact handling:** Evaluates how logs, embeddings, and temporary tensors interact with burst buffers or local SSD tiers.

## Framework and model considerations
Different stacks exercise storage in distinct ways:
- **PyTorch DataLoader / DataPipes:** Configurable prefetch, pinned memory, and shared-memory queues can hide latency or amplify it if mis-sized.
- **TensorFlow tf.data / JAX input pipelines:** Vectorized graph execution may issue wider prefetch bursts, demanding higher baseline throughput.
- **DeepSpeed, Megatron, FSDP, ZeRO:** Sharded checkpoints and optimizer states increase metadata operations and require aligned partition storage.
- **Diffusion vs. autoregressive models:** Diffusion workloads stream large images/audio; autoregressive decoders juggle many small JSON/token files.
Benchmarking these combinations reveals whether throttling stems from metadata, throughput, or application-level settings.

## Cloud-provider storage optimizations (awareness checklist)
- **AWS:** S3 Express One Zone reduces metadata latency; S3 Multi-GET and Transfer Acceleration improve parallel reads; FSx for Lustre and FSx for ONTAP offer POSIX semantics with warm caches; EBS io2 and instance-store SSDs provide consistent checkpoint bandwidth.
- **Google Cloud:** GCS Turbo Replication and Requester Pays tuning affect listing throughput; Filestore (Zonal & High Scale), Cloud Storage FUSE with caching, and local SSDs attached to GCE or TPU VMs reduce cold-start penalties.
- **Azure:** Premium SSD v2 and Ultra Disk sustain high checkpoint IOPS; Azure Blob Storage with hierarchical namespaces optimizes metadata scans; Azure NetApp Files and HPC Cache bridge NFS semantics with object storage backends.
The benchmarks in this repository let you validate how these features behave under your workload shapes, quantify tail behavior, and test tuning (prefetch, pagination, sharding) before touching production clusters.

## Benchmarks in this repository
- **Listing Emulated Benchmark (LEB)** — located in `listing_folder_benchmarks/`. Focuses on metadata latency, pagination cost, and enumeration concurrency. Ideal for sizing manifest caches, comparing filesystem mounts, or validating new object-store regions.
- **Serving Benchmarks** — located in `serving_benchmarks/`. Emulates batch inference workloads, measuring microbatch data-loader latency, tail behavior, and the knock-on effects on GPU utilization and cost.
- **Checkpointing Benchmarks** — located in `checkpointing_benchmarks/`. Simulates shard-write/read cycles with tunable concurrency, fsync, and retention policies to expose throughput limits and tail latencies across storage tiers.
- **Upcoming modules (roadmap)** — distributed shuffle loaders, streaming dataset replay, and mixed workload stressors that combine listing + checkpoint + streaming phases.

Each module is designed to run on commodity hardware without GPUs, yet scales to GPU-backed clusters when you want to observe device utilization side-by-side.

## Getting started
1. Clone the repo and create a Python environment (3.10+ recommended):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r listing_folder_benchmarks/requirements.txt
   pip install -r serving_benchmarks/requirements.txt
   ```
2. Choose a benchmark:
   - Listing: see `listing_folder_benchmarks/README.md` for configs such as `config/flat_tree.yaml`.
   - Serving: see `serving_benchmarks/README.md` and associated scripts in `serving_benchmarks/scripts/`.
3. Run locally or on a remote machine (GPU optional). All results land under `./metrics/<run-name>/`.

Benchmarks emit structured outputs (CSV + YAML) that capture per-call timings, tail percentiles, throughput, configuration knobs, and environment metadata. Version these artifacts or feed them into dashboards to track regressions.

## How this project helps you
- **Validate storage tiers before deployment:** Measure listing latency on S3 vs. FSx, or compare Azure Blob tiers with and without hierarchical namespaces.
- **Tune application knobs safely:** Experiment with PyTorch DataLoader workers, tf.data pipeline fusion, or checkpoint intervals and observe real latency/throughput changes.
- **Quantify ROI of provider features:** Demonstrate the impact of AWS S3 Express, GCS metadata caching, or Azure Premium SSD upgrades on time-to-first-batch and $/token.
- **Share reproducible results:** Use the portable outputs and configuration files to publish, compare, or reproduce runs across teams and vendors.

## Roadmap & contributions
- Near-term additions: checkpoint churn simulator, streaming dataloader stress tests, and mixed workloads that model full training steps.
- Contributions are welcome—open an issue with the workload you want to emulate or submit a PR with a new module/config.
- All code is released under the `LICENSE` file in this repo; please review before contributing.

## Support & feedback
Questions, tuning tips, or benchmark requests? File an issue or start a discussion. Real-world traces and anonymized metrics help us refine scenarios and defaults for other practitioners.
