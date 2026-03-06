# benchmarks-ai-io
Open, reproducible benchmarks and practical recipes to surface storage and data-path bottlenecks in AI training and large-scale inference—whether you have a rack of GPUs or only CPUs and a laptop.

## Project vision
- Build a neutral, scriptable harness that emulates common AI workloads and isolates the storage layer.
- Produce portable CSV/YAML outputs so teams can compare runs across clouds, filesystems, and hardware.
- Provide "good, better, best" optimization playbooks that translate benchmark wins into model performance and $/token savings.

## How AI workloads stress storage
Modern models mix bursty metadata traffic with sustained throughput requirements. The balance shifts depending on whether you are training or serving, creating different stress profiles for the I/O stack.

### Training profile
- **Checkpointing & snapshots:** Periodic multi-GB writes (and reads for restarts) demand fast, durable storage; inconsistent latency leads to idle accelerators.
- **Dataset enumeration:** Epoch resets or shuffle stages have to list and shard millions of files; metadata latency inflates time-to-first-batch.
- **Data loading & prefetch:** Background workers pull data continuously using sequential, random, or memory-mapped reads; they rely on predictable bandwidth and low jitter to keep device queues full.
- **Streaming & shuffle:** Random-access reads during epoch shuffling can be orders of magnitude slower than sequential scans—a key bottleneck for large datasets.
- **Gradient logging & artifacts:** Auxiliary outputs (metrics, tensor dumps) create write amplification that competes with primary data reads.

### Inference / bulk decoding profile
- **Request fan-out:** Serving pipelines fetch model weights, tokenizer state, and prompt/context files; cold starts punish metadata-heavy workloads.
- **Hot/cold splits:** Popular models or prefixes live in cache while long-tail requests round-trip to cloud storage; caching strategy defines tail latencies.
- **Batch assembly:** Micro-batching and sequence packing touch many small objects, which stresses metadata services differently than streaming large shards.

## Critical I/O behaviors to measure
- **Listing & metadata fan-out:** Determines time-to-first-batch and controls the number of RPCs the filesystem or object store must service. Small directories vs. deep trees show different scaling characteristics.
- **Checkpoint throughput & replay:** Drives how quickly training recovers from failures and whether mid-epoch checkpoints starve compute. Sync vs. async IO engines show different tail behavior.
- **Data loading & prefetch depth:** Captures the end-to-end impact of sequential, random-access, memory-mapped, and prefetch-based read strategies on steady-state throughput.
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
- **Listing Emulated Benchmark (LEB)** — `listing_folder_benchmarks/`. Focuses on metadata latency via real `os.stat()` and `os.scandir()` calls, pagination cost, and enumeration concurrency. Ideal for sizing manifest caches, comparing filesystem mounts, or validating new object-store regions.
- **Serving Benchmarks** — `serving_benchmarks/`. Measures real file-I/O data-loading latency for batch inference workloads, with configurable read buffers, auto-generated datasets, and tail behavior tracking.
- **Checkpointing Benchmarks** — `checkpointing_benchmarks/`. Simulates shard-write/read cycles with tunable concurrency, fsync, retention policies, and configurable IO engines (sync or async via aiofiles).
- **Dataloader Benchmarks** *(new)* — `dataloader_benchmarks/`. Emulates training data-loading pipelines with four read strategies: sequential, random (shuffled), memory-mapped (mmap), and prefetch (thread pool). Supports gzip-compressed datasets and tracks TTFB, per-sample tail latencies, and epoch throughput.

Each module is designed to run on commodity hardware without GPUs, yet scales to GPU-backed clusters when you want to observe device utilization side-by-side.

## Getting started
1. Clone the repo and install in development mode (Python 3.10+):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. Choose a benchmark and run it:
   ```bash
   # Listing — real metadata I/O with os.stat()
   python -m listing_folder_benchmarks.src.run \
     --run-name leb-demo --entries-per-dir 500 --depth 2

   # Checkpointing — with async IO engine
   python -m checkpointing_benchmarks.src.run \
     --run-name cpb-demo --iterations 3 --io-engine async

   # Serving — real file reads
   python -m serving_benchmarks.src.train \
     --run-name serve-demo --steps 50 --auto-generate true

   # Dataloader — compare read strategies
   python -m dataloader_benchmarks.src.run \
     --run-name dl-demo --strategy prefetch --epochs 2
   ```
3. Results land under `./metrics/<run-name>/` as structured CSV + YAML.

## Running tests
```bash
pytest tests/ -v
```

## How this project helps you
- **Validate storage tiers before deployment:** Measure listing latency on S3 vs. FSx, or compare Azure Blob tiers with and without hierarchical namespaces.
- **Tune application knobs safely:** Experiment with PyTorch DataLoader workers, tf.data pipeline fusion, or checkpoint intervals and observe real latency/throughput changes.
- **Compare IO engines:** Benchmark sync vs. async checkpoint write/read to quantify the impact of aiofiles on your storage path.
- **Evaluate read strategies:** Sequential vs. random vs. mmap vs. prefetch — find the right pattern for your dataset shape and storage tier.
- **Quantify ROI of provider features:** Demonstrate the impact of AWS S3 Express, GCS metadata caching, or Azure Premium SSD upgrades on time-to-first-batch and $/token.
- **Share reproducible results:** Use the portable outputs and configuration files to publish, compare, or reproduce runs across teams and vendors.

## Roadmap & contributions
- Near-term additions: distributed multi-node benchmarks, cloud object-store backends (S3/GCS/AzBlob), and mixed workloads that combine listing + checkpoint + streaming phases.
- Contributions are welcome—open an issue with the workload you want to emulate or submit a PR with a new module/config.
- All code is released under the `LICENSE` file in this repo; please review before contributing.

## Support & feedback
Questions, tuning tips, or benchmark requests? File an issue or start a discussion. Real-world traces and anonymized metrics help us refine scenarios and defaults for other practitioners.
