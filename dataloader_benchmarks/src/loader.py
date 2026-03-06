"""Core data-loading benchmark engine.

Supports four read strategies that model real training data-pipeline patterns:
- sequential: Read files in order (baseline, sequential scan)
- random: Shuffle files and read in random order (worst-case for HDDs / object stores)
- mmap: Memory-mapped reads via mmap (zero-copy, OS page cache)
- prefetch: Concurrent prefetching via ThreadPoolExecutor

Each strategy reports per-sample and per-epoch aggregate metrics.
"""

import gzip
import mmap
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from benchmarks_common.stats import throughput_mb_s


@dataclass
class SampleRecord:
    epoch: int
    sample_idx: int
    path: str
    bytes_read: int
    duration_sec: float
    throughput_mb_s: float


@dataclass
class EpochRecord:
    epoch: int
    strategy: str
    samples: int
    total_bytes: int
    duration_sec: float
    throughput_mb_s: float
    ttfb_sec: float  # time to first batch


@dataclass
class LoaderParams:
    data_root: str
    strategy: str = "sequential"      # sequential | random | mmap | prefetch
    epochs: int = 3
    prefetch_depth: int = 4
    read_buffer_kb: int = 256
    batch_size: int = 32
    shuffle_seed: Optional[int] = 42
    compressed: bool = False


def discover_samples(root: str, compressed: bool = False) -> List[str]:
    """Find all sample files under *root*."""
    ext = "*.bin.gz" if compressed else "*.bin"
    return sorted(str(p) for p in Path(root).glob(ext))


def run_loader(params: LoaderParams) -> tuple:
    """Run the data-loading benchmark and return (sample_records, epoch_records)."""
    samples = discover_samples(params.data_root, params.compressed)
    if not samples:
        raise FileNotFoundError(
            f"No sample files in {params.data_root}. "
            "Run dataset_gen.py first."
        )

    strategy_fn = {
        "sequential": _load_sequential,
        "random": _load_random,
        "mmap": _load_mmap,
        "prefetch": _load_prefetch,
    }.get(params.strategy)

    if strategy_fn is None:
        raise ValueError(
            f"Unknown strategy '{params.strategy}'. "
            "Choose from: sequential, random, mmap, prefetch"
        )

    all_sample_records: List[SampleRecord] = []
    all_epoch_records: List[EpochRecord] = []

    for epoch in range(1, params.epochs + 1):
        sample_records, epoch_record = strategy_fn(samples, epoch, params)
        all_sample_records.extend(sample_records)
        all_epoch_records.append(epoch_record)

    return all_sample_records, all_epoch_records


# --- strategies ---------------------------------------------------------

def _read_file(path: str, buffer_kb: int, compressed: bool) -> tuple:
    """Read a file, return (bytes_read, duration_sec)."""
    buf_size = buffer_kb * 1024
    opener = gzip.open if compressed else open
    start = time.perf_counter()
    total = 0
    with opener(path, "rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            total += len(chunk)
    end = time.perf_counter()
    return total, max(end - start, 1e-9)


def _load_sequential(
    samples: List[str], epoch: int, params: LoaderParams,
) -> tuple:
    """Read files in order — baseline sequential scan."""
    records: List[SampleRecord] = []
    epoch_start = time.perf_counter()
    ttfb = 0.0
    total_bytes = 0

    for idx, path in enumerate(samples):
        nbytes, dur = _read_file(path, params.read_buffer_kb, params.compressed)
        if idx == 0:
            ttfb = time.perf_counter() - epoch_start
        total_bytes += nbytes
        records.append(SampleRecord(
            epoch=epoch, sample_idx=idx, path=path,
            bytes_read=nbytes, duration_sec=dur,
            throughput_mb_s=throughput_mb_s(nbytes, dur),
        ))

    epoch_dur = max(time.perf_counter() - epoch_start, 1e-9)
    epoch_rec = EpochRecord(
        epoch=epoch, strategy="sequential", samples=len(samples),
        total_bytes=total_bytes, duration_sec=epoch_dur,
        throughput_mb_s=throughput_mb_s(total_bytes, epoch_dur),
        ttfb_sec=ttfb,
    )
    return records, epoch_rec


def _load_random(
    samples: List[str], epoch: int, params: LoaderParams,
) -> tuple:
    """Shuffle files and read in random order — worst-case for HDDs."""
    order = list(samples)
    if params.shuffle_seed is not None:
        rng = random.Random(params.shuffle_seed + epoch)
        rng.shuffle(order)
    else:
        random.shuffle(order)

    records: List[SampleRecord] = []
    epoch_start = time.perf_counter()
    ttfb = 0.0
    total_bytes = 0

    for idx, path in enumerate(order):
        nbytes, dur = _read_file(path, params.read_buffer_kb, params.compressed)
        if idx == 0:
            ttfb = time.perf_counter() - epoch_start
        total_bytes += nbytes
        records.append(SampleRecord(
            epoch=epoch, sample_idx=idx, path=path,
            bytes_read=nbytes, duration_sec=dur,
            throughput_mb_s=throughput_mb_s(nbytes, dur),
        ))

    epoch_dur = max(time.perf_counter() - epoch_start, 1e-9)
    epoch_rec = EpochRecord(
        epoch=epoch, strategy="random", samples=len(samples),
        total_bytes=total_bytes, duration_sec=epoch_dur,
        throughput_mb_s=throughput_mb_s(total_bytes, epoch_dur),
        ttfb_sec=ttfb,
    )
    return records, epoch_rec


def _load_mmap(
    samples: List[str], epoch: int, params: LoaderParams,
) -> tuple:
    """Memory-mapped reads — zero-copy via OS page cache."""
    records: List[SampleRecord] = []
    epoch_start = time.perf_counter()
    ttfb = 0.0
    total_bytes = 0

    for idx, path in enumerate(samples):
        start = time.perf_counter()
        sz = os.path.getsize(path)
        if sz == 0:
            dur = max(time.perf_counter() - start, 1e-9)
        else:
            with open(path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Touch every page to force actual reads
                    _ = mm[:]
            dur = max(time.perf_counter() - start, 1e-9)

        if idx == 0:
            ttfb = time.perf_counter() - epoch_start
        total_bytes += sz
        records.append(SampleRecord(
            epoch=epoch, sample_idx=idx, path=path,
            bytes_read=sz, duration_sec=dur,
            throughput_mb_s=throughput_mb_s(sz, dur),
        ))

    epoch_dur = max(time.perf_counter() - epoch_start, 1e-9)
    epoch_rec = EpochRecord(
        epoch=epoch, strategy="mmap", samples=len(samples),
        total_bytes=total_bytes, duration_sec=epoch_dur,
        throughput_mb_s=throughput_mb_s(total_bytes, epoch_dur),
        ttfb_sec=ttfb,
    )
    return records, epoch_rec


def _load_prefetch(
    samples: List[str], epoch: int, params: LoaderParams,
) -> tuple:
    """Concurrent prefetching via thread pool — models DataLoader workers."""
    records: List[SampleRecord] = []
    epoch_start = time.perf_counter()
    ttfb = 0.0
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=params.prefetch_depth) as pool:
        futures = {}
        for idx, path in enumerate(samples):
            fut = pool.submit(_read_file, path, params.read_buffer_kb,
                              params.compressed)
            futures[fut] = (idx, path)

        for i, fut in enumerate(as_completed(futures)):
            idx, path = futures[fut]
            nbytes, dur = fut.result()
            if i == 0:
                ttfb = time.perf_counter() - epoch_start
            total_bytes += nbytes
            records.append(SampleRecord(
                epoch=epoch, sample_idx=idx, path=path,
                bytes_read=nbytes, duration_sec=dur,
                throughput_mb_s=throughput_mb_s(nbytes, dur),
            ))

    epoch_dur = max(time.perf_counter() - epoch_start, 1e-9)
    epoch_rec = EpochRecord(
        epoch=epoch, strategy="prefetch", samples=len(samples),
        total_bytes=total_bytes, duration_sec=epoch_dur,
        throughput_mb_s=throughput_mb_s(total_bytes, epoch_dur),
        ttfb_sec=ttfb,
    )
    return records, epoch_rec
