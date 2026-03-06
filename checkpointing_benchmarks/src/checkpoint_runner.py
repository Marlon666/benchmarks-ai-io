import asyncio
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from benchmarks_common.stats import throughput_mb_s


@dataclass
class BenchmarkParams:
    run_name: str
    root: str
    iterations: int
    shard_count: int
    shard_size_mb: float
    concurrency: int
    fsync: bool
    mode: str  # write | read | write-read
    retention: int
    chunk_mb: float
    read_buffer_kb: int
    cleanup_after: bool
    io_engine: str = "sync"  # sync | async


@dataclass
class ShardRecord:
    iteration: int
    phase: str
    shard_id: int
    bytes: int
    duration_sec: float
    throughput_mb_s: float
    path: str


@dataclass
class IterationRecord:
    iteration: int
    phase: str
    duration_sec: float
    total_bytes: int
    throughput_mb_s: float


class CheckpointingBenchmark:
    def __init__(self, params: BenchmarkParams):
        self.params = params
        self.root = Path(params.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._validate_mode()

    def _validate_mode(self) -> None:
        if self.params.mode not in {"write", "read", "write-read"}:
            raise ValueError(
                f"Unsupported mode '{self.params.mode}'. "
                "Expected one of: write, read, write-read.")

    def _existing_checkpoints(self) -> List[Path]:
        if not self.root.exists():
            return []
        return sorted(p for p in self.root.iterdir() if p.is_dir())

    def _maybe_cleanup(self, directories: Iterable[Path]) -> None:
        if not self.params.cleanup_after:
            return
        for d in directories:
            shutil.rmtree(d, ignore_errors=True)

    def _retention_trim(self, history: List[Path]) -> None:
        if self.params.retention <= 0:
            return
        while len(history) > self.params.retention:
            doomed = history.pop(0)
            shutil.rmtree(doomed, ignore_errors=True)

    def run(self) -> Tuple[List[ShardRecord], List[IterationRecord]]:
        if self.params.io_engine == "async":
            return asyncio.run(self._run_async())
        return self._run_sync()

    def _run_sync(self) -> Tuple[List[ShardRecord], List[IterationRecord]]:
        shard_records: List[ShardRecord] = []
        iteration_records: List[IterationRecord] = []

        do_write = self.params.mode in {"write", "write-read"}
        do_read = self.params.mode in {"read", "write-read"}

        existing = self._existing_checkpoints()
        read_cycle_dirs: List[Path]

        if do_read and not do_write:
            if len(existing) < self.params.iterations:
                raise ValueError(
                    f"Read-only mode requires at least {self.params.iterations} "
                    f"checkpoint directories under {self.root}. Found {len(existing)}."
                )
            read_cycle_dirs = existing[:self.params.iterations]
        else:
            read_cycle_dirs = []

        created_dirs: List[Path] = []

        for iteration in range(1, self.params.iterations + 1):
            checkpoint_dir: Path
            if do_write:
                checkpoint_dir = self.root / f"{self.params.run_name}_ckpt_{iteration:04d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.append(checkpoint_dir)
                write_records, iter_record = self._run_write_phase(
                    checkpoint_dir, iteration)
                shard_records.extend(write_records)
                iteration_records.append(iter_record)
                self._retention_trim(created_dirs)
            else:
                checkpoint_dir = read_cycle_dirs[iteration - 1]

            if do_read:
                read_target = checkpoint_dir
                read_records, iter_record = self._run_read_phase(
                    read_target, iteration)
                shard_records.extend(read_records)
                iteration_records.append(iter_record)

        self._maybe_cleanup(created_dirs)
        return shard_records, iteration_records

    async def _run_async(self) -> Tuple[List[ShardRecord], List[IterationRecord]]:
        """Async IO engine using aiofiles for write/read phases."""
        try:
            import aiofiles  # type: ignore
        except ImportError:
            raise RuntimeError(
                "aiofiles is required for --io-engine=async. "
                "Install with: pip install aiofiles"
            )

        shard_records: List[ShardRecord] = []
        iteration_records: List[IterationRecord] = []

        do_write = self.params.mode in {"write", "write-read"}
        do_read = self.params.mode in {"read", "write-read"}

        existing = self._existing_checkpoints()
        if do_read and not do_write:
            if len(existing) < self.params.iterations:
                raise ValueError(
                    f"Read-only mode requires at least {self.params.iterations} "
                    f"checkpoint directories under {self.root}. Found {len(existing)}."
                )
            read_cycle_dirs = existing[:self.params.iterations]
        else:
            read_cycle_dirs = []

        created_dirs: List[Path] = []

        for iteration in range(1, self.params.iterations + 1):
            if do_write:
                checkpoint_dir = self.root / f"{self.params.run_name}_ckpt_{iteration:04d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.append(checkpoint_dir)
                write_records, iter_record = await self._run_async_write_phase(
                    checkpoint_dir, iteration)
                shard_records.extend(write_records)
                iteration_records.append(iter_record)
                self._retention_trim(created_dirs)
            else:
                checkpoint_dir = read_cycle_dirs[iteration - 1]

            if do_read:
                read_records, iter_record = await self._run_async_read_phase(
                    checkpoint_dir, iteration)
                shard_records.extend(read_records)
                iteration_records.append(iter_record)

        self._maybe_cleanup(created_dirs)
        return shard_records, iteration_records

    # --- sync helpers -------------------------------------------------

    def _shard_paths(self, checkpoint_dir: Path) -> List[Path]:
        return [
            checkpoint_dir / f"shard_{idx:05d}.ckpt"
            for idx in range(self.params.shard_count)
        ]

    def _run_write_phase(self, checkpoint_dir: Path,
                         iteration: int) -> Tuple[List[ShardRecord], IterationRecord]:
        shard_paths = self._shard_paths(checkpoint_dir)
        shard_bytes = int(self.params.shard_size_mb * 1024 * 1024)
        chunk_bytes = max(1, int(self.params.chunk_mb * 1024 * 1024))
        shard_records: List[ShardRecord] = []

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.params.concurrency) as pool:
            futures = []
            for shard_id, path in enumerate(shard_paths):
                futures.append(
                    pool.submit(
                        _write_shard,
                        path,
                        shard_bytes,
                        chunk_bytes,
                        self.params.fsync,
                    ))
            for shard_id, future in enumerate(futures):
                duration = float(future.result())
                tp = throughput_mb_s(shard_bytes, duration)
                shard_records.append(
                    ShardRecord(
                        iteration=iteration,
                        phase="write",
                        shard_id=shard_id,
                        bytes=shard_bytes,
                        duration_sec=duration,
                        throughput_mb_s=tp,
                        path=str(shard_paths[shard_id]),
                    ))
        end = time.perf_counter()
        total_duration = max(end - start, 1e-6)
        total_bytes = shard_bytes * self.params.shard_count
        iter_record = IterationRecord(
            iteration=iteration,
            phase="write",
            duration_sec=total_duration,
            total_bytes=total_bytes,
            throughput_mb_s=throughput_mb_s(total_bytes, total_duration),
        )
        return shard_records, iter_record

    def _run_read_phase(self, checkpoint_dir: Path,
                        iteration: int) -> Tuple[List[ShardRecord], IterationRecord]:
        shard_paths = self._resolve_read_paths(checkpoint_dir)
        buffer_bytes = max(1, int(self.params.read_buffer_kb * 1024))
        shard_records: List[ShardRecord] = []

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.params.concurrency) as pool:
            futures = []
            for shard_id, path in enumerate(shard_paths):
                futures.append(pool.submit(_read_shard, path, buffer_bytes))
            for shard_id, future in enumerate(futures):
                duration = float(future.result())
                bytes_read = os.path.getsize(shard_paths[shard_id])
                tp = throughput_mb_s(bytes_read, duration)
                shard_records.append(
                    ShardRecord(
                        iteration=iteration,
                        phase="read",
                        shard_id=shard_id,
                        bytes=bytes_read,
                        duration_sec=duration,
                        throughput_mb_s=tp,
                        path=str(shard_paths[shard_id]),
                    ))
        end = time.perf_counter()
        total_duration = max(end - start, 1e-6)
        total_bytes = sum(os.path.getsize(p) for p in shard_paths)
        iter_record = IterationRecord(
            iteration=iteration,
            phase="read",
            duration_sec=total_duration,
            total_bytes=total_bytes,
            throughput_mb_s=throughput_mb_s(total_bytes, total_duration),
        )
        return shard_records, iter_record

    def _resolve_read_paths(self, checkpoint_dir: Path) -> List[Path]:
        candidates = sorted(checkpoint_dir.glob("*.ckpt"))
        if candidates:
            return candidates
        return self._shard_paths(checkpoint_dir)

    # --- async helpers ------------------------------------------------

    async def _run_async_write_phase(
        self, checkpoint_dir: Path, iteration: int,
    ) -> Tuple[List[ShardRecord], IterationRecord]:
        import aiofiles  # type: ignore

        shard_paths = self._shard_paths(checkpoint_dir)
        shard_bytes = int(self.params.shard_size_mb * 1024 * 1024)
        chunk_bytes = max(1, int(self.params.chunk_mb * 1024 * 1024))

        sem = asyncio.Semaphore(self.params.concurrency)

        async def _write_one(shard_id: int, path: Path) -> ShardRecord:
            async with sem:
                path.parent.mkdir(parents=True, exist_ok=True)
                chunk = os.urandom(min(shard_bytes, chunk_bytes))
                remaining = shard_bytes
                start = time.perf_counter()
                async with aiofiles.open(path, "wb") as fh:
                    while remaining > 0:
                        to_write = chunk if remaining >= chunk_bytes else chunk[:remaining]
                        await fh.write(to_write)
                        remaining -= len(to_write)
                    if self.params.fsync:
                        await fh.flush()
                        os.fsync(fh.fileno())
                end = time.perf_counter()
                duration = max(end - start, 1e-9)
                return ShardRecord(
                    iteration=iteration, phase="write", shard_id=shard_id,
                    bytes=shard_bytes, duration_sec=duration,
                    throughput_mb_s=throughput_mb_s(shard_bytes, duration),
                    path=str(path),
                )

        start = time.perf_counter()
        results = await asyncio.gather(
            *[_write_one(i, p) for i, p in enumerate(shard_paths)]
        )
        end = time.perf_counter()
        total_bytes = shard_bytes * self.params.shard_count
        total_duration = max(end - start, 1e-6)
        return list(results), IterationRecord(
            iteration=iteration, phase="write",
            duration_sec=total_duration, total_bytes=total_bytes,
            throughput_mb_s=throughput_mb_s(total_bytes, total_duration),
        )

    async def _run_async_read_phase(
        self, checkpoint_dir: Path, iteration: int,
    ) -> Tuple[List[ShardRecord], IterationRecord]:
        import aiofiles  # type: ignore

        shard_paths = self._resolve_read_paths(checkpoint_dir)
        buffer_bytes = max(1, int(self.params.read_buffer_kb * 1024))

        sem = asyncio.Semaphore(self.params.concurrency)

        async def _read_one(shard_id: int, path: Path) -> ShardRecord:
            async with sem:
                start = time.perf_counter()
                async with aiofiles.open(path, "rb") as fh:
                    while await fh.read(buffer_bytes):
                        pass
                end = time.perf_counter()
                duration = max(end - start, 1e-9)
                sz = os.path.getsize(path)
                return ShardRecord(
                    iteration=iteration, phase="read", shard_id=shard_id,
                    bytes=sz, duration_sec=duration,
                    throughput_mb_s=throughput_mb_s(sz, duration),
                    path=str(path),
                )

        start = time.perf_counter()
        results = await asyncio.gather(
            *[_read_one(i, p) for i, p in enumerate(shard_paths)]
        )
        end = time.perf_counter()
        total_bytes = sum(os.path.getsize(p) for p in shard_paths)
        total_duration = max(end - start, 1e-6)
        return list(results), IterationRecord(
            iteration=iteration, phase="read",
            duration_sec=total_duration, total_bytes=total_bytes,
            throughput_mb_s=throughput_mb_s(total_bytes, total_duration),
        )


# --- module-level sync helpers ----------------------------------------

def _write_shard(path: Path, size_bytes: int, chunk_bytes: int,
                 fsync: bool) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_bytes = min(size_bytes, chunk_bytes)
    block = os.urandom(chunk_bytes)
    remaining = size_bytes

    start = time.perf_counter()
    with open(path, "wb") as fh:
        while remaining > 0:
            to_write = block if remaining >= chunk_bytes else block[:remaining]
            fh.write(to_write)
            remaining -= len(to_write)
        if fsync:
            fh.flush()
            os.fsync(fh.fileno())
    end = time.perf_counter()
    return max(end - start, 1e-9)


def _read_shard(path: Path, buffer_bytes: int) -> float:
    start = time.perf_counter()
    with open(path, "rb") as fh:
        while fh.read(buffer_bytes):
            continue
    end = time.perf_counter()
    return max(end - start, 1e-9)
