"""Unit tests for the checkpointing benchmark."""

import os
import tempfile

from checkpointing_benchmarks.src.checkpoint_runner import (
    BenchmarkParams, CheckpointingBenchmark,
)


def _default_params(**overrides) -> BenchmarkParams:
    defaults = dict(
        run_name="test-ckpt",
        root="",
        iterations=2,
        shard_count=2,
        shard_size_mb=0.1,  # 100 KiB — small for fast tests
        concurrency=2,
        fsync=False,
        mode="write-read",
        retention=0,
        chunk_mb=0.1,
        read_buffer_kb=64,
        cleanup_after=True,
        io_engine="sync",
    )
    defaults.update(overrides)
    return BenchmarkParams(**defaults)


class TestCheckpointingSync:
    def test_write_read(self):
        with tempfile.TemporaryDirectory() as td:
            params = _default_params(root=td)
            bench = CheckpointingBenchmark(params)
            shards, iters = bench.run()
            # 2 iterations × 2 shards × 2 phases = 8 shard records
            assert len(shards) == 8
            # 2 iterations × 2 phases = 4 iteration records
            assert len(iters) == 4
            # All throughputs should be positive
            for s in shards:
                assert s.throughput_mb_s > 0

    def test_write_only(self):
        with tempfile.TemporaryDirectory() as td:
            params = _default_params(root=td, mode="write")
            bench = CheckpointingBenchmark(params)
            shards, iters = bench.run()
            assert all(s.phase == "write" for s in shards)
            assert all(i.phase == "write" for i in iters)

    def test_retention(self):
        with tempfile.TemporaryDirectory() as td:
            params = _default_params(root=td, mode="write",
                                     iterations=5, retention=2,
                                     cleanup_after=False)
            bench = CheckpointingBenchmark(params)
            bench.run()
            remaining = [d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
            assert len(remaining) <= 2

    def test_invalid_mode(self):
        import pytest
        with pytest.raises(ValueError, match="Unsupported mode"):
            _default_params(mode="invalid")
            # Params validation happens in __init__
            with tempfile.TemporaryDirectory() as td:
                params = _default_params(root=td, mode="invalid")
                CheckpointingBenchmark(params)


class TestCheckpointingAsync:
    def test_async_write_read(self):
        with tempfile.TemporaryDirectory() as td:
            params = _default_params(root=td, io_engine="async")
            bench = CheckpointingBenchmark(params)
            shards, iters = bench.run()
            assert len(shards) == 8
            assert len(iters) == 4
            for s in shards:
                assert s.throughput_mb_s > 0
