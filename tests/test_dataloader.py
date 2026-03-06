"""Unit tests for the dataloader benchmark."""

import os
import tempfile

from dataloader_benchmarks.src.dataset_gen import generate_dataset
from dataloader_benchmarks.src.loader import (
    LoaderParams, discover_samples, run_loader,
)


class TestDatasetGen:
    def test_creates_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "data")
            generate_dataset(root, count=10, size_kb=1)
            files = os.listdir(root)
            assert len(files) == 10
            assert all(f.endswith(".bin") for f in files)

    def test_file_size(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "data")
            generate_dataset(root, count=1, size_kb=4)
            path = os.path.join(root, "sample_000000.bin")
            assert os.path.getsize(path) == 4 * 1024

    def test_compressed(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "data")
            generate_dataset(root, count=5, size_kb=1, compress=True)
            files = os.listdir(root)
            assert all(f.endswith(".bin.gz") for f in files)


class TestDiscoverSamples:
    def test_finds_bin_files(self):
        with tempfile.TemporaryDirectory() as td:
            generate_dataset(td, count=3, size_kb=1)
            samples = discover_samples(td)
            assert len(samples) == 3

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            assert discover_samples(td) == []


class TestRunLoader:
    def _make_data(self, td, count=20, size_kb=1):
        root = os.path.join(td, "data")
        generate_dataset(root, count=count, size_kb=size_kb)
        return root

    def test_sequential(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_data(td)
            params = LoaderParams(data_root=root, strategy="sequential", epochs=1)
            samples, epochs = run_loader(params)
            assert len(samples) == 20
            assert len(epochs) == 1
            assert epochs[0].strategy == "sequential"
            assert epochs[0].ttfb_sec > 0

    def test_random(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_data(td)
            params = LoaderParams(data_root=root, strategy="random", epochs=1)
            samples, epochs = run_loader(params)
            assert len(samples) == 20

    def test_mmap(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_data(td)
            params = LoaderParams(data_root=root, strategy="mmap", epochs=1)
            samples, epochs = run_loader(params)
            assert len(samples) == 20

    def test_prefetch(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_data(td)
            params = LoaderParams(data_root=root, strategy="prefetch",
                                  epochs=1, prefetch_depth=2)
            samples, epochs = run_loader(params)
            assert len(samples) == 20
            assert epochs[0].strategy == "prefetch"

    def test_multiple_epochs(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_data(td, count=5)
            params = LoaderParams(data_root=root, strategy="sequential", epochs=3)
            samples, epochs = run_loader(params)
            assert len(epochs) == 3
            assert len(samples) == 15  # 5 samples × 3 epochs

    def test_invalid_strategy(self):
        import pytest
        with tempfile.TemporaryDirectory() as td:
            root = self._make_data(td, count=1)
            params = LoaderParams(data_root=root, strategy="bogus", epochs=1)
            with pytest.raises(ValueError, match="Unknown strategy"):
                run_loader(params)
