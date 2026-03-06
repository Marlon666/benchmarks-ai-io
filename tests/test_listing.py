"""Unit tests for the listing benchmark."""

import os
import tempfile

from listing_folder_benchmarks.src.synthetic_tree import make_tree
from listing_folder_benchmarks.src.fs_lister import list_tree, _scandir_walk


class TestSyntheticTree:
    def test_creates_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "tree")
            make_tree(root, entries_per_dir=10, depth=2)
            # Root should have files
            files = [f for f in os.listdir(root)
                     if os.path.isfile(os.path.join(root, f))]
            assert len(files) == 10

    def test_creates_subdirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "tree")
            make_tree(root, entries_per_dir=20, depth=3)
            dirs = [d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))]
            assert len(dirs) >= 1  # 20 // 20 = 1 subdir


class TestListTree:
    def _make_small_tree(self, td):
        root = os.path.join(td, "tree")
        make_tree(root, entries_per_dir=20, depth=2)
        return root

    def test_basic_listing(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_small_tree(td)
            records = list_tree(root, page_size=10, concurrency=2)
            assert len(records) > 0
            for r in records:
                assert len(r) == 4
                assert float(r[1]) >= float(r[0])  # end >= start
                assert int(r[2]) > 0

    def test_stat_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_small_tree(td)
            records = list_tree(root, page_size=10, concurrency=2,
                                use_stat=True)
            assert len(records) > 0
            # Stat-mode should take measurable time
            for r in records:
                dur = float(r[1]) - float(r[0])
                assert dur >= 0

    def test_scandir_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_small_tree(td)
            records = list_tree(root, page_size=10, concurrency=2,
                                use_scandir=True)
            assert len(records) > 0


class TestScandirWalk:
    def test_finds_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "tree")
            make_tree(root, entries_per_dir=5, depth=2)
            results = _scandir_walk(root)
            total_files = sum(len(entries) for _, entries in results)
            assert total_files >= 5
