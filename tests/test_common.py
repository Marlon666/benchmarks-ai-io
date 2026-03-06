"""Unit tests for benchmarks_common utilities."""

import os
import math
import tempfile

from benchmarks_common.cli import parse_bool, load_yaml_config
from benchmarks_common.metadata import RunMetadata, build_metadata
from benchmarks_common.outputs import write_csv, write_yaml
from benchmarks_common.stats import percentile, throughput_mb_s, safe_mean, safe_median


class TestParseBool:
    def test_true_values(self):
        for v in [True, 1, "true", "True", "yes", "1", "on"]:
            assert parse_bool(v) is True, f"Expected True for {v!r}"

    def test_false_values(self):
        for v in [False, 0, "false", "no", "0", None, ""]:
            assert parse_bool(v) is False, f"Expected False for {v!r}"


class TestPercentile:
    def test_empty(self):
        assert percentile([], 0.5) == 0.0

    def test_single(self):
        assert percentile([42.0], 0.5) == 42.0

    def test_median(self):
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0

    def test_p95(self):
        values = list(range(100))
        assert percentile([float(x) for x in values], 0.95) == 94.0

    def test_extremes(self):
        vals = [1.0, 10.0]
        assert percentile(vals, 0.0) == 1.0
        assert percentile(vals, 1.0) == 10.0


class TestThroughput:
    def test_basic(self):
        # 1 MiB in 1 second = 1 MiB/s
        assert throughput_mb_s(1024 * 1024, 1.0) == 1.0

    def test_zero_duration(self):
        assert throughput_mb_s(1024, 0.0) == math.inf

    def test_negative_duration(self):
        assert throughput_mb_s(1024, -1.0) == math.inf


class TestSafeMeanMedian:
    def test_empty(self):
        assert safe_mean([]) == 0.0
        assert safe_median([]) == 0.0

    def test_values(self):
        assert safe_mean([2.0, 4.0]) == 3.0
        assert safe_median([1.0, 2.0, 3.0]) == 2.0


class TestBuildMetadata:
    def test_structure(self):
        meta = build_metadata("test-run", "checkpointing",
                              {"key": "val"}, {"throughput": 100})
        assert meta["run_name"] == "test-run"
        assert meta["benchmark"] == "checkpointing"
        assert "host" in meta
        assert meta["host"]["hostname"]
        assert meta["parameters"]["key"] == "val"
        assert meta["summary"]["throughput"] == 100
        assert meta["timestamp_utc"].endswith("Z")


class TestRunMetadata:
    def test_to_dict(self):
        m = RunMetadata(run_name="r", benchmark="b")
        d = m.to_dict()
        assert d["run_name"] == "r"
        assert d["benchmark"] == "b"


class TestOutputs:
    def test_write_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sub", "test.csv")
            write_csv(path, [["a", "b"], [1, 2]])
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "a,b" in content
            assert "1,2" in content

    def test_write_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.yaml")
            write_yaml(path, {"key": "value", "num": 42})
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "key" in content

    def test_non_atomic_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.csv")
            write_csv(path, [["x"]], atomic=False)
            assert os.path.exists(path)
