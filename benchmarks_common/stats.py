"""Shared statistical helpers for all benchmark modules."""

import math
import statistics
from typing import List


def percentile(values: List[float], pct: float) -> float:
    """Return the *pct*-th percentile (0–1 scale) from *values*."""
    if not values:
        return 0.0
    values = sorted(values)
    if pct <= 0:
        return values[0]
    if pct >= 1:
        return values[-1]
    idx = int(round(pct * (len(values) - 1)))
    return values[idx]


def throughput_mb_s(byte_count: int, duration_sec: float) -> float:
    """Compute throughput in MiB/s. Returns inf if *duration_sec* is zero."""
    if duration_sec <= 0:
        return math.inf
    return (byte_count / 1024 / 1024) / duration_sec


def safe_mean(values: List[float]) -> float:
    """Mean that returns 0.0 for empty lists instead of raising."""
    return statistics.mean(values) if values else 0.0


def safe_median(values: List[float]) -> float:
    """Median that returns 0.0 for empty lists instead of raising."""
    return statistics.median(values) if values else 0.0
