"""Unified CSV and YAML writers with atomic-write support."""

import csv
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore
    import json


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: str, rows: Sequence[Sequence[Any]], *, atomic: bool = True) -> None:
    """Write *rows* to a CSV file. Uses atomic rename by default."""
    _ensure_parent(path)
    if atomic:
        tmp = os.path.join(os.path.dirname(path) or ".", f".tmp-{uuid.uuid4()}.csv")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        os.replace(tmp, path)
    else:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)


def write_yaml(path: str, payload: Mapping[str, Any]) -> None:
    """Write *payload* to YAML (falls back to JSON if PyYAML is missing)."""
    _ensure_parent(path)
    data = dict(payload)
    with open(path, "w", encoding="utf-8") as fh:
        if yaml is not None:
            yaml.safe_dump(data, fh, sort_keys=True)
        else:
            json.dump(data, fh, indent=2, sort_keys=True)
