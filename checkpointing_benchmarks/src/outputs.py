import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    yaml = None  # type: ignore
    import json


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: str, rows: Iterable[Sequence[Any]]) -> None:
    _ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(list(row))


def write_yaml(path: str, payload: Mapping[str, Any]) -> None:
    _ensure_parent(path)
    data = dict(payload)
    with open(path, "w", encoding="utf-8") as fh:
        if yaml is not None:
            yaml.safe_dump(data, fh, sort_keys=True)  # type: ignore[arg-type]
        else:
            json.dump(data, fh, indent=2, sort_keys=True)
