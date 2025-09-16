import os, csv, uuid, pathlib, yaml
from typing import Sequence, Mapping, Any

def _ensure_parent(path: str):
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

def write_csv_local(path: str, rows: Sequence[Sequence[Any]]):
    _ensure_parent(path)
    tmp = f"{uuid.uuid4()}.csv"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    os.replace(tmp, path)

def write_yaml_local(path: str, data: Mapping[str, Any]):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)