import os, pathlib, random, string
from typing import Optional

def make_tree(root: str, entries_per_dir: int, depth: int, seed: Optional[int]=42):
    random.seed(seed)
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)

    def _mk(level: int, base: str):
        # files
        for i in range(entries_per_dir):
            fname = "".join(random.choices(string.ascii_lowercase, k=12))
            (pathlib.Path(base) / f"{fname}.dat").write_bytes(b"")
        # recurse
        if level <= 1:
            return
        for d in range(max(1, entries_per_dir // 20)):  # ~5% become subdirs
            dname = "".join(random.choices(string.ascii_lowercase, k=8))
            newdir = pathlib.Path(base) / dname
            newdir.mkdir(exist_ok=True)
            _mk(level-1, str(newdir))

    _mk(depth, root)
    return os.path.abspath(root)
