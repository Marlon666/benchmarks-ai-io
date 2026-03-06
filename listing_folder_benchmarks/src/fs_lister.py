"""Filesystem listing benchmarks — real I/O with os.stat and os.scandir support."""

import os
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


def _chunks(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def list_tree(root: str, page_size: int, concurrency: int,
              use_stat: bool = False, use_scandir: bool = False):
    """
    Enumerate a directory tree and return raw records for CSV:
    [start_ts, end_ts, entries_count, path]

    Parameters
    ----------
    root : str
        Root directory to enumerate.
    page_size : int
        Number of entries per listing "page" (chunked).
    concurrency : int
        Thread concurrency for parallel directory listing.
    use_stat : bool
        If True, call os.stat() on every entry to probe real metadata latency.
    use_scandir : bool
        If True, use os.scandir() instead of os.walk() for the initial
        enumeration (faster on most systems, avoids double-stat).
    """
    records: List[List[str]] = []

    if use_scandir:
        dirs = _scandir_walk(root)
    else:
        dirs = []
        for p, subdirs, files in os.walk(root):
            entries = [os.path.join(p, f) for f in files]
            dirs.append((p, entries))

    def _list_chunk(path: str, entries: List[str]):
        start = time.time()
        if use_stat:
            # Real metadata probing — exercises the filesystem stat path
            for entry in entries:
                try:
                    os.stat(entry)
                except OSError:
                    pass
        else:
            # Lightweight enumeration — still validates path existence
            for entry in entries:
                os.path.exists(entry)
        end = time.time()
        return [f"{start:.6f}", f"{end:.6f}", str(len(entries)), path]

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = []
        for path, entries in dirs:
            for chunk in _chunks(entries, max(1, page_size)):
                futures.append(ex.submit(_list_chunk, path, chunk))
        for fut in as_completed(futures):
            records.append(fut.result())

    return sorted(records, key=lambda r: float(r[0]))


def _scandir_walk(root: str) -> List[tuple]:
    """Walk a tree using os.scandir() for faster enumeration with DirEntry."""
    results = []
    stack = [root]
    while stack:
        current = stack.pop()
        entries = []
        subdirs = []
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        entries.append(entry.path)
                    elif entry.is_dir(follow_symlinks=False):
                        subdirs.append(entry.path)
        except PermissionError:
            continue
        results.append((current, entries))
        stack.extend(subdirs)
    return results
