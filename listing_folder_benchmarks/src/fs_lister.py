import os, time, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

def _chunks(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk: break
        yield chunk

def list_tree(root: str, page_size: int, concurrency: int):
    """
    Returns raw records for CSV:
    [start_ts, end_ts, entries_count, path]
    """
    records: List[List[str]] = []

    # walk once to get directories; emulate pagination by chunking entries
    dirs = []
    for p, subdirs, files in os.walk(root):
        entries = [os.path.join(p, f) for f in files]
        dirs.append((p, entries))

    def _list_chunk(path: str, entries: List[str]):
        start = time.time()
        # simulate "list page" by touching file metadata (stat)
        for _ in entries:  # os.stat(_) could be added if desired
            pass
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
