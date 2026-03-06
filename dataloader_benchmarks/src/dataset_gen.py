"""Generate synthetic datasets for the data-loader benchmark.

Creates binary files of configurable size with optional gzip compression
to simulate realistic AI training data samples.
"""

import gzip
import os
import pathlib
import random
import argparse


def generate_dataset(
    root: str,
    count: int,
    size_kb: int,
    compress: bool = False,
    seed: int = 42,
) -> str:
    """Create *count* sample files under *root*.

    Returns the absolute path to *root*.
    """
    random.seed(seed)
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    ext = ".bin.gz" if compress else ".bin"

    # Pre-allocate a chunk for efficient writing
    chunk_size = min(size_kb * 1024, 1024 * 1024)
    chunk = os.urandom(chunk_size)

    for i in range(count):
        path = os.path.join(root, f"sample_{i:06d}{ext}")
        remaining = size_kb * 1024
        opener = gzip.open if compress else open
        with opener(path, "wb") as f:
            while remaining > 0:
                to_write = chunk if remaining >= len(chunk) else chunk[:remaining]
                f.write(to_write)
                remaining -= len(to_write)

    return os.path.abspath(root)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for data-loader benchmark")
    parser.add_argument("--out", type=str, default="./data/dataloader")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--size-kb", type=int, default=512)
    parser.add_argument("--compress", action="store_true",
                        help="Gzip-compress each sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = generate_dataset(args.out, args.count, args.size_kb,
                            args.compress, args.seed)
    comp = " (gzip)" if args.compress else ""
    print(f"Generated {args.count} × {args.size_kb} KiB{comp} samples in {path}")


if __name__ == "__main__":
    main()
