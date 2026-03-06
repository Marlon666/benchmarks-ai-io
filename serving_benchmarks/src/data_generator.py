"""Generate synthetic binary tensor files for serving benchmark data loading."""

import os
import random
import pathlib
import argparse


def generate_dataset(root: str, count: int, size_kb: int,
                     seed: int = 42) -> str:
    """Create *count* binary files of *size_kb* KiB under *root*.

    Files are named ``sample_XXXX.bin`` and contain random bytes to
    prevent filesystem dedup or compression from skewing results.
    """
    random.seed(seed)
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    chunk = os.urandom(min(size_kb * 1024, 1024 * 1024))  # 1 MiB max chunk

    for i in range(count):
        path = os.path.join(root, f"sample_{i:06d}.bin")
        remaining = size_kb * 1024
        with open(path, "wb") as f:
            while remaining > 0:
                to_write = chunk if remaining >= len(chunk) else chunk[:remaining]
                f.write(to_write)
                remaining -= len(to_write)

    return os.path.abspath(root)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic tensor files for serving benchmarks")
    parser.add_argument("--out", type=str, default="./data/serving",
                        help="Output directory for generated files")
    parser.add_argument("--count", type=int, default=1000,
                        help="Number of sample files to generate")
    parser.add_argument("--size-kb", type=int, default=256,
                        help="Size of each sample file in KiB")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = generate_dataset(args.out, args.count, args.size_kb, args.seed)
    print(f"Generated {args.count} files ({args.size_kb} KiB each) in {path}")


if __name__ == "__main__":
    main()
