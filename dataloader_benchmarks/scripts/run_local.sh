#!/usr/bin/env bash
set -euo pipefail

# Quick local benchmark: sequential read with small dataset
python -m dataloader_benchmarks.src.run \
  --run-name dl-local-seq \
  --data-root ./data/dataloader \
  --strategy sequential \
  --epochs 2 \
  --sample-count 100 \
  --sample-size-kb 256 \
  --auto-generate true
