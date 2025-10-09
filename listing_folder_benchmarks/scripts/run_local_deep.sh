#!/usr/bin/env bash
set -euo pipefail
python -m src.listing_bench.run --config configs/deep_tree.yaml
