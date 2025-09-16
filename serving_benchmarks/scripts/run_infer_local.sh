#!/usr/bin/env bash
set -euo pipefail
python -m src.io_bench.training --run-name infer-small --steps 200 --gbs 256 --mbs 1 --num-workers 4 --mode infer