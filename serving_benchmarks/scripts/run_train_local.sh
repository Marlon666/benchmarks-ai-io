#!/usr/bin/env bash
set -euo pipefail
python -m src.io_bench.training --run-name train-small --steps 100 --gbs 128 --mbs 8 --num-workers 4 --mode train