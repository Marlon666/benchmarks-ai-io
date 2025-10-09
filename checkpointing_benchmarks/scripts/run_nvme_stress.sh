#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODULE_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python3 -m checkpointing_benchmarks.src.run \
  --config "${MODULE_DIR}/config/nvme_stress.yaml"
