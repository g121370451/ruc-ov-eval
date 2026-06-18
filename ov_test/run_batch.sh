#!/bin/bash
# Run MoDora benchmark configs sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

step="${1:-all}"

for config in config_modora/*.yaml; do
    echo "[MoDora] Running $config step=$step"
    uv run python run.py --config "$config" --step "$step"
done
