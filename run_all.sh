#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA="${1:-data.zip}"
WORKERS="${WORKERS:-4}"
python benchmark.py --data "$DATA" --phase calibrate --out results/calibration
for SPLIT in train test; do
  python benchmark.py --data "$DATA" --phase benchmark --config results/calibration/locked_config.json --split "$SPLIT" --out "results/$SPLIT" --workers "$WORKERS" --resume
done
python benchmark.py --data "$DATA" --phase benchmark --config results/calibration/locked_config.json --split test --ablation-only --out results/ablation_test --workers "$WORKERS" --resume
