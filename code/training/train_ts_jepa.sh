#!/usr/bin/env bash
# Train + evaluate ts_jepa. Usage: ./train_ts_jepa.sh [EPOCHS] [BATCH]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=ts_jepa
EPOCHS="${1:-30}"
BATCH="${2:-128}"
python training.py --model "$MODEL" --epochs "$EPOCHS" --batch-size "$BATCH" --out "checkpoints/${MODEL}.pt"
python eval.py --model "$MODEL" --json "checkpoints/${MODEL}.json"
