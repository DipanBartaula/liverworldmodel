#!/usr/bin/env bash
# Train + evaluate ebt. Usage: ./train_ebt.sh [EPOCHS] [BATCH]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=ebt
EPOCHS="${1:-30}"
BATCH="${2:-128}"
python training.py --model "$MODEL" --epochs "$EPOCHS" --batch-size "$BATCH" --out "checkpoints/${MODEL}.pt"
python eval.py --model "$MODEL" --json "checkpoints/${MODEL}.json"
