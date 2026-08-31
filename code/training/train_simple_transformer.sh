#!/usr/bin/env bash
# Train + evaluate simple_transformer. Usage: ./train_simple_transformer.sh [EPOCHS] [BATCH]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=simple_transformer
EPOCHS="${1:-30}"
BATCH="${2:-128}"
python training.py --model "$MODEL" --epochs "$EPOCHS" --batch-size "$BATCH" --out "checkpoints/${MODEL}.pt"
python eval.py --model "$MODEL" --json "checkpoints/${MODEL}.json"
