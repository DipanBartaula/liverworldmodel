#!/usr/bin/env bash
# Train + evaluate ttt_val_model. Usage: ./train_ttt_val_model.sh [EPOCHS] [BATCH]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=ttt_val_model
EPOCHS="${1:-30}"
BATCH="${2:-128}"
python training.py --model "$MODEL" --epochs "$EPOCHS" --batch-size "$BATCH" --out "checkpoints/${MODEL}.pt"
python eval.py --model "$MODEL" --json "checkpoints/${MODEL}.json"
