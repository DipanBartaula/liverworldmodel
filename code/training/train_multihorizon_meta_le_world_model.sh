#!/usr/bin/env bash
# Train + evaluate multihorizon_meta_le_world_model. Usage: ./train_...sh [EPOCHS] [BATCH] [SCALE]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=multihorizon_meta_le_world_model
EPOCHS="${1:-30}"; BATCH="${2:-128}"; SCALE="${3:-1.0}"
python training.py --model "$MODEL" --epochs "$EPOCHS" --batch-size "$BATCH" --scale "$SCALE" --out "checkpoints/${MODEL}.pt"
python eval.py --model "$MODEL" --ckpt "checkpoints/${MODEL}.pt" --json "checkpoints/${MODEL}.json"
