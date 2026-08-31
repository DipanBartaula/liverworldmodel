#!/usr/bin/env bash
# Train + evaluate every model in sequence. Usage: ./train_all.sh [EPOCHS] [BATCH]
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS="${1:-30}"
BATCH="${2:-128}"
MODELS="le_world_model ts_jepa ebt simple_transformer ttt_transformer node_world_model meta_le_world_model genie_world_model ts_diffusion gnn_world_model multihorizon_le_world_model ttt_val_model multihorizon_meta_le_world_model"
for M in $MODELS; do
  echo "==================== $M ===================="
  python training.py --model "$M" --epochs "$EPOCHS" --batch-size "$BATCH" --out "checkpoints/${M}.pt"
done
