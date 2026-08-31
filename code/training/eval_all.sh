#!/usr/bin/env bash
# Evaluate every trained model and print a comparison table. Usage: ./eval_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="le_world_model ts_jepa ebt simple_transformer ttt_transformer node_world_model meta_le_world_model genie_world_model ts_diffusion gnn_world_model multihorizon_le_world_model ttt_val_model multihorizon_meta_le_world_model"
for M in $MODELS; do
  python eval.py --model "$M" --json "checkpoints/${M}.json"
done
