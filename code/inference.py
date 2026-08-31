"""
Inference / prediction utility: load a trained model and roll a patient's future forward.

    python inference.py --model ts_jepa --ckpt checkpoints/ts_jepa.pt --patient 0 --observe 24

Prints the predicted vs true trajectory for the ratchet fields and flags any constraint
violation. Also exposes `predict()` for programmatic use (e.g. from a notebook or a UI).
"""

from __future__ import annotations
import argparse

import numpy as np
import torch

from dataloader import get_data, _build_arrays, _stack
from models import build, REGISTRY
from util import FIELD_NAMES, RATCHETS, constraint_violations


def load(model_name, ckpt, T):
    model = build(model_name, T)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model


def predict(model, batch, K):
    """Return predicted trajectory [B, T, 8] observing months 0..K."""
    with torch.no_grad():
        return model.rollout(batch["x"], batch["ctx"], batch["ercp"], K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--csv", default="liver_data.csv")
    ap.add_argument("--patient", type=int, default=0, help="index into the val set")
    ap.add_argument("--observe", type=int, default=24)
    args = ap.parse_args()

    X, ctx, ercp, susc, udca = _build_arrays(args.csv)
    T = X.shape[1]
    idx = np.array([args.patient])
    batch = _stack((X, ctx, ercp, susc), idx)

    model = load(args.model, args.ckpt or f"checkpoints/{args.model}.pt", T)
    pred = predict(model, batch, args.observe)

    v, _, rate = constraint_violations(pred, batch["ercp"])
    print(f"patient {args.patient}  observe 0..{args.observe}  predict {args.observe + 1}..{T - 1}")
    print(f"constraint violations in this trajectory: {v} (rate {rate:.6f})")
    print("\nratchet fields  (month : true -> pred)")
    for j in RATCHETS:
        name = FIELD_NAMES[j]
        line = "  ".join(
            f"{t}:{batch['x'][0, t, j]:.2f}->{pred[0, t, j]:.2f}"
            for t in range(args.observe + 1, T, max(1, (T - args.observe) // 6))
        )
        print(f"  {name:5s} {line}")


if __name__ == "__main__":
    main()
