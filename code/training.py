"""
Shared training entry point for every model.

    python training.py --model ts_jepa --epochs 20 --batch-size 128 --out checkpoints/ts_jepa.pt

Every model exposes the same `training_step(batch) -> {"loss": ...}` contract, so this loop is
model-agnostic. `train_model` is factored out so the scaling-sweep harness can reuse it in
process (no subprocess overhead).
"""

from __future__ import annotations
import argparse
import os
import time

import torch

from dataloader import get_data
from models import build, REGISTRY
from util import count_params, set_seed


def train_model(model, train_loader, val_batch, T, epochs, lr=1e-3, log=True):
    """Adam on model.training_step's loss. Returns the trained model."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            loss = model.training_step(batch)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        if log:
            model.eval()
            with torch.no_grad():
                pred = model.rollout(val_batch["x"], val_batch["ctx"], val_batch["ercp"], T // 2)
                vmae = (pred[:, T // 2 + 1:] - val_batch["x"][:, T // 2 + 1:]).abs().mean().item()
            print(f"  ep {ep + 1:02d}/{epochs}  train_loss={tot / max(nb, 1):.5f}  "
                  f"val_rollout_mae={vmae:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--csv", default="liver_data.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="width multiplier (sqrt(3)~triples params)")
    ap.add_argument("--max-train", type=int, default=None, help="cap #train patients (smoke tests)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    out = args.out or f"checkpoints/{args.model}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    train_loader, val_batch, _, T = get_data(args.csv, args.batch_size, args.seed, args.max_train)
    model = build(args.model, T, scale=args.scale)
    n = count_params(model)
    flag = "OK" if n < 2_000_000 else ("<4M" if n < 4_000_000 else "OVER-BUDGET")
    print(f"[model] {args.model}  params={n:,}  ({flag})")

    train_model(model, train_loader, val_batch, T, args.epochs, args.lr, log=True)

    torch.save({"model": args.model, "T": T, "scale": args.scale,
                "state_dict": model.state_dict(), "params": n}, out)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
