"""
Shared evaluation harness for every model. Honest numbers, including where models fail.

    python eval.py --model ts_jepa --ckpt checkpoints/ts_jepa.pt

Reports (all under FREE multi-step rollout, not teacher forcing):
  * params + budget flag (<2M target, <4M ceiling)
  * ratchet / fast / all MAE at K=24 on held-out val vs a persist-last naive baseline
  * constraint-violation rate over a full rollout (K=1) -- should be exactly 0 by construction
    (0 by projection for ts_diffusion)
  * generalisation probes: ratchet MAE on held-out-susceptibility and unseen-treatment-timing
    cohorts, so the OOD gap is visible
  * latent effective rank vs the data's intrinsic rank (collapse check) for latent models
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from dataloader import (get_data, get_long_probe, reconstruct_patients, _build_arrays,
                        make_splits)
from Data_Generator import simulate_conditioned
from models import build, REGISTRY
from util import (count_params, effective_rank, mae_over, constraint_violations,
                  RATCHETS, FAST, set_seed)

K_EVAL = 24
K_CONSTRAINT = 1
TRAIN_T = 60


def persist_mae(batch, K, cols):
    x = batch["x"]
    last = x[:, K:K + 1].expand(-1, x.shape[1] - K - 1, -1)
    return mae_over(last, x[:, K + 1:], cols)


def noise_floor(csv, K, seed=0, n_patients=60, n_real=30):
    """Irreducible aleatoric floor of the task: re-run the generator CONDITIONED on each val
    patient's observed 0..K, and measure the spread of futures around their own mean. A model
    cannot beat this. Returns (ratchet_floor, fast_floor)."""
    X, ctx, ercp, susc, udca = _build_arrays(csv)
    _, val_idx, _ = make_splits(susc, udca, seed)
    idx = list(val_idx[:n_patients])
    pats = reconstruct_patients(csv, idx)
    T = X.shape[1]
    rng = np.random.default_rng(seed + 123)
    rat, fast = [], []
    for i, p in zip(idx, pats):
        runs = np.stack([simulate_conditioned(p, X[i], K, T, rng) for _ in range(n_real)])
        dev = np.abs(runs - runs.mean(0))[:, K + 1:]
        rat.append(dev[..., RATCHETS].mean())
        fast.append(dev[..., FAST].mean())
    return float(np.mean(rat)), float(np.mean(fast))


def noise_floor_full(csv, csv_long=None, K=K_EVAL, seed=0, n_patients=60, n_real=30):
    """Every floor the eval actually needs, in one pass. Returns a flat dict.

    Two corrections over `noise_floor` above, which is left untouched so existing scripts keep
    their numbers:

    1. CENTRE. `noise_floor` reports mean|X - mean(X)|. The eval headlines MAE, and MAE is
       minimised by the conditional MEDIAN, not the mean; the ratchet's conditional law is
       right-skewed (bounded below by `prev`). So the mean-centred value is an UPPER bound on
       achievable MAE, not a floor -- measured loose by 4.5% on the ratchets and 22% on the fast
       fields, which is why the sweep's best `fast_mae` (0.1186) appeared to beat its own
       "irreducible" floor (0.1297). Both are returned; `*_median` is the real floor.

    2. COVERAGE. Floors were only ever computed for the in-distribution validation cohort, so
       the OOD-susceptibility, unseen-UDCA, and beyond-horizon numbers had nothing to be read
       against -- nobody could say whether 0.0766 on the susc probe was near its floor or 5x
       away. Each cohort gets its own floor here, since each has its own aleatoric spread.
    """
    X, ctx, ercp, susc, udca = _build_arrays(csv)
    _, val_idx, probe_idx = make_splits(susc, udca, seed)
    T = X.shape[1]
    out = {}

    def floor_for(idx, tag, Xs, csv_path, n_months):
        idx = list(idx[:n_patients])
        if not idx:
            return
        pats = reconstruct_patients(csv_path, idx)
        rng = np.random.default_rng(seed + 123)
        acc = {"ratchet_mean": [], "ratchet_median": [], "fast_mean": [], "fast_median": []}
        for i, p in zip(idx, pats):
            runs = np.stack([simulate_conditioned(p, Xs[i], K, n_months, rng)
                             for _ in range(n_real)])[:, K + 1:]
            for centre, fn in (("mean", runs.mean(0, keepdims=True)),
                               ("median", np.median(runs, axis=0, keepdims=True))):
                dev = np.abs(runs - fn)
                acc[f"ratchet_{centre}"].append(dev[..., RATCHETS].mean())
                acc[f"fast_{centre}"].append(dev[..., FAST].mean())
        for k, v in acc.items():
            out[f"floor_{tag}_{k}"] = float(np.mean(v))

    floor_for(val_idx, "val", X, csv, T)
    for name, idx in probe_idx.items():
        floor_for(idx, name, X, csv, T)

    if csv_long and os.path.exists(csv_long):
        Xl, _, _, suscl, _ = _build_arrays(csv_long)
        band = np.where((suscl >= 0.5) & (suscl <= 2.0))[0]
        idx = list(band[:n_patients])
        pats = reconstruct_patients(csv_long, idx)
        Tl = Xl.shape[1]
        rng = np.random.default_rng(seed + 123)
        acc = {"in_mean": [], "in_median": [], "beyond_mean": [], "beyond_median": []}
        for i, p in zip(idx, pats):
            runs = np.stack([simulate_conditioned(p, Xl[i], K, Tl, rng) for _ in range(n_real)])
            for centre, fn in (("mean", runs.mean(0, keepdims=True)),
                               ("median", np.median(runs, axis=0, keepdims=True))):
                dev = np.abs(runs - fn)
                acc[f"in_{centre}"].append(dev[:, K + 1:TRAIN_T][..., RATCHETS].mean())
                acc[f"beyond_{centre}"].append(dev[:, TRAIN_T:][..., RATCHETS].mean())
        for k, v in acc.items():
            out[f"floor_long_{k}"] = float(np.mean(v))
    return out


def longer_horizon_probe(model, csv_long, seed):
    """Roll a T=60-trained model out to 96 months; report ratchet MAE split at the training
    horizon. The beyond-60 number is where models break (and where learned-position models
    degrade hardest)."""
    batch, T_long = get_long_probe(csv_long, seed)
    with torch.no_grad():
        pred = model.rollout(batch["x"], batch["ctx"], batch["ercp"], K_EVAL)
    true = batch["x"]
    in_h = mae_over(pred[:, K_EVAL + 1:TRAIN_T], true[:, K_EVAL + 1:TRAIN_T], RATCHETS)
    beyond = mae_over(pred[:, TRAIN_T:], true[:, TRAIN_T:], RATCHETS)
    return {"in_horizon": in_h, "beyond_horizon": beyond, "T_long": T_long}


def eval_model(model, val, probes, T):
    model.eval()
    r = {}
    with torch.no_grad():
        pred = model.rollout(val["x"], val["ctx"], val["ercp"], K_EVAL)
    true = val["x"]
    r["ratchet_mae"] = mae_over(pred[:, K_EVAL + 1:], true[:, K_EVAL + 1:], RATCHETS)
    r["fast_mae"] = mae_over(pred[:, K_EVAL + 1:], true[:, K_EVAL + 1:], FAST)
    r["all_mae"] = mae_over(pred[:, K_EVAL + 1:], true[:, K_EVAL + 1:], list(range(8)))
    r["persist_ratchet_mae"] = persist_mae(val, K_EVAL, RATCHETS)

    with torch.no_grad():
        pc = model.rollout(val["x"], val["ctx"], val["ercp"], K_CONSTRAINT)
    v, n, rate = constraint_violations(pc[:, K_CONSTRAINT:], val["ercp"][:, K_CONSTRAINT:])
    r["constraint_violations"] = v
    r["constraint_rate"] = rate

    r["probes"] = {}
    for name, pb in probes.items():
        with torch.no_grad():
            pp = model.rollout(pb["x"], pb["ctx"], pb["ercp"], K_EVAL)
        r["probes"][name] = mae_over(pp[:, K_EVAL + 1:], pb["x"][:, K_EVAL + 1:], RATCHETS)

    if hasattr(model, "latent"):
        with torch.no_grad():
            z = model.latent(val)
        r["latent_eff_rank"] = effective_rank(z)
    r["data_eff_rank"] = effective_rank(val["x"].reshape(-1, 8))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--csv", default="liver_data.csv")
    ap.add_argument("--csv-long", default="liver_data_long.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    _, val, probes, T = get_data(args.csv, seed=args.seed)
    ckpt = args.ckpt or f"checkpoints/{args.model}.pt"
    scale = 1.0
    try:
        sd = torch.load(ckpt, map_location="cpu")
        scale = sd.get("scale", 1.0)                           # rebuild the trained architecture
        model = build(args.model, T, scale=scale)
        model.load_state_dict(sd["state_dict"])
        print(f"[load] {ckpt}  (scale={scale})")
    except FileNotFoundError:
        model = build(args.model, T)
        print(f"[warn] no checkpoint at {ckpt} -- evaluating UNTRAINED model")

    n = count_params(model)
    flag = "OK(<2M)" if n < 2_000_000 else ("<4M" if n < 4_000_000 else "OVER-BUDGET")
    r = eval_model(model, val, probes, T)

    # noise floor (task-level, model-independent) + longer-than-training probe
    rat_floor, fast_floor = noise_floor(args.csv, K_EVAL, args.seed)
    r["noise_floor_ratchet"] = rat_floor
    r["noise_floor_fast"] = fast_floor
    if os.path.exists(args.csv_long):
        r["longer_horizon"] = longer_horizon_probe(model, args.csv_long, args.seed)

    print(f"\n=== {args.model}  (params={n:,} {flag}) ===")
    print(f"  ratchet MAE  (K={K_EVAL}, free rollout) : {r['ratchet_mae']:.4f}   "
          f"[persist {r['persist_ratchet_mae']:.4f} | noise floor {rat_floor:.4f}]")
    print(f"  fast MAE                                : {r['fast_mae']:.4f}   "
          f"[noise floor {fast_floor:.4f}]")
    print(f"  all-field MAE                           : {r['all_mae']:.4f}")
    print(f"  constraint-violation rate (full rollout): {r['constraint_rate']:.8f} "
          f"({r['constraint_violations']} viol)")
    print("  OOD probes (ratchet MAE):")
    for k, v in r["probes"].items():
        print(f"    {k:16s} : {v:.4f}")
    if "longer_horizon" in r:
        lh = r["longer_horizon"]
        print(f"    longer_horizon   : in-horizon [25:60] {lh['in_horizon']:.4f}  "
              f"beyond [60:{lh['T_long']}] {lh['beyond_horizon']:.4f}")
    if "latent_eff_rank" in r:
        print(f"  latent eff-rank {r['latent_eff_rank']:.2f}  (data intrinsic {r['data_eff_rank']:.2f})")

    if args.json:
        r["params"] = n
        with open(args.json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"[json] {args.json}")


if __name__ == "__main__":
    main()
