"""
Parameter-scaling sweep: for every model, train + evaluate 10 sizes spanning ~25K-250K params,
recording every metric. Results feed the scaling curves and the PDF report.

    python sweep.py --epochs 10 --points 10 --min-params 25000 --max-params 250000

Design:
  * For each (model, target-params) we binary-search the width `scale` that hits the target,
    then train and evaluate. The ACTUAL param count is recorded (x-axis of the curves).
  * The irreducible noise floor and the longer-horizon probe batch are model-independent, so
    they are computed ONCE and reused across all 130 runs.
  * Resumable: a run whose JSON already exists is skipped, so an interrupted sweep resumes.
"""

from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import torch

from dataloader import get_data, get_long_probe
from models import build, REGISTRY
from training import train_model
from eval import eval_model, noise_floor, noise_floor_full, K_EVAL, TRAIN_T
from util import count_params, mae_over, RATCHETS, set_seed

RESULT_DIR = "sweep_results"      # overridable via --result-dir
CKPT_DIR = "sweep_ckpts"          # overridable via --ckpt-dir

# model families for the stratified analysis (base -> small modification chains)
FAMILIES = {
    "transformer": ["simple_transformer", "ttt_transformer", "ebt"],
    "jepa_latent": ["le_world_model", "ts_jepa", "node_world_model"],
    "meta_search": ["meta_le_world_model", "ttt_val_model"],
    "multi_horizon": ["multihorizon_le_world_model", "multihorizon_meta_le_world_model"],
    "generative": ["genie_world_model", "ts_diffusion"],
    "graph": ["gnn_world_model"],
    "kan": ["kan_mh_meta"],
    "neural_sde": ["neural_sde", "sde_kan_mh", "sde_kan_mh_meta", "sde_leworld_latent"],
    "ebt_world": ["ebt_le_mh_meta"],
    "composed": ["gnn_le_meta", "gnn_sde", "kan_gnn_meta", "ebt_jepa_sde"],
    "neural_pde": ["kan_pde", "leworld_pde", "leworld_pde_meta", "gnn_leworld_meta_pde"],
    "leworld_ode": ["gnn_leworld_meta_ode"],
    "ada_jepa": ["ada_jepa", "gnn_ada_jepa", "mamba2_ada_jepa"],
    "fno_jepa": ["fno_jepa", "gnn_fno_jepa"],
    # single-ingredient additions to gnn_leworld_meta_ode, targeting the failures the
    # preregistered harness found (coefficient brittleness, counterfactual magnitude,
    # median-vs-mean objective mismatch, event channel)
    "rate_invariance": ["rate_anchor", "hazard_mono", "time_warp", "ude_hybrid"],
    "distributional": ["quantile_head", "tpp_events", "npe_head", "cf_paired"],
    # the same eight, plus the LeWorld/TS-JEPA anti-collapse pair (VICReg + EMA target)
    "anticollapse": ["rate_anchor_ac", "hazard_mono_ac", "time_warp_ac", "ude_hybrid_ac",
                     "quantile_head_ac", "tpp_events_ac", "npe_head_ac", "cf_paired_ac"],
}
ORDER = [m for fam in FAMILIES.values() for m in fam]


def solve_scale(name, T, target, iters=34):
    """Binary-search the width multiplier that yields ~target params (params grow with scale)."""
    lo, hi = 0.03, 12.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if count_params(build(name, T, scale=mid)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    global RESULT_DIR, CKPT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--points", type=int, default=10)
    ap.add_argument("--min-params", type=int, default=25000)
    ap.add_argument("--max-params", type=int, default=250000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--csv", default="liver_data.csv")
    ap.add_argument("--csv-long", default="liver_data_long.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--result-dir", default=RESULT_DIR)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    args = ap.parse_args()

    RESULT_DIR, CKPT_DIR = args.result_dir, args.ckpt_dir
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    set_seed(args.seed)

    print("loading data + computing model-independent quantities once...", flush=True)
    train_loader, val, probes, T = get_data(args.csv, args.batch, args.seed)
    long_batch, T_long = get_long_probe(args.csv_long, args.seed)
    nf_rat, nf_fast = noise_floor(args.csv, K_EVAL, args.seed)
    print(f"  T={T}  noise floor: ratchet={nf_rat:.4f} fast={nf_fast:.4f}", flush=True)
    # per-cohort, median-centred floors (see eval.noise_floor_full). Model-independent, so
    # computed once and copied into every run's JSON alongside the legacy mean-centred pair.
    floors = noise_floor_full(args.csv, args.csv_long, K_EVAL, args.seed)
    print("  floors: " + "  ".join(f"{k.replace('floor_','')}={v:.4f}"
                                   for k, v in sorted(floors.items())
                                   if k.endswith("_median") or k.endswith("median")), flush=True)

    targets = np.geomspace(args.min_params, args.max_params, args.points).round().astype(int)
    models = args.models or ORDER
    total = len(models) * len(targets)
    print(f"sweep: {len(models)} models x {len(targets)} sizes = {total} runs "
          f"[{args.min_params:,}..{args.max_params:,}] @ {args.epochs} epochs\n", flush=True)

    done, t0 = 0, time.time()
    for m in models:
        fam = next((f for f, ms in FAMILIES.items() if m in ms), "other")
        for tgt in targets:
            done += 1
            out = os.path.join(RESULT_DIR, f"{m}__{tgt}.json")
            if os.path.exists(out):
                print(f"[{done}/{total}] {m} @~{tgt:,}  (cached, skip)", flush=True)
                continue
            scale = solve_scale(m, T, tgt)
            model = build(m, T, scale=scale)
            n = count_params(model)
            print(f"[{done}/{total}] {m} @~{tgt:,} -> {n:,} params (scale={scale:.3f}) "
                  f"training {args.epochs} ep...", flush=True)
            ts = time.time()
            train_model(model, train_loader, val, T, args.epochs, log=False)

            r = eval_model(model, val, probes, T)
            with torch.no_grad():
                lp = model.rollout(long_batch["x"], long_batch["ctx"], long_batch["ercp"], K_EVAL)
            r["h_in"] = mae_over(lp[:, K_EVAL + 1:TRAIN_T], long_batch["x"][:, K_EVAL + 1:TRAIN_T], RATCHETS)
            r["h_beyond"] = mae_over(lp[:, TRAIN_T:], long_batch["x"][:, TRAIN_T:], RATCHETS)
            r.update({"model": m, "family": fam, "target": int(tgt), "params": n,
                      "scale": scale, "epochs": args.epochs,
                      "noise_floor_ratchet": nf_rat, "noise_floor_fast": nf_fast})
            r.update(floors)
            with open(out, "w") as f:
                json.dump(r, f, indent=2)

            eta = (time.time() - t0) / done * (total - done)
            print(f"    ratchet={r['ratchet_mae']:.4f} susc={r['probes'].get('held_out_susc', 0):.4f} "
                  f"h>60={r['h_beyond']:.4f} viol={r['constraint_rate']:.1e} "
                  f"({time.time() - ts:.0f}s, ETA {eta/60:.0f} min)", flush=True)

    print(f"\nSWEEP DONE: {total} runs in {(time.time() - t0)/60:.1f} min -> {RESULT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
