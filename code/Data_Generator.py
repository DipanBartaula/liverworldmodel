"""
LiverModel.py -- synthetic Digital Liver trajectory generator (portable CLI).

Ported verbatim (dynamics-wise) from the digital_liver `generator.py` reference so the two
repos emit identical trajectories for the same seed. Emits monthly trajectories of an 8-D
clinical state x(t) and writes them to a long-format CSV (one row per patient-month).

State layout x(t) in R^8, monthly timesteps. All fields in [0,1] except M in [0,2].

    idx  field  meaning                        temporal behaviour
    0    F      fibrosis                        ratchet, non-decreasing
    1    D      ductopenia (duct loss)          ratchet, irreversible (non-decreasing)
    2    S      biliary strictures              ratchet up, steps DOWN at an ERCP event
    3    P      portal hypertension             ratchet, non-decreasing
    4    A      inflammatory activity           fast, mean-reverting
    5    C      cholestasis                     fast, with flares
    6    M      malignancy hazard accumulator   monotone non-decreasing, in [0,2]
    7    flare  acute cholangitis flare         transient, decays

Context constants (supplied to a model, NOT predicted): disease_class, age, sex,
responder in {0,1}, udca_start (month), ercp_months (list), susceptibility (hidden).

Usage:
    python LiverModel.py --n-patients 10000 --n-months 60 --seed 0 --out liver_data.csv
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

# --- field indices (single source of truth) ----------------------------------------------
F, D, S, P, A, C, M, FLARE = range(8)
FIELD_NAMES = ["F", "D", "S", "P", "A", "C", "M", "flare"]
N_FIELDS = 8

# fields that must never decrease month-to-month (S handled separately: may drop at ERCP)
MONOTONE_UP = (F, D, P, M)
FIELD_MAX = np.array([1, 1, 1, 1, 1, 1, 2, 1], dtype=np.float32)  # per-field upper bound


@dataclass
class Patient:
    """Context constants + hidden per-patient parameters that drive the dynamics."""
    disease_class: int          # 0..2, baseline inflammation/cholestasis "aggressiveness"
    age: float                  # normalised ~[0,1]
    sex: int                    # 0/1 (kept for realism; weak effect)
    responder: int              # 1 => treatment actually suppresses A and C
    udca_start: int             # month UDCA therapy begins (>= n_months => never)
    ercp_months: List[int] = field(default_factory=list)  # months an ERCP is performed
    susceptibility: float = 1.0  # hidden per-patient multiplier on ratchet creep speed


def sample_patient(rng: np.random.Generator, n_months: int,
                   susc_range=None, udca_start_range=None) -> Patient:
    """Draw one patient's context + hidden parameters."""
    disease_class = int(rng.integers(0, 3))
    responder = int(rng.random() < 0.6)                      # 60% respond to therapy
    if udca_start_range is not None:
        udca_start = int(rng.integers(udca_start_range[0], udca_start_range[1]))
    else:
        udca_start = int(rng.integers(2, n_months // 2)) if rng.random() < 0.7 else n_months + 1
    n_ercp = int(rng.integers(0, 3))
    ercp_months = sorted(int(m) for m in rng.integers(6, n_months, size=n_ercp))
    susc = float(rng.lognormal(mean=0.0, sigma=0.80))        # median 1, wide spread
    if susc_range is not None:
        lo, hi = susc_range
        while not (lo <= susc <= hi):                        # reject until inside the band
            susc = float(rng.lognormal(mean=0.0, sigma=0.80))
    return Patient(
        disease_class=disease_class,
        age=float(rng.uniform(0.2, 0.9)),
        sex=int(rng.integers(0, 2)),
        responder=responder,
        udca_start=udca_start,
        ercp_months=ercp_months,
        susceptibility=susc,
    )


def _baselines(p: Patient):
    a_base = 0.15 + 0.10 * p.disease_class      # inflammatory set point
    c_base = 0.15 + 0.10 * p.disease_class      # cholestatic set point
    susc = p.susceptibility * (0.8 + 0.4 * p.age)  # older patients ratchet a little faster
    return a_base, c_base, susc


def _advance(prev, p, a_base, c_base, susc, t, rng):
    """One month of the dynamics: prev state -> next state. The single source of the update
    rule, reused by simulate() and simulate_conditioned() so the noise-floor continuations
    obey EXACTLY the same generator (no drift)."""
    cur = prev.copy()

    on_udca = (t >= p.udca_start) and (p.responder == 1)
    supp = 0.6 if on_udca else 0.0              # 60% knock-down of A/C set points

    # --- flare (idx 7): transient. Random onset, then geometric decay. -------------------
    p_onset = 0.04 + 0.10 * prev[S]
    onset = 1.0 if rng.random() < p_onset else 0.0
    cur[FLARE] = max(onset, prev[FLARE] * 0.4)

    # --- A (idx 4): fast mean-reverting toward a (possibly treated) set point + flare -----
    a_set = a_base * (1 - supp)
    cur[A] = prev[A] + 0.5 * (a_set - prev[A]) + 0.5 * cur[FLARE] + 0.03 * rng.standard_normal()
    cur[A] = np.clip(cur[A], 0, 1)

    # --- C (idx 5): same shape as A; treatment suppresses it too -------------------------
    c_set = c_base * (1 - supp)
    cur[C] = prev[C] + 0.4 * (c_set - prev[C]) + 0.5 * cur[FLARE] + 0.03 * rng.standard_normal()
    cur[C] = np.clip(cur[C], 0, 1)

    # --- ratchets F, D, P (idx 0,1,3): non-negative creep driven by A and C --------------
    drive = susc * (0.6 * prev[A] + 0.4 * prev[C])
    cur[F] = min(prev[F] + 0.022 * drive, 1.0)
    cur[D] = min(prev[D] + 0.015 * drive, 1.0)
    cur[P] = min(prev[P] + 0.011 * (drive + 0.5 * prev[F]), 1.0)

    # --- S (idx 2): creeps up with inflammation; ERCP steps it DOWN ----------------------
    cur[S] = min(prev[S] + 0.018 * susc * prev[A], 1.0)
    if t in p.ercp_months:
        cur[S] = max(cur[S] - 0.4, 0.0)          # transient mechanical relief

    # --- M (idx 6): hazard accumulator of sustained F*C. Monotone, capped at 2. ----------
    cur[M] = min(prev[M] + 0.05 * prev[F] * prev[C], 2.0)
    return cur


def simulate(p: Patient, n_months: int, rng: np.random.Generator) -> np.ndarray:
    """Roll one patient forward for n_months. Returns x of shape [n_months, 8]."""
    x = np.zeros((n_months, N_FIELDS), dtype=np.float32)
    a_base, c_base, susc = _baselines(p)
    x[0, A] = np.clip(a_base + 0.05 * rng.standard_normal(), 0, 1)
    x[0, C] = np.clip(c_base + 0.05 * rng.standard_normal(), 0, 1)
    for t in range(1, n_months):
        x[t] = _advance(x[t - 1], p, a_base, c_base, susc, t, rng)
    return x


def simulate_conditioned(p: Patient, x_obs, K: int, n_months: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Warm-start with observed months 0..K, then continue with FRESH noise K+1..n_months-1.
    Used to estimate the generator's irreducible (aleatoric) spread CONDITIONED on the same
    history a model sees -- i.e. the noise floor of the prediction task."""
    x = np.zeros((n_months, N_FIELDS), dtype=np.float32)
    x[: K + 1] = np.asarray(x_obs)[: K + 1]
    a_base, c_base, susc = _baselines(p)
    for t in range(K + 1, n_months):
        x[t] = _advance(x[t - 1], p, a_base, c_base, susc, t, rng)
    return x


def _project(cur, prev, t, p):
    """Project a (possibly noisy) state onto the constraint set given the previous state:
    F,D,P,M non-decreasing; S non-decreasing except at an ERCP month; all fields in bounds.
    Guarantees NO constraint violation even after adding noise."""
    cur = np.clip(cur, 0.0, FIELD_MAX)
    for i in (F, D, P, M):
        cur[i] = max(cur[i], prev[i])
    if t not in p.ercp_months:
        cur[S] = max(cur[S], prev[S])
    return np.clip(cur, 0.0, FIELD_MAX)


def simulate_stochastic(p: Patient, n_months: int, rng: np.random.Generator,
                        sigma: float = 0.03) -> np.ndarray:
    """Like simulate() but adds Gaussian noise to every field each month, then PROJECTS back
    onto the constraint set -- a stochastic substrate on top of x(t) that never violates the
    one-directional/bounds constraints (the brief's 'real imaging pipeline' stochasticity)."""
    x = np.zeros((n_months, N_FIELDS), dtype=np.float32)
    a_base, c_base, susc = _baselines(p)
    x[0, A] = np.clip(a_base + 0.05 * rng.standard_normal(), 0, 1)
    x[0, C] = np.clip(c_base + 0.05 * rng.standard_normal(), 0, 1)
    for t in range(1, n_months):
        cur = _advance(x[t - 1], p, a_base, c_base, susc, t, rng)
        cur = cur + sigma * rng.standard_normal(N_FIELDS).astype(np.float32)
        x[t] = _project(cur, x[t - 1], t, p)
    return x


def generate_dataframe(n_patients: int, n_months: int, seed: int,
                       susc_range=None, udca_start_range=None, sigma: float = 0.0) -> pd.DataFrame:
    """Generate a long-format DataFrame: one row per (patient, month).
    sigma>0 uses the stochastic (constraint-projected Gaussian-noise) generator."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        p = sample_patient(rng, n_months, susc_range=susc_range, udca_start_range=udca_start_range)
        x = simulate_stochastic(p, n_months, rng, sigma) if sigma > 0 else simulate(p, n_months, rng)
        ercp_str = ";".join(str(m) for m in p.ercp_months)  # avoid CSV comma clash
        for t in range(n_months):
            rows.append({
                "patient_id": pid,
                "month": t,
                **{FIELD_NAMES[i]: float(x[t, i]) for i in range(N_FIELDS)},
                "disease_class": p.disease_class,
                "age": p.age,
                "sex": p.sex,
                "responder": p.responder,
                "udca_start": p.udca_start,
                "ercp_months": ercp_str,
                "susceptibility": p.susceptibility,
            })
    return pd.DataFrame(rows)


def check_constraints(df: pd.DataFrame) -> None:
    """Assert the generated data satisfies the one-directional + bounds constraints."""
    g = df.sort_values(["patient_id", "month"]).groupby("patient_id")
    for name in [FIELD_NAMES[i] for i in MONOTONE_UP]:
        dmin = g[name].diff().min()
        assert dmin >= -1e-6, f"{name} decreased! (min delta={dmin})"
    for i, name in enumerate(FIELD_NAMES):
        lo, hi = df[name].min(), df[name].max()
        assert lo >= -1e-6 and hi <= FIELD_MAX[i] + 1e-6, f"{name} out of bounds [{lo}, {hi}]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Digital Liver synthetic trajectory generator")
    ap.add_argument("--n-patients", type=int, default=10000, help="number of patient trajectories")
    ap.add_argument("--n-months", type=int, default=60, help="months per trajectory")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--out", type=str, default="liver_data.csv", help="output CSV path")
    ap.add_argument("--sigma", type=float, default=0.0, help="Gaussian noise std (>0 => stochastic)")
    ap.add_argument("--no-check", action="store_true", help="skip constraint self-check")
    args = ap.parse_args()

    t0 = time.time()
    kind = f"stochastic(sigma={args.sigma})" if args.sigma > 0 else "clean"
    print(f"[generate] {args.n_patients} patients x {args.n_months} months, seed={args.seed} [{kind}]")
    df = generate_dataframe(args.n_patients, args.n_months, args.seed, sigma=args.sigma)
    print(f"[generate] built {len(df):,} rows in {time.time() - t0:.1f}s")

    if not args.no_check:
        check_constraints(df)
        print("[check]    OK: monotone fields never decrease; all fields in range")

    df.to_csv(args.out, index=False)
    print(f"[write]    {args.out}  ({len(df):,} rows, {df.shape[1]} cols, "
          f"total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
