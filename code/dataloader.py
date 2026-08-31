"""
Load the generated liver_data.csv into per-patient trajectory tensors + context features,
and carve out in-distribution train/val plus out-of-distribution probe cohorts.

Batch dict fed to every model:
    x    : [B, T, 8]   float  clinical state trajectory
    ctx  : [B, T, 8]   float  per-timestep context [dc0,dc1,dc2, age, sex, responder, on_udca, is_ercp]
    ercp : [B, T]      float  1.0 on an ERCP month (target-month aligned in the head)
    susc : [B]         float  HIDDEN susceptibility (never fed to a model; probes/analysis only)

Splits (from the existing CSV, no regeneration needed):
    train / val   : in-distribution   susceptibility in [0.5, 2.0] AND udca_start < 35
    probe cohorts :
        held_out_susc : susceptibility > 2.0        (faster progressors, unseen)
        unseen_udca   : udca_start >= 35, susc<=2.0 (late treatment timing, unseen)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from util import CTX_DIM
from Data_Generator import Patient

_DEF_CSV = "liver_data.csv"
TRAIN_SUSC = (0.5, 2.0)
LATE_UDCA = 35


def _build_arrays(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["patient_id", "month"]).reset_index(drop=True)
    pids = df["patient_id"].unique()
    T = int(df["month"].max()) + 1
    N = len(pids)

    state_cols = ["F", "D", "S", "P", "A", "C", "M", "flare"]
    X = df[state_cols].to_numpy(np.float32).reshape(N, T, 8)

    # per-patient constants
    g = df.groupby("patient_id")
    dc = g["disease_class"].first().to_numpy()
    age = g["age"].first().to_numpy(np.float32)
    sex = g["sex"].first().to_numpy(np.float32)
    resp = g["responder"].first().to_numpy(np.float32)
    udca = g["udca_start"].first().to_numpy(np.float32)
    susc = g["susceptibility"].first().to_numpy(np.float32)

    months = np.arange(T)[None, :]                       # [1, T]
    on_udca = (months >= udca[:, None]).astype(np.float32)  # [N, T]

    ercp = np.zeros((N, T), np.float32)
    ercp_raw = g["ercp_months"].first().fillna("").astype(str).to_numpy()
    for i, s in enumerate(ercp_raw):
        for tok in s.split(";"):
            if tok.strip() != "":
                m = int(tok)
                if 0 <= m < T:
                    ercp[i, m] = 1.0

    # context features [N, T, 8]
    dc_oh = np.eye(3, dtype=np.float32)[dc]              # [N, 3]
    const = np.concatenate([dc_oh, age[:, None], sex[:, None], resp[:, None]], axis=1)  # [N,6]
    ctx = np.zeros((N, T, CTX_DIM), np.float32)
    ctx[:, :, :6] = const[:, None, :]
    ctx[:, :, 6] = on_udca
    ctx[:, :, 7] = ercp
    return X, ctx, ercp, susc, udca


def make_splits(susc, udca, seed=0):
    rng = np.random.default_rng(seed)
    in_dist = (susc >= TRAIN_SUSC[0]) & (susc <= TRAIN_SUSC[1]) & (udca < LATE_UDCA)
    idx = np.where(in_dist)[0]
    rng.shuffle(idx)
    n_val = max(1, int(0.2 * len(idx)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    probes = {
        "held_out_susc": np.where(susc > TRAIN_SUSC[1])[0],
        "unseen_udca": np.where((udca >= LATE_UDCA) & (susc <= TRAIN_SUSC[1]))[0],
    }
    return train_idx, val_idx, probes


class LiverDataset(Dataset):
    def __init__(self, X, ctx, ercp, susc, idx):
        self.X, self.ctx, self.ercp, self.susc, self.idx = X, ctx, ercp, susc, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return {
            "x": torch.from_numpy(self.X[j]),
            "ctx": torch.from_numpy(self.ctx[j]),
            "ercp": torch.from_numpy(self.ercp[j]),
            "susc": torch.tensor(self.susc[j]),
        }


def _stack(ds, idx):
    return {
        "x": torch.from_numpy(ds[0][idx]),
        "ctx": torch.from_numpy(ds[1][idx]),
        "ercp": torch.from_numpy(ds[2][idx]),
        "susc": torch.from_numpy(ds[3][idx]),
    }


def get_data(csv_path: str = _DEF_CSV, batch_size: int = 128, seed: int = 0,
             max_train: int | None = None):
    """Returns (train_loader, val_batch, probe_batches, T). Batches are dicts of tensors."""
    X, ctx, ercp, susc, udca = _build_arrays(csv_path)
    train_idx, val_idx, probes = make_splits(susc, udca, seed)
    if max_train is not None:
        train_idx = train_idx[:max_train]
    T = X.shape[1]

    train_loader = DataLoader(
        LiverDataset(X, ctx, ercp, susc, train_idx),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    ds = (X, ctx, ercp, susc)
    val_batch = _stack(ds, val_idx)
    probe_batches = {k: _stack(ds, v) for k, v in probes.items() if len(v) > 0}
    return train_loader, val_batch, probe_batches, T


def reconstruct_patients(csv_path: str, indices, n_max=None):
    """Rebuild Patient objects (with hidden susceptibility) for the given patient indices, so
    the noise floor can be estimated by re-running the exact generator dynamics."""
    df = pd.read_csv(csv_path)
    g = df.sort_values(["patient_id", "month"]).groupby("patient_id")
    dc = g["disease_class"].first().to_numpy()
    age = g["age"].first().to_numpy()
    sex = g["sex"].first().to_numpy()
    resp = g["responder"].first().to_numpy()
    udca = g["udca_start"].first().to_numpy()
    susc = g["susceptibility"].first().to_numpy()
    ercp_raw = g["ercp_months"].first().fillna("").astype(str).to_numpy()
    pats = []
    idx = list(indices)[: n_max] if n_max else list(indices)
    for i in idx:
        months = [int(t) for t in str(ercp_raw[i]).split(";") if t.strip() != ""]
        pats.append(Patient(disease_class=int(dc[i]), age=float(age[i]), sex=int(sex[i]),
                            responder=int(resp[i]), udca_start=int(udca[i]),
                            ercp_months=months, susceptibility=float(susc[i])))
    return pats


def get_long_probe(csv_long: str, seed: int = 0):
    """Load the 96-month cohort as a longer-than-training probe batch [B, 96, 8].

    Uses only in-distribution patients (susceptibility in the training band) so the ONLY OOD
    axis is horizon. Returns (batch, T_long, train_T=60)."""
    X, ctx, ercp, susc, udca = _build_arrays(csv_long)
    in_band = np.where((susc >= TRAIN_SUSC[0]) & (susc <= TRAIN_SUSC[1]))[0]
    batch = _stack((X, ctx, ercp, susc), in_band)
    return batch, X.shape[1]
