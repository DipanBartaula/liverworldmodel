"""
Shared building blocks for every LiverModel model.

The single most important thing here is `ConstraintHead`: it is the universal output layer
that ALL 10+ models route their final prediction through, so the domain's hard constraints
(F,D,P,M non-decreasing; S non-decreasing except a step-down at an ERCP month; M coupled to
sustained F*C; every field in bounds) hold BY CONSTRUCTION for every model regardless of its
internal machinery or weights. A model only ever emits a raw 9-vector per step; the head turns
that into a valid next state. This mirrors the reference generator's guarantees.
"""

from __future__ import annotations
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# --- state layout (single source of truth; matches Data_Generator.py) --------------------
F, D, S, P, A, C, M, FLARE = range(8)
FIELD_NAMES = ["F", "D", "S", "P", "A", "C", "M", "flare"]
STATE_DIM = 8
FIELD_MAX = [1, 1, 1, 1, 1, 1, 2, 1]
RATCHETS = [F, D, S, P, M]     # constrained / clinically decisive fields
FAST = [A, C, FLARE]           # fast, stochastic, near the noise floor
FREE = [A, C, FLARE]           # bounded but non-monotone

# context feature layout produced by dataloader: [dc0,dc1,dc2, age, sex, responder, on_udca, is_ercp]
CTX_DIM = 8

RAW_DIM = STATE_DIM + 1        # 8 field raws + 1 ERCP-relief raw -> ConstraintHead input width


# =========================================================================================
# The universal by-construction constraint head (shared by every model)
# =========================================================================================
class ConstraintHead(nn.Module):
    """raw (..., 9) + prev_x (..., 8) + is_ercp (...) -> valid next state (..., 8).

    raw[..., :8] are per-field raws, raw[..., 8] is the ERCP-relief raw.
    Guarantees (independent of weights):
      F,D,P,M : next = prev + softplus(raw)         >= prev      (non-decreasing)
      S       : next = prev + softplus(raw) - relief, relief only on ERCP months
      A,C,flr : next = sigmoid(raw)                              (free in [0,1])
      M       : (couple_m) increment scaled by prev_F*prev_C     (hazard of sustained F*C)
      all     : clamped to [0, fmax]                             (bounds)
    """

    # softplus(0-shift) ~ 0.018 ~ the generator's true monthly ratchet increment, so an
    # untrained/early model starts NEAR-STATIONARY instead of saturating the ratchets in a few
    # steps. The network shifts raw upward to make larger increments; this is just a good prior.
    INC_SHIFT = 4.0

    def __init__(self, couple_m: bool = True):
        super().__init__()
        self.couple_m = couple_m
        self.register_buffer("fmax", torch.tensor(FIELD_MAX, dtype=torch.float32))
        free_mask = torch.zeros(STATE_DIM, dtype=torch.bool)
        free_mask[FREE] = True
        self.register_buffer("free_mask", free_mask)
        s_onehot = torch.zeros(STATE_DIM); s_onehot[S] = 1.0
        self.register_buffer("s_onehot", s_onehot)
        m_onehot = torch.zeros(STATE_DIM); m_onehot[M] = 1.0
        self.register_buffer("m_onehot", m_onehot)

    def forward(self, raw, prev_x, is_ercp):
        raw_fields = raw[..., :STATE_DIM]
        raw_relief = raw[..., STATE_DIM]
        inc = Fn.softplus(raw_fields - self.INC_SHIFT)
        if self.couple_m:
            fc = (prev_x[..., F] * prev_x[..., C]).unsqueeze(-1)
            inc = inc * (1.0 - self.m_onehot + self.m_onehot * fc)
        val = torch.sigmoid(raw_fields)
        nxt = torch.where(self.free_mask, val, prev_x + inc)
        relief = Fn.softplus(raw_relief) * is_ercp
        nxt = nxt - relief.unsqueeze(-1) * self.s_onehot
        return torch.minimum(torch.clamp(nxt, min=0.0), self.fmax)


# =========================================================================================
# Embeddings
# =========================================================================================
class SinusoidalPosEmb(nn.Module):
    """Standard fixed sinusoidal positional embedding for sequence position / horizon."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, pos):
        # pos: (...,) integer or float positions -> (..., dim)
        pos = pos.float()
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=pos.device) / max(half - 1, 1))
        ang = pos.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.dim % 2:
            emb = Fn.pad(emb, (0, 1))
        return emb


class TimestepEmbedding(nn.Module):
    """Diffusion timestep embedding: sinusoidal -> MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.sin = SinusoidalPosEmb(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        return self.net(self.sin(t))


# Per-field loss weights: focus training on the clinically decisive ratchets, down-weight the
# near-random fast fields (A,C,flare) so models don't burn capacity fitting noise, and rescale
# M (range [0,2]) so it doesn't dominate the MSE. Makes training well-posed for the ratchet-MAE
# metric the eval headlines. Order: F,D,S,P,A,C,M,flare.
FIELD_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 0.3])


def state_loss(pred, true):
    """Field-weighted MSE between predicted and true states. pred/true: [..., 8]."""
    w = FIELD_WEIGHTS.to(pred.device)
    return (((pred - true) ** 2) * w).mean()


def mlp(sizes, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


# =========================================================================================
# Metrics / diagnostics
# =========================================================================================
def scale_dim(v, scale, multiple=1):
    """Scale a width hyperparameter by `scale`, rounded to a multiple (e.g. nhead). Since most
    params grow ~quadratically with width, scale=sqrt(3)~1.732 roughly TRIPLES parameter count."""
    v2 = int(round(v * scale))
    if multiple > 1:
        v2 = int(round(v2 / multiple)) * multiple
    return max(multiple, v2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def effective_rank(z: torch.Tensor) -> float:
    """Roy-Vetterli effective rank of a [N, d] representation. ~d = healthy, ->1 = collapsed."""
    z = z.detach().float()
    z = z - z.mean(0, keepdim=True)
    if z.shape[0] < 2:
        return 0.0
    cov = (z.T @ z) / (z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    p = ev / ev.sum()
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy))


def mae_over(pred, true, cols):
    """Mean absolute error over selected field columns. pred/true: [..., 8] numpy or tensor."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    return float(np.abs(pred[..., cols] - true[..., cols]).mean())


def constraint_violations(pred, ercp):
    """Count one-directional / bounds violations over a predicted trajectory batch.

    pred: [B, T, 8]; ercp: [B, T] bool (ERCP months). Returns (n_violations, n_steps, rate).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(ercp, torch.Tensor):
        ercp = ercp.detach().cpu().numpy()
    d = np.diff(pred, axis=1)                       # [B, T-1, 8]
    tol = 1e-5
    viol = 0
    mono_up = [F, D, P, M]
    viol += int((d[..., mono_up] < -tol).sum())
    # S may only drop at an ERCP month (aligned to the target month t+1)
    s_drop = d[..., S] < -tol                        # [B, T-1]
    ercp_next = ercp[:, 1:]                          # target-month ERCP flag
    viol += int((s_drop & (~ercp_next.astype(bool))).sum())
    fmax = np.array(FIELD_MAX)
    viol += int((pred < -tol).sum())
    viol += int((pred > fmax + tol).sum())
    n_steps = d[..., mono_up].size + s_drop.size + pred.size + pred.size
    return viol, n_steps, viol / max(n_steps, 1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
