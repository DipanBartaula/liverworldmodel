"""
TimeSeriesDiffusion -- a conditional DDPM that generates the future state trajectory given the
observed window.

A GRU encodes the observed months into a conditioning vector; a small Transformer denoiser
(with sinusoidal horizon positions + a sinusoidal diffusion-timestep embedding) predicts the
noise added to the future-state block. Sampling runs the reverse diffusion to produce a future
trajectory.

Constraint mechanism -- deliberately DIFFERENT from the other models: instead of the
ConstraintHead's parameterisation, the sampled trajectory is PROJECTED onto the valid set
(monotone fields forced non-decreasing from the last observed state; S allowed to drop only on
ERCP months; all fields clamped to bounds). This is the "projection" route the brief names, and
it lets us compare parameterisation-vs-projection head-to-head in eval.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import (STATE_DIM, CTX_DIM, FIELD_MAX, F as iF, D as iD, S as iS,
                  P as iP, M as iM, SinusoidalPosEmb, TimestepEmbedding, scale_dim)


def project_valid(fut, x_K, ercp_fut):
    """Project a generated future block onto the constraint set. fut:[B,Hf,8]."""
    fmax = torch.tensor(FIELD_MAX, device=fut.device)
    mono = [iF, iD, iP, iM]
    out = fut.clone()
    prev = x_K.clone()
    for t in range(out.shape[1]):
        cur = out[:, t].clamp(min=0.0)
        for j in mono:
            cur[:, j] = torch.maximum(cur[:, j], prev[:, j])         # non-decreasing
        no_ercp = ercp_fut[:, t] < 0.5
        s_floor = torch.where(no_ercp, prev[:, iS], torch.zeros_like(prev[:, iS]))
        cur[:, iS] = torch.maximum(cur[:, iS], s_floor)              # S drops only at ERCP
        cur = torch.minimum(cur, fmax)
        out[:, t] = cur
        prev = cur
    return out


class _Denoiser(nn.Module):
    def __init__(self, d, dcond, nhead=4, nlayers=2):
        super().__init__()
        self.tok = nn.Linear(STATE_DIM, d)
        self.pos = SinusoidalPosEmb(d)
        self.cond = nn.Linear(dcond, d)
        self.temb = TimestepEmbedding(d)
        layer = nn.TransformerEncoderLayer(d, nhead, d * 2, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.out = nn.Linear(d, STATE_DIM)

    def forward(self, xt, cond, t_diff, positions):
        h = self.tok(xt) + self.pos(positions).unsqueeze(0)
        h = h + self.cond(cond).unsqueeze(1) + self.temb(t_diff).unsqueeze(1)
        return self.out(self.tr(h))


class TimeSeriesDiffusion(BaseModel):
    name = "ts_diffusion"

    def __init__(self, T, d=64, dcond=48, n_diff=50, scale=1.0):
        super().__init__(T)
        d = scale_dim(d, scale, 4)                             # denoiser attn: divisible by nhead=4
        dcond = scale_dim(dcond, scale)
        self.n_diff = n_diff
        self.enc = nn.GRU(STATE_DIM + CTX_DIM, dcond, batch_first=True)
        self.net = _Denoiser(d, dcond)
        betas = torch.linspace(1e-4, 0.02, n_diff)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)

    def _cond(self, x_obs, ctx_obs):
        _, hN = self.enc(torch.cat([x_obs, ctx_obs], -1))
        return hN[-1]

    def training_step(self, batch):
        x, ctx = batch["x"], batch["ctx"]
        T = x.shape[1]
        K = int(torch.randint(6, T // 2 + 1, (1,)).item())
        fut = x[:, K + 1:]
        B, Hf, _ = fut.shape
        cond = self._cond(x[:, : K + 1], ctx[:, : K + 1])
        t = torch.randint(0, self.n_diff, (B,), device=x.device)
        eps = torch.randn_like(fut)
        abar = self.abar[t].view(B, 1, 1)
        xt = abar.sqrt() * fut + (1 - abar).sqrt() * eps
        positions = torch.arange(Hf, device=x.device)
        pred = self.net(xt, cond, t, positions)
        return {"loss": F.mse_loss(pred, eps)}

    @torch.no_grad()
    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        B = x.shape[0]
        Hf = x.shape[1] - K - 1                                 # sinusoidal positions -> any horizon
        cond = self._cond(x[:, : K + 1], ctx[:, : K + 1])
        positions = torch.arange(Hf, device=x.device)
        xt = torch.randn(B, Hf, STATE_DIM, device=x.device)
        for i in reversed(range(self.n_diff)):
            t = torch.full((B,), i, device=x.device, dtype=torch.long)
            eps = self.net(xt, cond, t, positions)
            a = self.alphas[i]
            abar = self.abar[i]
            mean = (xt - (self.betas[i] / (1 - abar).sqrt()) * eps) / a.sqrt()
            if i > 0:
                xt = mean + self.betas[i].sqrt() * torch.randn_like(xt)
            else:
                xt = mean
        pred[:, K + 1:] = project_valid(xt, x[:, K], ercp[:, K + 1:])
        return pred
