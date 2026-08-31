"""
NeuralODEWorldModel -- the LeWorldModel idea, but the latent evolves in CONTINUOUS time via a
learned ODE instead of a discrete one-step predictor.

A GRU encodes the observed window into an initial latent z(K). A learned vector field
f(z, ctx, t) defines dz/dt; we integrate it forward month-by-month with a fixed-step RK4
solver (no torchdiffeq dependency). At each integer month the latent is decoded to a raw
increment for the ConstraintHead, chained from the previous state so the ratchet holds. The
time input to f uses a sinusoidal time embedding, so the field is time-aware (treatment epochs
land at real months).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss, scale_dim


class NeuralODEWorldModel(BaseModel):
    name = "node_world_model"

    def __init__(self, T, dz=32, hid=64, t_dim=16, scale=1.0):
        super().__init__(T)
        dz, hid, t_dim = scale_dim(dz, scale), scale_dim(hid, scale), scale_dim(t_dim, scale)
        self.dz = dz
        self.enc = nn.GRU(STATE_DIM + CTX_DIM, dz, batch_first=True)
        self.temb = SinusoidalPosEmb(t_dim)
        self.f = mlp([dz + CTX_DIM + t_dim, hid, hid, dz])     # vector field dz/dt
        self.dec = mlp([dz, hid, RAW_DIM])

    def _field(self, z, ctx_t, t):
        te = self.temb(torch.full((z.shape[0],), float(t), device=z.device))
        return self.f(torch.cat([z, ctx_t, te], -1))

    def _rk4(self, z, ctx_t, t, dt=1.0):
        k1 = self._field(z, ctx_t, t)
        k2 = self._field(z + 0.5 * dt * k1, ctx_t, t + 0.5)
        k3 = self._field(z + 0.5 * dt * k2, ctx_t, t + 0.5)
        k4 = self._field(z + dt * k3, ctx_t, t + 1.0)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _integrate(self, x, ctx, ercp, K, T):
        _, hN = self.enc(torch.cat([x[:, : K + 1], ctx[:, : K + 1]], -1))
        z = hN[-1]
        prev = x[:, K]
        outs = []
        for t in range(K, T - 1):
            z = self._rk4(z, ctx[:, t + 1], t)
            prev = self.head(self.dec(z), prev, ercp[:, t + 1])
            outs.append(prev)
        return torch.stack(outs, 1)                            # [B, T-K-1, 8]

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]
        K = int(torch.randint(6, T // 2 + 1, (1,)).item())
        preds = self._integrate(x, ctx, ercp, K, T)
        return {"loss": state_loss(preds, x[:, K + 1:])}

    @torch.no_grad()
    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        pred[:, K + 1:] = self._integrate(x, ctx, ercp, K, x.shape[1])
        return pred
