"""
Neural-SDE world models -- a stochastic generalisation of the Neural-ODE model, in four
configurations the user requested. Shared base `_LatentSDE`; the variants flip flags.

The latent evolves as an SDE integrated with Euler-Maruyama (dt=1 month):
    z_{t+1} = z_t + f_theta(z_t, ctx, t [, z_task]) + softplus(g_theta(z_t)) * eps,   eps~N(0,I)
and the emission is a heteroscedastic Gaussian in state space (mean via the ConstraintHead, so
constraints hold; per-field log-variance head), trained by weighted Gaussian NLL. So it is a
"neural stochastic (Gaussian) process": a learned stochastic transition + calibrated Gaussian
emission. At eval we roll the MEAN path (eps=0) for a point MAE comparable to the other models;
the stochastic structure lives in the emission variance / sampled paths.

Variants
  neural_sde          : MLP drift, no task code            (stochastic replacement for Neural-ODE)
  sde_kan_mh          : KAN drift, multi-horizon decode    (SDE + KAN)
  sde_kan_mh_meta     : + meta test-time gradient search, OOD/beyond-horizon-simulated episodes
  sde_leworld_latent  : SDE evolves the LeWorld (JEPA) state latent, + meta + OOD/beyond-horizon
Time-conditioning (sinusoidal absolute month) makes every decode horizon-aware -> multi-horizon.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.kan import KANLayer
from util import (STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss,
                  scale_dim, FIELD_WEIGHTS)


class _KANNet(nn.Module):
    def __init__(self, din, hid, dout, grid=5, order=3):
        super().__init__()
        self.norm = nn.LayerNorm(din)
        self.k1 = KANLayer(din, hid, grid, order)
        self.k2 = KANLayer(hid, dout, grid, order)

    def forward(self, x):
        return self.k2(F.silu(self.k1(self.norm(x))))


class _LatentSDE(BaseModel):
    def __init__(self, T, dz=24, hid=32, dtask=8, drift_kind="mlp", use_task=False,
                 leworld_enc=False, ood=False, grid=5, order=3, tdim=16,
                 val_window=6, inner_steps=2, inner_lr=0.2, p_ood=0.4, scale=1.0):
        super().__init__(T)
        dz, hid, dtask = scale_dim(dz, scale), scale_dim(hid, scale), scale_dim(dtask, scale)
        self.dz, self.dtask = dz, dtask
        self.use_task, self.ood, self.leworld_enc = use_task, ood, leworld_enc
        self.val_window, self.inner_steps, self.inner_lr, self.p_ood = \
            val_window, inner_steps, inner_lr, p_ood
        if leworld_enc:
            self.state_enc = mlp([STATE_DIM, hid, dz])            # LeWorld/JEPA state latent
            self.enc_gru = nn.GRU(dz + CTX_DIM, dz, batch_first=True)
        else:
            self.enc_gru = nn.GRU(STATE_DIM + CTX_DIM, dz, batch_first=True)
        self.temb = SinusoidalPosEmb(tdim)
        din = dz + CTX_DIM + tdim + (dtask if use_task else 0)
        Net = (lambda i, o: _KANNet(i, hid, o, grid, order)) if drift_kind == "kan" \
            else (lambda i, o: mlp([i, hid, hid, o]))
        self.drift = Net(din, dz)
        self.diffusion = mlp([dz, hid, dz])                       # latent log-noise
        self.dec = mlp([dz, hid, RAW_DIM])
        self.logvar = mlp([dz, hid, STATE_DIM])                   # heteroscedastic emission var

    def _encode(self, x_obs, ctx_obs):
        if self.leworld_enc:
            seq = torch.cat([self.state_enc(x_obs), ctx_obs], -1)
        else:
            seq = torch.cat([x_obs, ctx_obs], -1)
        _, hN = self.enc_gru(seq)
        return hN[-1]

    def _drift(self, z, ctx_t, t, z_task):
        te = self.temb(torch.full((z.shape[0],), float(t), device=z.device))
        parts = [z, ctx_t, te] + ([z_task] if self.use_task else [])
        return self.drift(torch.cat(parts, -1))

    def _integrate(self, z0, x_base, ctx, ercp, K, end, z_task, noise):
        z, prev = z0, x_base
        states, lvs = [], []
        for t in range(K, end - 1):
            z = z + self._drift(z, ctx[:, t + 1], t, z_task)
            if noise:
                z = z + F.softplus(self.diffusion(z)) * 0.1 * torch.randn_like(z)
            prev = self.head(self.dec(z), prev, ercp[:, t + 1])
            states.append(prev); lvs.append(self.logvar(z))
        return torch.stack(states, 1), torch.stack(lvs, 1)

    # -- meta test-time search (only when use_task) --------------------------------------
    def _support_loss(self, x, ctx, ercp, z_task, Ks, K):
        z0 = self._encode(x[:, : Ks + 1], ctx[:, : Ks + 1])
        s, _ = self._integrate(z0, x[:, Ks], ctx, ercp, Ks, K + 1, z_task, noise=False)
        return state_loss(s, x[:, Ks + 1:K + 1])

    def _adapt(self, x, ctx, ercp, Ks, K, create_graph):
        z_task = torch.zeros(x.shape[0], self.dtask, device=x.device, requires_grad=True)
        for _ in range(self.inner_steps):
            loss = self._support_loss(x, ctx, ercp, z_task, Ks, K)
            (g,) = torch.autograd.grad(loss, z_task, create_graph=create_graph)
            z_task = z_task - self.inner_lr * g
        return z_task

    def _nll(self, states, lvs, target):
        se = (states - target) ** 2
        nll = 0.5 * (se / lvs.exp().clamp(min=1e-4) + lvs)
        return (FIELD_WEIGHTS.to(states.device) * nll).mean()

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]; V = self.val_window
        if self.ood and torch.rand(1).item() < self.p_ood:
            K = int(torch.randint(V + 2, V + 8, (1,)).item())     # short obs -> long/beyond query
            x = x.clone()
            x[:, : K + 1] = (x[:, : K + 1] + 0.03 * torch.randn_like(x[:, : K + 1])).clamp(0, 2)
        else:
            K = int(torch.randint(V + 4, T // 2 + 1, (1,)).item())
        z_task = None
        if self.use_task:
            z_task = self._adapt(x, ctx, ercp, K - V, K, create_graph=True)
        z0 = self._encode(x[:, : K + 1], ctx[:, : K + 1])
        states, lvs = self._integrate(z0, x[:, K], ctx, ercp, K, T, z_task, noise=True)
        return {"loss": self._nll(states, lvs, x[:, K + 1:])}

    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        z_task = torch.zeros(x.shape[0], self.dtask, device=x.device) if self.use_task else None
        if self.use_task and K >= 3:
            V = min(self.val_window, K - 1)
            with torch.enable_grad():
                z_task = self._adapt(x, ctx, ercp, max(1, K - V), K, create_graph=False).detach()
        with torch.no_grad():
            z0 = self._encode(x[:, : K + 1], ctx[:, : K + 1])
            states, _ = self._integrate(z0, x[:, K], ctx, ercp, K, x.shape[1], z_task, noise=False)
            pred[:, K + 1:] = states
        return pred


class NeuralSDE(_LatentSDE):
    name = "neural_sde"
    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="mlp", use_task=False, leworld_enc=False, ood=False, scale=scale)


class SDEKANMultiHorizon(_LatentSDE):
    name = "sde_kan_mh"
    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="kan", use_task=False, leworld_enc=False, ood=False, scale=scale)


class SDEKANMultiHorizonMeta(_LatentSDE):
    name = "sde_kan_mh_meta"
    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="kan", use_task=True, leworld_enc=False, ood=True, scale=scale)


class SDELeWorldLatent(_LatentSDE):
    name = "sde_leworld_latent"
    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="kan", use_task=True, leworld_enc=True, ood=True, scale=scale)
