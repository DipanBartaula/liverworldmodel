"""
Neural-PDE world models: the latent is treated as a 1-D field and evolved by a learned
REACTION-DIFFUSION PDE (method-of-lines, integrated in time with forward Euler):

    dz/dt = softplus(D) * Laplacian(z)  +  f_theta(z, ctx, t [, z_task])
            [ diffusion term ]              [ reaction term ]

The discrete Laplacian (second difference across latent dims, Neumann boundary) is the spatial
operator that makes this a PDE rather than a plain Neural-ODE; per-dim diffusion coefficients D
are learned. Euler stepping keeps the KAN reaction affordable. Decoding each month through the
ConstraintHead keeps the ratchet valid.

  kan_pde              : KAN reaction term, standard state encoder.
  leworld_pde          : the LeWorld/JEPA state latent is the field the PDE evolves (JEPA state
                         encoder + MLP reaction).
  leworld_pde_meta     : + meta test-time gradient search (a per-patient task code z_task
                         conditions the reaction and is found by inner gradient steps on the
                         held-in tail of the observed window; meta-trained so the PDE is
                         steerable in a few test-time steps -- better on OOD patients).
  gnn_leworld_meta_pde : + a GraphRefine message pass over the patients in a batch, so the
                         initial PDE latent borrows dynamics from similar patients (the GNN
                         ingredient that tied for SOTA in gnn_le_meta), on top of the meta PDE.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.neural_sde import _KANNet
from models.composed import GraphRefine
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss, scale_dim


class _LatentPDE(BaseModel):
    def __init__(self, T, dz=24, hid=32, dtask=8, drift_kind="kan", leworld_enc=False,
                 use_task=False, use_diffusion=True, val_window=6, inner_steps=2, inner_lr=0.2,
                 grid=5, order=3, tdim=16, scale=1.0):
        super().__init__(T)
        dz, hid, dtask = scale_dim(dz, scale), scale_dim(hid, scale), scale_dim(dtask, scale)
        self.dz, self.dtask, self.leworld_enc, self.use_task = dz, dtask, leworld_enc, use_task
        self.use_diffusion = use_diffusion
        self.val_window, self.inner_steps, self.inner_lr = val_window, inner_steps, inner_lr
        if leworld_enc:
            self.state_enc = mlp([STATE_DIM, hid, dz])
            self.enc_gru = nn.GRU(dz + CTX_DIM, dz, batch_first=True)
        else:
            self.enc_gru = nn.GRU(STATE_DIM + CTX_DIM, dz, batch_first=True)
        self.temb = SinusoidalPosEmb(tdim)
        din = dz + CTX_DIM + tdim + (dtask if use_task else 0)
        self.reaction = _KANNet(din, hid, dz, grid, order) if drift_kind == "kan" \
            else mlp([din, hid, hid, dz])
        self.logD = nn.Parameter(torch.zeros(dz))               # per-dim diffusion coefficient
        self.dec = mlp([dz, hid, RAW_DIM])

    def _encode(self, x_obs, ctx_obs):
        seq = torch.cat([self.state_enc(x_obs), ctx_obs], -1) if self.leworld_enc \
            else torch.cat([x_obs, ctx_obs], -1)
        _, hN = self.enc_gru(seq)
        return hN[-1]

    def _laplacian(self, z):
        zl = torch.cat([z[:, :1], z[:, :-1]], dim=1)
        zr = torch.cat([z[:, 1:], z[:, -1:]], dim=1)
        return zl - 2 * z + zr                                  # Neumann boundary

    def _field(self, z, ctx_t, t, z_task=None):
        te = self.temb(torch.full((z.shape[0],), float(t), device=z.device))
        parts = [z, ctx_t, te] + ([z_task] if self.use_task else [])
        react = self.reaction(torch.cat(parts, -1))                 # reaction / ODE drift
        if not self.use_diffusion:
            return react                                            # plain Neural-ODE (no diffusion)
        diff = F.softplus(self.logD) * self._laplacian(z)
        return diff + react

    def _integrate(self, z0, x_base, ctx, ercp, K, end, z_task=None):
        z, prev = z0, x_base
        outs = []
        for t in range(K, end - 1):
            z = z + self._field(z, ctx[:, t + 1], t, z_task)    # forward Euler (dt=1 month)
            prev = self.head(self.dec(z), prev, ercp[:, t + 1])
            outs.append(prev)
        return torch.stack(outs, 1)

    # -- meta test-time search (only when use_task) --------------------------------------
    def _support_loss(self, x, ctx, ercp, z_task, Ks, K):
        z0 = self._encode(x[:, : Ks + 1], ctx[:, : Ks + 1])
        s = self._integrate(z0, x[:, Ks], ctx, ercp, Ks, K + 1, z_task)
        return state_loss(s, x[:, Ks + 1:K + 1])

    def _adapt(self, x, ctx, ercp, Ks, K, create_graph):
        z_task = torch.zeros(x.shape[0], self.dtask, device=x.device, requires_grad=True)
        for _ in range(self.inner_steps):
            loss = self._support_loss(x, ctx, ercp, z_task, Ks, K)
            (g,) = torch.autograd.grad(loss, z_task, create_graph=create_graph)
            z_task = z_task - self.inner_lr * g
        return z_task

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]; V = self.val_window
        K = int(torch.randint(V + 4, T // 2 + 1, (1,)).item())
        z_task = None
        if self.use_task:
            z_task = self._adapt(x, ctx, ercp, K - V, K, create_graph=True)   # inner search
        z0 = self._encode(x[:, : K + 1], ctx[:, : K + 1])
        preds = self._integrate(z0, x[:, K], ctx, ercp, K, T, z_task)         # outer: future
        return {"loss": state_loss(preds, x[:, K + 1:])}

    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        z_task = torch.zeros(x.shape[0], self.dtask, device=x.device) if self.use_task else None
        if self.use_task and K >= 3:
            V = min(self.val_window, K - 1)
            with torch.enable_grad():
                z_task = self._adapt(x, ctx, ercp, max(1, K - V), K, create_graph=False).detach()
        with torch.no_grad():
            z0 = self._encode(x[:, : K + 1], ctx[:, : K + 1])
            pred[:, K + 1:] = self._integrate(z0, x[:, K], ctx, ercp, K, x.shape[1], z_task)
        return pred

    def latent(self, batch, anchors=(12, 18, 24)):
        """Encoder latent z0 across patients at several observation lengths -> [N*len(anchors), dz],
        for effective-rank collapse diagnostics (rank -> 1 = collapsed encoder). Measures ENCODER
        diversity (comparable to the LeWorld/TS-JEPA collapse metric), not the evolved trajectory --
        the latter would conflate representation collapse with the disease's dynamical convergence
        to late-stage attractors."""
        x, ctx = batch["x"], batch["ctx"]
        with torch.no_grad():
            zs = [self._encode(x[:, : k + 1], ctx[:, : k + 1]) for k in anchors]
        return torch.stack(zs, 1).reshape(-1, self.dz)


class KANNeuralPDE(_LatentPDE):
    name = "kan_pde"
    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="kan", leworld_enc=False, scale=scale)


class LeWorldPDE(_LatentPDE):
    name = "leworld_pde"
    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="mlp", leworld_enc=True, scale=scale)


class LeWorldPDEMeta(_LatentPDE):
    name = "leworld_pde_meta"
    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="mlp", leworld_enc=True, use_task=True, scale=scale)


class GNNLeWorldMetaPDE(_LatentPDE):
    name = "gnn_leworld_meta_pde"
    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="mlp", leworld_enc=True, use_task=True, scale=scale)
        self.graph = GraphRefine(self.dz)                       # patient-graph refine of z0

    def _encode(self, x_obs, ctx_obs):
        return self.graph(_LatentPDE._encode(self, x_obs, ctx_obs))


class GNNLeWorldMetaODE(_LatentPDE):
    """LeWorld + Neural-ODE hybrid: a GNN-refined, meta-adapted LeWorld/JEPA state latent is the
    code evolved by a learned Neural-ODE (drift-only field, forward Euler) -- no diffusion term,
    so it is a continuous-time ODE on the latent rather than a reaction-diffusion PDE."""
    name = "gnn_leworld_meta_ode"
    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="mlp", leworld_enc=True, use_task=True,
                         use_diffusion=False, scale=scale)
        self.graph = GraphRefine(self.dz)                       # patient-graph refine of z0

    def _encode(self, x_obs, ctx_obs):
        return self.graph(_LatentPDE._encode(self, x_obs, ctx_obs))
