"""
Four composed world models (each a stack of building blocks already validated in the zoo):

  gnn_le_meta   : GNN-over-patients context + LeWorld multi-horizon predictor + meta test-time
                  gradient search (MultiHorizonMeta with a patient-graph-refined context).
  gnn_sde       : GNN-over-patients embedding as the initial latent of a Neural-SDE.
  kan_gnn_meta  : KAN multi-horizon meta predictor on a GNN-refined patient context.
  ebt_jepa_sde  : an EBT-refined JEPA latent (energy-descent decode) evolved by a Neural-SDE
                  (leworld/JEPA state encoder -> SDE latent -> energy-based decode -> head).

`GraphRefine` is a cosine self-attention message pass over the patients in a batch: a patient's
representation is updated with a similarity-weighted sum of the others, so forecasts can borrow
dynamics from similar patients (transductive within the batch).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neural_sde import _LatentSDE
from models.multihorizon_meta_le_world_model import MultiHorizonMetaLeWorldModel
from models.kan_mh_meta import KANMultiHorizonMeta
from util import RAW_DIM, mlp


class GraphRefine(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)

    def forward(self, e):                                     # e: [B, d]
        q = F.normalize(self.q(e), dim=-1)
        k = F.normalize(self.k(e), dim=-1)
        att = torch.softmax(q @ k.t() / (self.d ** 0.5), dim=-1)
        return e + self.o(att @ self.v(e))


class GNNLeMeta(MultiHorizonMetaLeWorldModel):
    name = "gnn_le_meta"

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale)
        self.graph = GraphRefine(self.enc_gru.hidden_size)

    def _context(self, x_obs, ctx_obs):
        return self.graph(super()._context(x_obs, ctx_obs))


class KANGNNMeta(KANMultiHorizonMeta):
    name = "kan_gnn_meta"

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale)
        self.graph = GraphRefine(self.enc.hidden_size)

    def _context(self, x_obs, ctx_obs):
        return self.graph(super()._context(x_obs, ctx_obs))


class GNNSDE(_LatentSDE):
    name = "gnn_sde"

    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="mlp", use_task=False, leworld_enc=False, ood=False, scale=scale)
        self.graph = GraphRefine(self.dz)

    def _encode(self, x_obs, ctx_obs):
        return self.graph(_LatentSDE._encode(self, x_obs, ctx_obs))


class EBTJepaSDE(_LatentSDE):
    name = "ebt_jepa_sde"

    def __init__(self, T, scale=1.0):
        super().__init__(T, drift_kind="mlp", use_task=True, leworld_enc=True, ood=True, scale=scale)
        self.ecand = nn.Linear(RAW_DIM, self.dz)
        self.energy = mlp([2 * self.dz, self.dz, 1])          # EBT energy on (latent, candidate)
        self.esteps, self.ealpha = 2, 0.5

    def _ebt_decode(self, z):
        """Energy-based decode: refine the raw candidate by gradient descent on E(z, u)."""
        with torch.enable_grad():
            u = self.dec(z)
            if not u.requires_grad:
                u = u.detach().requires_grad_(True)
            for _ in range(self.esteps):
                E = self.energy(torch.cat([z, self.ecand(u)], -1)).sum()
                (g,) = torch.autograd.grad(E, u, create_graph=self.training)
                u = u - self.ealpha * g
        return u

    def _integrate(self, z0, x_base, ctx, ercp, K, end, z_task, noise):
        z, prev = z0, x_base
        states, lvs = [], []
        for t in range(K, end - 1):
            z = z + self._drift(z, ctx[:, t + 1], t, z_task)
            if noise:
                z = z + F.softplus(self.diffusion(z)) * 0.1 * torch.randn_like(z)
            prev = self.head(self._ebt_decode(z), prev, ercp[:, t + 1])
            states.append(prev); lvs.append(self.logvar(z))
        return torch.stack(states, 1), torch.stack(lvs, 1)
