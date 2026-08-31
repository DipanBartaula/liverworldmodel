"""
EnergyBasedTransformer (EBT) -- prediction as energy minimisation, in the spirit of the
Energy-Based Transformers paper (a learned scalar energy E(context, candidate) whose minimum
over candidates IS the prediction; you "think" by descending the energy at inference).

A causal Transformer encodes the observed history into a per-position context. An energy head
scores (context, candidate raw). To predict, we start from a candidate and take a few
GRADIENT-DESCENT steps on the energy w.r.t. the candidate (System-2 refinement); the refined
candidate is decoded through the ConstraintHead. Training unrolls those inner steps
differentiably and backprops the prediction error, so the energy landscape learns to put its
valley at the correct next state.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss, scale_dim


class EnergyBasedTransformer(BaseModel):
    name = "ebt"

    def __init__(self, T, d_model=64, nhead=4, nlayers=2, inner_steps=2, alpha=0.5, scale=1.0):
        super().__init__(T)
        self.inner_steps, self.alpha = inner_steps, alpha
        d_model = scale_dim(d_model, scale, nhead)
        self.inp = nn.Linear(STATE_DIM + CTX_DIM, d_model)
        self.pos = SinusoidalPosEmb(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.cand = nn.Linear(RAW_DIM, d_model)
        self.energy = mlp([2 * d_model, d_model, 1])

    def _context(self, states, ctx_tgt):
        L = states.shape[1]
        h = self.inp(torch.cat([states, ctx_tgt], -1))
        h = h + self.pos(torch.arange(L, device=states.device)).unsqueeze(0)
        mask = torch.triu(torch.ones(L, L, device=states.device, dtype=torch.bool), 1)
        return self.tr(h, mask=mask)

    def _energy(self, c, u):
        return self.energy(torch.cat([c, self.cand(u)], -1)).squeeze(-1)

    def _refine(self, c, create_graph):
        u = torch.zeros(*c.shape[:-1], RAW_DIM, device=c.device, requires_grad=True)
        for _ in range(self.inner_steps):
            E = self._energy(c, u).sum()
            (g,) = torch.autograd.grad(E, u, create_graph=create_graph)
            u = u - self.alpha * g
        return u

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        c = self._context(x[:, :-1], ctx[:, 1:])               # [B, T-1, d]
        with torch.enable_grad():
            u = self._refine(c, create_graph=True)
        pred = self.head(u, x[:, :-1], ercp[:, 1:])
        return {"loss": state_loss(pred, x[:, 1:])}

    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        T = x.shape[1]
        with torch.no_grad():
            for t in range(K, T - 1):
                c = self._context(pred[:, : t + 1], ctx[:, 1: t + 2])[:, -1]
                with torch.enable_grad():
                    u = self._refine(c, create_graph=False)
                pred[:, t + 1] = self.head(u.detach(), pred[:, t], ercp[:, t + 1])
        return pred
