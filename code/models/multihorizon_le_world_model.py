"""
MultiHorizonLeWorldModel -- the LeWorldModel trained to predict MANY horizons at once, not
just the immediate next state.

An encoder summarises the observed window into a context vector c. For each future horizon h
we form a query from a LEARNED horizon embedding (a lookup table over h) plus a fixed
SINUSOIDAL positional embedding of h, and a predictor maps (c, query) -> a raw increment for
that horizon. The per-horizon raws are chained through the ConstraintHead starting from the
last observed state, so the monotone fields are non-decreasing ACROSS horizons by construction
(each step adds a softplus increment) -- direct multi-horizon prediction that still cannot
break the ratchet.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss, scale_dim


class MultiHorizonLeWorldModel(BaseModel):
    name = "multihorizon_le_world_model"

    def __init__(self, T, dz=64, hid=96, max_h=None, scale=1.0):
        super().__init__(T)
        dz, hid = scale_dim(dz, scale), scale_dim(hid, scale)
        self.max_h = max_h or T
        self.enc_gru = nn.GRU(STATE_DIM + CTX_DIM, dz, batch_first=True)
        self.h_emb = nn.Embedding(self.max_h + 1, dz)          # learned horizon embedding
        self.h_pos = SinusoidalPosEmb(dz)                      # sinusoidal horizon embedding
        self.pred = mlp([dz + dz + CTX_DIM, hid, hid, RAW_DIM])

    def _context(self, x_obs, ctx_obs):
        _, hN = self.enc_gru(torch.cat([x_obs, ctx_obs], -1))
        return hN[-1]                                          # [B, dz]

    def _horizon_raws(self, c, ctx, K, horizons):
        # horizons: 1-D tensor of h values; returns raw for each future month K+h.
        # Learned horizon embedding is a FINITE table (max_h): clamp the lookup so rollouts past
        # the training horizon degrade gracefully instead of crashing -- this is the structural
        # ceiling of a learned-position model, surfaced (not hidden) by the longer-horizon probe.
        h_idx = horizons.clamp(max=self.max_h)
        hq = self.h_emb(h_idx) + self.h_pos(horizons)          # [H, dz]
        B = c.shape[0]
        H = horizons.shape[0]
        cexp = c.unsqueeze(1).expand(B, H, -1)
        hexp = hq.unsqueeze(0).expand(B, H, -1)
        ctx_tgt = ctx[:, K + horizons]                         # [B, H, CTX]
        return self.pred(torch.cat([cexp, hexp, ctx_tgt], -1))  # [B, H, RAW]

    def _chain(self, raws, prev, ercp, K, horizons):
        outs = []
        for i, h in enumerate(horizons.tolist()):
            prev = self.head(raws[:, i], prev, ercp[:, K + h])
            outs.append(prev)
        return torch.stack(outs, 1)                            # [B, H, 8]

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        B, T, _ = x.shape
        K = int(torch.randint(6, T // 2 + 1, (1,)).item())     # random observed window
        c = self._context(x[:, : K + 1], ctx[:, : K + 1])
        horizons = torch.arange(1, T - K, device=x.device)
        raws = self._horizon_raws(c, ctx, K, horizons)
        preds = self._chain(raws, x[:, K], ercp, K, horizons)
        return {"loss": state_loss(preds, x[:, K + 1:])}

    @torch.no_grad()
    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        c = self._context(x[:, : K + 1], ctx[:, : K + 1])
        horizons = torch.arange(1, x.shape[1] - K, device=x.device)
        raws = self._horizon_raws(c, ctx, K, horizons)
        pred[:, K + 1:] = self._chain(raws, x[:, K], ercp, K, horizons)
        return pred
