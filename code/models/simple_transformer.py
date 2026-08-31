"""
SimpleTransformer -- a plain causal Transformer next-state predictor.

The honest, strong baseline: at each month it attends over the observed history and predicts
the next state directly (raw -> ConstraintHead). No latent, no test-time tricks. Uses fixed
sinusoidal positional embeddings. Convention: the token at step t carries (state_t, ctx_{t+1})
so the model sees the target month's treatment/ERCP context, matching the generator alignment.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, state_loss, scale_dim


class SimpleTransformer(BaseModel):
    name = "simple_transformer"

    def __init__(self, T, d_model=64, nhead=4, nlayers=2, scale=1.0):
        super().__init__(T)
        d_model = scale_dim(d_model, scale, nhead)
        self.inp = nn.Linear(STATE_DIM + CTX_DIM, d_model)
        self.pos = SinusoidalPosEmb(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                           batch_first=True, activation="gelu", dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.out = nn.Linear(d_model, RAW_DIM)

    def _raw(self, states, ctx_tgt):
        # states, ctx_tgt: [B, L, *]  token_t = (state_t, ctx_{t+1})
        B, L, _ = states.shape
        h = self.inp(torch.cat([states, ctx_tgt], -1))
        h = h + self.pos(torch.arange(L, device=states.device)).unsqueeze(0)
        mask = torch.triu(torch.ones(L, L, device=states.device, dtype=torch.bool), 1)
        h = self.tr(h, mask=mask)
        return self.out(h)

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        raw = self._raw(x[:, :-1], ctx[:, 1:])
        pred = self.head(raw, x[:, :-1], ercp[:, 1:])
        return {"loss": state_loss(pred, x[:, 1:])}

    @torch.no_grad()
    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        T = x.shape[1]
        for t in range(K, T - 1):
            raw = self._raw(pred[:, : t + 1], ctx[:, 1: t + 2])[:, -1]
            pred[:, t + 1] = self.head(raw, pred[:, t], ercp[:, t + 1])
        return pred
