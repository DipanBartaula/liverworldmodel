"""
TTTTransformer -- a Transformer that does Test-Time Training to cope with OOD / non-stationary
patients.

Architecturally it is a causal next-state Transformer. The twist is at INFERENCE: before
forecasting a patient, it clones itself and takes a few self-supervised gradient steps on THAT
patient's observed window (predict each observed next-state from its past). This adapts the
weights to the patient's current regime -- a faster progressor, an unusual treatment response --
which a frozen model cannot do. The adapted clone then rolls the future out; the trained model
is never mutated.
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, state_loss, scale_dim


class TTTTransformer(BaseModel):
    name = "ttt_transformer"

    def __init__(self, T, d_model=64, nhead=4, nlayers=2, ttt_steps=8, ttt_lr=1e-2, scale=1.0):
        super().__init__(T)
        self.ttt_steps, self.ttt_lr = ttt_steps, ttt_lr
        d_model = scale_dim(d_model, scale, nhead)
        self.inp = nn.Linear(STATE_DIM + CTX_DIM, d_model)
        self.pos = SinusoidalPosEmb(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.out = nn.Linear(d_model, RAW_DIM)

    def _raw(self, states, ctx_tgt):
        L = states.shape[1]
        h = self.inp(torch.cat([states, ctx_tgt], -1))
        h = h + self.pos(torch.arange(L, device=states.device)).unsqueeze(0)
        mask = torch.triu(torch.ones(L, L, device=states.device, dtype=torch.bool), 1)
        return self.out(self.tr(h, mask=mask))

    def _next_step_loss(self, x, ctx, ercp, upto):
        """Self-supervised loss on observed months only: predict x[1..upto] from x[0..upto-1]."""
        raw = self._raw(x[:, :upto], ctx[:, 1: upto + 1])
        pred = self.head(raw, x[:, :upto], ercp[:, 1: upto + 1])
        return state_loss(pred, x[:, 1: upto + 1])

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        return {"loss": self._next_step_loss(x, ctx, ercp, x.shape[1] - 1)}

    def _ar_rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        T = x.shape[1]
        for t in range(K, T - 1):
            raw = self._raw(pred[:, : t + 1], ctx[:, 1: t + 2])[:, -1]
            pred[:, t + 1] = self.head(raw, pred[:, t], ercp[:, t + 1])
        return pred

    def rollout(self, x, ctx, ercp, K):
        clone = copy.deepcopy(self)
        clone.train()
        opt = torch.optim.SGD(clone.parameters(), lr=self.ttt_lr, momentum=0.9)
        with torch.enable_grad():                              # callers wrap rollout in no_grad
            for _ in range(self.ttt_steps):                    # adapt on the observed window
                opt.zero_grad()
                loss = clone._next_step_loss(x, ctx, ercp, K)
                loss.backward()
                opt.step()
        clone.eval()
        with torch.no_grad():
            return clone._ar_rollout(x, ctx, ercp, K)
