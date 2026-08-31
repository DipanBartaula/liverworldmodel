"""
MultiHorizonMetaLeWorldModel -- the multi-horizon LeWorldModel, but trained so that INFERENCE
is a gradient-based test-time search (meta-learning), combining the two ideas explicitly.

Like MultiHorizonLeWorldModel it encodes the observed window into a context c and predicts ALL
future horizons directly from learned + sinusoidal horizon embeddings, chained through the
ConstraintHead so the ratchet holds across horizons. The addition: a per-patient task code
z_task also conditions the predictor and is found at test time by GRADIENT DESCENT on a
self-supervised objective -- re-predict the held-in tail of the observed window from an earlier
anchor. The model is META-TRAINED for this: each step inner-adapts z_task on a support split of
the observed window and the OUTER loss on the future is backpropagated through that inner
search, so the multi-horizon predictor is explicitly optimised to be steerable by a few
test-time gradient steps (better on OOD patients than a single frozen forward pass).
"""

from __future__ import annotations
import torch
import torch.nn as nn

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss, scale_dim


class MultiHorizonMetaLeWorldModel(BaseModel):
    name = "multihorizon_meta_le_world_model"

    def __init__(self, T, dz=64, dtask=16, hid=96, val_window=6,
                 inner_steps=3, inner_lr=0.2, max_h=None, scale=1.0):
        super().__init__(T)
        dz, dtask, hid = scale_dim(dz, scale), scale_dim(dtask, scale), scale_dim(hid, scale)
        self.dtask, self.val_window = dtask, val_window
        self.inner_steps, self.inner_lr = inner_steps, inner_lr
        self.max_h = max_h or T
        self.enc_gru = nn.GRU(STATE_DIM + CTX_DIM, dz, batch_first=True)
        self.h_emb = nn.Embedding(self.max_h + 1, dz)          # learned horizon embedding
        self.h_pos = SinusoidalPosEmb(dz)                      # sinusoidal horizon embedding
        self.pred = mlp([dz + dz + dtask + CTX_DIM, hid, hid, RAW_DIM])

    def _context(self, x_obs, ctx_obs):
        _, hN = self.enc_gru(torch.cat([x_obs, ctx_obs], -1))
        return hN[-1]

    def _horizon_raws(self, c, z_task, ctx, base, horizons):
        h_idx = horizons.clamp(max=self.max_h)                 # learned table is finite -> clamp
        hq = self.h_emb(h_idx) + self.h_pos(horizons)          # [H, dz]
        B, H = c.shape[0], horizons.shape[0]
        cexp = c.unsqueeze(1).expand(B, H, -1)
        hexp = hq.unsqueeze(0).expand(B, H, -1)
        zexp = z_task.unsqueeze(1).expand(B, H, -1)
        ctx_tgt = ctx[:, base + horizons]
        return self.pred(torch.cat([cexp, hexp, zexp, ctx_tgt], -1))

    def _predict_block(self, x, ctx, ercp, z_task, base, end):
        """Predict months base+1..end-1 from an anchor at `base`."""
        c = self._context(x[:, : base + 1], ctx[:, : base + 1])
        horizons = torch.arange(1, end - base, device=x.device)
        raws = self._horizon_raws(c, z_task, ctx, base, horizons)
        prev = x[:, base]
        outs = []
        for i, h in enumerate(horizons.tolist()):
            prev = self.head(raws[:, i], prev, ercp[:, base + h])
            outs.append(prev)
        return torch.stack(outs, 1)

    def _support_loss(self, x, ctx, ercp, z_task, Ks, K):
        preds = self._predict_block(x, ctx, ercp, z_task, Ks, K + 1)
        return state_loss(preds, x[:, Ks + 1:K + 1])

    def _adapt(self, x, ctx, ercp, Ks, K, create_graph):
        z_task = torch.zeros(x.shape[0], self.dtask, device=x.device, requires_grad=True)
        for _ in range(self.inner_steps):
            loss = self._support_loss(x, ctx, ercp, z_task, Ks, K)
            (g,) = torch.autograd.grad(loss, z_task, create_graph=create_graph)
            z_task = z_task - self.inner_lr * g
        return z_task

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]
        V = self.val_window
        K = int(torch.randint(V + 4, T // 2 + 1, (1,)).item())
        Ks = K - V
        z_task = self._adapt(x, ctx, ercp, Ks, K, create_graph=True)     # inner test-time search
        preds = self._predict_block(x, ctx, ercp, z_task, K, T)          # outer objective: future
        return {"loss": state_loss(preds, x[:, K + 1:])}

    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        if K < 3:
            z_task = torch.zeros(x.shape[0], self.dtask, device=x.device)
        else:
            V = min(self.val_window, K - 1)
            Ks = max(1, K - V)
            with torch.enable_grad():
                z_task = self._adapt(x, ctx, ercp, Ks, K, create_graph=False).detach()
        with torch.no_grad():
            pred[:, K + 1:] = self._predict_block(x, ctx, ercp, z_task, K, x.shape[1])
        return pred
