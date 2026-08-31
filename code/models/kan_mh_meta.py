"""
KANMultiHorizonMeta -- a Kolmogorov-Arnold-Network world model that does multi-horizon
prediction from an embedding, is trained to perform gradient-based test-time search (meta-
learning), and is meta-trained on SIMULATED-OOD episodes so the test-time adaptation transfers
to out-of-distribution / out-of-horizon patients.

Three ideas fused:
  1. KAN predictor. The learned dynamics function (embedding + horizon query + task code -> raw
     increment) is a stack of KAN layers, so every input->output edge is an inspectable 1-D
     spline (interpretability).
  2. Multi-horizon from an embedding. A GRU encodes the observed window into a context vector;
     for each horizon h a query = learned horizon embedding + sinusoidal(h). All horizons are
     predicted at once and chained through the ConstraintHead, so the ratchet holds across
     horizons by construction and there is no autoregressive error accumulation.
  3. Meta-learned test-time gradient search, OOD-simulated. A per-patient task code z_task is
     found at inference by gradient descent on a support split (predict the observed tail). The
     model is meta-trained for this (inner-adapt on support, outer loss on the true future,
     backprop through the inner search), and crucially each training episode SIMULATES OOD:
     with some probability the episode uses a very short support / very long query (out-of-
     horizon) and injects observation noise into the support, so the inner gradient search is
     explicitly optimised to cope with shift rather than a fixed easy split.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.kan import KANLayer
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, state_loss, scale_dim


class KANMultiHorizonMeta(BaseModel):
    name = "kan_mh_meta"

    def __init__(self, T, dz=16, dtask=8, hid=24, grid=5, order=3, val_window=6,
                 inner_steps=2, inner_lr=0.2, p_ood=0.4, max_h=None, scale=1.0):
        super().__init__(T)
        dz, dtask, hid = scale_dim(dz, scale), scale_dim(dtask, scale), scale_dim(hid, scale)
        self.dtask, self.val_window = dtask, val_window
        self.inner_steps, self.inner_lr, self.p_ood = inner_steps, inner_lr, p_ood
        self.max_h = max_h or T
        self.enc = nn.GRU(STATE_DIM + CTX_DIM, dz, batch_first=True)
        self.h_emb = nn.Embedding(self.max_h + 1, dz)         # learned horizon embedding
        self.h_pos = SinusoidalPosEmb(dz)                     # sinusoidal horizon embedding
        din = dz + dz + dtask + CTX_DIM
        self.norm = nn.LayerNorm(din)                         # keep inputs inside the spline grid
        self.kan1 = KANLayer(din, hid, grid, order)
        self.kan2 = KANLayer(hid, RAW_DIM, grid, order)

    def _context(self, x_obs, ctx_obs):
        _, hN = self.enc(torch.cat([x_obs, ctx_obs], -1))
        return hN[-1]

    def _horizon_raws(self, c, z_task, ctx, base, horizons):
        h_idx = horizons.clamp(max=self.max_h)
        q = self.h_emb(h_idx) + self.h_pos(horizons)          # [H, dz]
        B, H = c.shape[0], horizons.shape[0]
        inp = torch.cat([c.unsqueeze(1).expand(B, H, -1),
                         q.unsqueeze(0).expand(B, H, -1),
                         z_task.unsqueeze(1).expand(B, H, -1),
                         ctx[:, base + horizons]], -1)
        z = self.norm(inp)
        return self.kan2(F.silu(self.kan1(z)))                # [B, H, RAW]

    def _predict_block(self, x, ctx, ercp, z_task, base, end):
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
        # OOD-simulated episodes: sometimes a very short support / very long query + noisy support
        if torch.rand(1).item() < self.p_ood:
            K = int(torch.randint(V + 2, V + 8, (1,)).item())          # short obs -> long extrapolation
            x = x.clone()
            x[:, : K + 1] = (x[:, : K + 1] + 0.03 * torch.randn_like(x[:, : K + 1])).clamp(0, 2)
        else:
            K = int(torch.randint(V + 4, T // 2 + 1, (1,)).item())
        Ks = K - V
        z_task = self._adapt(x, ctx, ercp, Ks, K, create_graph=True)   # inner test-time search
        preds = self._predict_block(x, ctx, ercp, z_task, K, T)        # outer: true future
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
