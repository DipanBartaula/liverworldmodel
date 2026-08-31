"""
MetaLeWorldModel -- the LeWorldModel trained so that INFERENCE is a gradient-based test-time
search (a meta-learning / context-adaptation approach, in the CAVIA / latent-optimisation
family).

A per-patient task code z_task conditions the latent predictor. At inference we do not just
run a forward pass: we initialise z_task = 0 and take a few gradient-descent steps to fit the
patient's OBSERVED window (a test-time search in latent space), then roll the future out with
the found z_task. Training mirrors this: an inner loop adapts z_task on the support (observed)
months, and the OUTER loss on the query (future) months is backpropagated THROUGH the inner
search into the shared weights (first-/second-order MAML on the task code). So the network is
explicitly trained to be quickly adaptable by gradient search, and handed OOD patients it
searches rather than extrapolates.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, mlp, state_loss, scale_dim


class MetaLeWorldModel(BaseModel):
    name = "meta_le_world_model"

    def __init__(self, T, dz=32, dtask=16, hid=64, inner_steps=3, inner_lr=0.2, scale=1.0):
        super().__init__(T)
        dz, dtask, hid = scale_dim(dz, scale), scale_dim(dtask, scale), scale_dim(hid, scale)
        self.dtask, self.inner_steps, self.inner_lr = dtask, inner_steps, inner_lr
        self.enc = mlp([STATE_DIM, hid, dz])
        self.pred = mlp([dz + dtask + CTX_DIM, hid, dz])
        self.dec = mlp([dz, hid, RAW_DIM])

    def _step(self, prev, z_task, ctx_tgt):
        z = self.enc(prev)
        zt = z_task.unsqueeze(1).expand(-1, prev.shape[1], -1) if prev.dim() == 3 \
            else z_task
        znext = self.pred(torch.cat([z, zt, ctx_tgt], -1))
        return self.dec(znext)

    def _support_loss(self, x, ctx, ercp, z_task, K):
        raw = self._step(x[:, :K], z_task, ctx[:, 1: K + 1])
        pred = self.head(raw, x[:, :K], ercp[:, 1: K + 1])
        return state_loss(pred, x[:, 1: K + 1])

    def _adapt(self, x, ctx, ercp, K, create_graph):
        B = x.shape[0]
        z_task = torch.zeros(B, self.dtask, device=x.device, requires_grad=True)
        for _ in range(self.inner_steps):
            loss = self._support_loss(x, ctx, ercp, z_task, K)
            (g,) = torch.autograd.grad(loss, z_task, create_graph=create_graph)
            z_task = z_task - self.inner_lr * g
        return z_task

    def _query_rollout(self, x, ctx, ercp, z_task, K, T):
        prev = x[:, K]
        outs = []
        for t in range(K, T - 1):
            raw = self._step(prev, z_task, ctx[:, t + 1])
            prev = self.head(raw, prev, ercp[:, t + 1])
            outs.append(prev)
        return torch.stack(outs, 1)

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]
        K = int(torch.randint(6, T // 2 + 1, (1,)).item())
        z_task = self._adapt(x, ctx, ercp, K, create_graph=True)   # MAML: inner search
        preds = self._query_rollout(x, ctx, ercp, z_task, K, T)    # outer loss on future
        return {"loss": state_loss(preds, x[:, K + 1:])}

    def rollout(self, x, ctx, ercp, K):
        with torch.enable_grad():
            z_task = self._adapt(x, ctx, ercp, K, create_graph=False).detach()  # test-time search
        with torch.no_grad():
            pred = self._init_pred(x, K)
            pred[:, K + 1:] = self._query_rollout(x, ctx, ercp, z_task, K, x.shape[1])
        return pred
