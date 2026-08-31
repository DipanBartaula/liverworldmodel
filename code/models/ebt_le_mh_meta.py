"""
EBTLeWorldMHMeta -- a LeWorld world model whose PREDICTOR is an Energy-Based Transformer,
doing multi-horizon prediction conditioned on a horizon embedding, meta-trained with
OOD / beyond-horizon-simulated episodes.

Pipeline: a GRU encodes the observed window into a context c (the LeWorld latent). For each
horizon h we form a conditioning vector [c, horizon-emb(h)+sinusoidal(h), z_task, ctx]. An
ENERGY network E(cond, u) scores a candidate raw u; the prediction is obtained by System-2
gradient descent on u (u <- u - alpha*dE/du), then passed through the ConstraintHead. All
horizons are optimised in parallel and chained through the head (monotone across horizons).

Two nested forms of test-time gradient search:
  * the EBT energy descent (inference-time optimisation of the prediction itself), and
  * a meta task code z_task found by gradient descent on a support split (first-order MAML),
which is trained on OOD/beyond-horizon-simulated episodes (short support, long/extrapolated
query, noisy observations) so the energy landscape + adaptation transfer to shifted patients.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss, scale_dim


class EBTLeWorldMHMeta(BaseModel):
    name = "ebt_le_mh_meta"

    def __init__(self, T, dz=32, dtask=8, hid=48, energy_steps=2, alpha=0.5,
                 val_window=6, task_steps=1, task_lr=0.2, p_ood=0.4, max_h=None, scale=1.0):
        super().__init__(T)
        dz, dtask, hid = scale_dim(dz, scale), scale_dim(dtask, scale), scale_dim(hid, scale)
        self.dtask, self.val_window = dtask, val_window
        self.energy_steps, self.alpha = energy_steps, alpha
        self.task_steps, self.task_lr, self.p_ood = task_steps, task_lr, p_ood
        self.max_h = max_h or T
        self.enc = nn.GRU(STATE_DIM + CTX_DIM, dz, batch_first=True)
        self.h_emb = nn.Embedding(self.max_h + 1, dz)
        self.h_pos = SinusoidalPosEmb(dz)
        self.cond_proj = nn.Linear(dz + dz + dtask + CTX_DIM, hid)
        self.cand = nn.Linear(RAW_DIM, hid)
        self.energy = mlp([2 * hid, hid, 1])

    def _context(self, x_obs, ctx_obs):
        _, hN = self.enc(torch.cat([x_obs, ctx_obs], -1))
        return hN[-1]

    def _cond(self, c, z_task, ctx, base, horizons):
        q = self.h_emb(horizons.clamp(max=self.max_h)) + self.h_pos(horizons)   # [H, dz]
        B, H = c.shape[0], horizons.shape[0]
        cond = torch.cat([c.unsqueeze(1).expand(B, H, -1),
                          q.unsqueeze(0).expand(B, H, -1),
                          z_task.unsqueeze(1).expand(B, H, -1),
                          ctx[:, base + horizons]], -1)
        return self.cond_proj(cond)                                             # [B, H, hid]

    def _descend(self, cond_h, create_graph):
        u = torch.zeros(*cond_h.shape[:-1], RAW_DIM, device=cond_h.device, requires_grad=True)
        for _ in range(self.energy_steps):
            E = self.energy(torch.cat([cond_h, self.cand(u)], -1)).squeeze(-1).sum()
            (g,) = torch.autograd.grad(E, u, create_graph=create_graph)
            u = u - self.alpha * g
        return u

    def _predict_block(self, x, ctx, ercp, z_task, base, end, create_graph):
        c = self._context(x[:, : base + 1], ctx[:, : base + 1])
        horizons = torch.arange(1, end - base, device=x.device)
        cond_h = self._cond(c, z_task, ctx, base, horizons)
        u = self._descend(cond_h, create_graph)
        prev = x[:, base]
        outs = []
        for i, h in enumerate(horizons.tolist()):
            prev = self.head(u[:, i], prev, ercp[:, base + h])
            outs.append(prev)
        return torch.stack(outs, 1)

    def _adapt(self, x, ctx, ercp, Ks, K):
        z_task = torch.zeros(x.shape[0], self.dtask, device=x.device, requires_grad=True)
        for _ in range(self.task_steps):                       # first-order meta on support
            s = self._predict_block(x, ctx, ercp, z_task, Ks, K + 1, create_graph=True)
            loss = state_loss(s, x[:, Ks + 1:K + 1])
            (g,) = torch.autograd.grad(loss, z_task, create_graph=False)
            z_task = (z_task - self.task_lr * g).detach().requires_grad_(True)
        return z_task.detach()

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]; V = self.val_window
        if torch.rand(1).item() < self.p_ood:
            K = int(torch.randint(V + 2, V + 8, (1,)).item())  # short support -> long/beyond query
            x = x.clone()
            x[:, : K + 1] = (x[:, : K + 1] + 0.03 * torch.randn_like(x[:, : K + 1])).clamp(0, 2)
        else:
            K = int(torch.randint(V + 4, T // 2 + 1, (1,)).item())
        z_task = self._adapt(x, ctx, ercp, K - V, K)
        preds = self._predict_block(x, ctx, ercp, z_task, K, T, create_graph=True)
        return {"loss": state_loss(preds, x[:, K + 1:])}

    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        with torch.enable_grad():
            if K >= 3:
                V = min(self.val_window, K - 1)
                z_task = self._adapt(x, ctx, ercp, max(1, K - V), K)
            else:
                z_task = torch.zeros(x.shape[0], self.dtask, device=x.device)
            states = self._predict_block(x, ctx, ercp, z_task, K, x.shape[1], create_graph=False)
        pred[:, K + 1:] = states.detach()
        return pred
