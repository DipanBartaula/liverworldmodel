"""
TTTValModel -- test-time training whose inner adaptation is SELECTED by a validation loss on a
held-out slice of the observed window, and which is META-TRAINED to make that work.

At inference we split the observed months 0..K into a SUPPORT part (0..Ks) and a VALIDATION
part (Ks+1..K). We adapt a per-patient task code by gradient descent on the support next-step
loss, and after each step measure the VALIDATION next-step loss; we keep the code that
minimises validation loss (validation-based early stopping / model selection). The future is
then rolled out with the selected code.

The model is trained to do exactly this: each training step splits a patient the same way,
inner-adapts on support, and the OUTER loss is validation loss + future-rollout loss computed
with the adapted code. So (a) adaptation is trained to transfer support -> future, and
(b) validation loss is trained to be a faithful selector -- which is what makes the test-time
validation search well posed. Handles OOD / non-stationary patients by searching per patient
and trusting the validation signal rather than extrapolating a frozen model.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, mlp, state_loss, scale_dim


class TTTValModel(BaseModel):
    name = "ttt_val_model"

    def __init__(self, T, dz=32, dtask=16, hid=64, val_window=6,
                 inner_steps=6, inner_lr=0.2, scale=1.0):
        super().__init__(T)
        dz, dtask, hid = scale_dim(dz, scale), scale_dim(dtask, scale), scale_dim(hid, scale)
        self.dtask, self.val_window = dtask, val_window
        self.inner_steps, self.inner_lr = inner_steps, inner_lr
        self.enc = mlp([STATE_DIM, hid, dz])
        self.pred = mlp([dz + dtask + CTX_DIM, hid, dz])
        self.dec = mlp([dz, hid, RAW_DIM])

    def _decode(self, prev, z_task, ctx_tgt):
        z = self.enc(prev)
        zt = z_task.unsqueeze(1).expand(-1, prev.shape[1], -1) if prev.dim() == 3 else z_task
        return self.dec(self.pred(torch.cat([z, zt, ctx_tgt], -1)))

    def _range_loss(self, x, ctx, ercp, z_task, lo, hi):
        """Next-step loss over transitions month lo..hi-1 -> lo+1..hi."""
        raw = self._decode(x[:, lo:hi], z_task, ctx[:, lo + 1:hi + 1])
        pred = self.head(raw, x[:, lo:hi], ercp[:, lo + 1:hi + 1])
        return state_loss(pred, x[:, lo + 1:hi + 1])

    def _adapt(self, x, ctx, ercp, Ks, K, create_graph, select):
        """Inner gradient descent on the SUPPORT window [0..Ks]. If select, keep the code with
        the lowest VALIDATION loss on (Ks..K]; else return the final code."""
        B = x.shape[0]
        z_task = torch.zeros(B, self.dtask, device=x.device, requires_grad=True)
        best_z, best_v = z_task.detach(), None
        for _ in range(self.inner_steps):
            loss = self._range_loss(x, ctx, ercp, z_task, 0, Ks)
            (g,) = torch.autograd.grad(loss, z_task, create_graph=create_graph)
            z_task = z_task - self.inner_lr * g
            if select:
                with torch.no_grad():
                    v = self._range_loss(x, ctx, ercp, z_task, Ks, K).item()
                if best_v is None or v < best_v:
                    best_v, best_z = v, z_task.detach().clone()
        return best_z if select else z_task

    def _query_rollout(self, x, ctx, ercp, z_task, K, T):
        prev = x[:, K]
        outs = []
        for t in range(K, T - 1):
            raw = self._decode(prev, z_task, ctx[:, t + 1])
            prev = self.head(raw, prev, ercp[:, t + 1])
            outs.append(prev)
        return torch.stack(outs, 1)

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]
        V = self.val_window
        K = int(torch.randint(V + 4, T // 2 + 1, (1,)).item())
        Ks = K - V
        z_task = self._adapt(x, ctx, ercp, Ks, K, create_graph=True, select=False)
        val_loss = self._range_loss(x, ctx, ercp, z_task, Ks, K)       # make val a good selector
        fut = self._query_rollout(x, ctx, ercp, z_task, K, T)          # the real objective
        fut_loss = state_loss(fut, x[:, K + 1:])
        return {"loss": fut_loss + val_loss, "future": fut_loss.item(), "val": val_loss.item()}

    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        if K < 3:
            # too few observed months to carve a support/validation split -> no test-time search
            z_task = torch.zeros(x.shape[0], self.dtask, device=x.device)
        else:
            V = min(self.val_window, K - 1)
            Ks = max(1, K - V)
            with torch.enable_grad():
                z_task = self._adapt(x, ctx, ercp, Ks, K, create_graph=False, select=True)
        with torch.no_grad():
            pred[:, K + 1:] = self._query_rollout(x, ctx, ercp, z_task, K, x.shape[1])
        return pred
