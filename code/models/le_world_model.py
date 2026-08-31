"""
LeWorldModel -- a LeCun/JEPA-style world model on the 8-D clinical state.

The core JEPA idea: predict the REPRESENTATION of the next state, not its raw values. An
encoder maps a state to a latent z; a predictor rolls z forward one step in latent space
(conditioned on the target-month context); a decoder maps the predicted latent back to a raw
9-vector that the ConstraintHead turns into a valid next state.

Two things keep the latent honest:
  * a JEPA latent-prediction loss against a stop-gradient target encoding (predict the
    representation of x_{t+1}); and
  * a VICReg-style variance+covariance regulariser that is the deliberate anti-collapse
    mechanism -- effective_rank of z would catch collapse if it crept in (see eval.py).
The decoder is trained on BOTH the predicted latent and the (stop-grad) target latent, so it
stays anchored to the same space the invariance loss drags predictions into.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, mlp, state_loss, scale_dim


def vicreg(z, gamma=1.0, eps=1e-4):
    """Variance + covariance terms (the anti-collapse regulariser). z: [N, d]."""
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0) + eps)
    var_loss = F.relu(gamma - std).mean()               # push each dim's std up to gamma
    N, d = z.shape
    cov = (z.T @ z) / max(N - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / d                     # decorrelate dims
    return var_loss + cov_loss


class LeWorldModel(BaseModel):
    name = "le_world_model"

    def __init__(self, T, dz=32, hid=64, beta_latent=1.0, beta_vic=1.0, scale=1.0):
        super().__init__(T)
        dz, hid = scale_dim(dz, scale), scale_dim(hid, scale)
        self.dz = dz
        self.beta_latent, self.beta_vic = beta_latent, beta_vic
        self.enc = mlp([STATE_DIM, hid, dz])
        self.pred = mlp([dz + CTX_DIM, hid, dz])
        self.dec = mlp([dz, hid, RAW_DIM])

    def _step_latent(self, z, ctx_tgt):
        return self.pred(torch.cat([z, ctx_tgt], -1))

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        cur, nxt = x[:, :-1], x[:, 1:]                  # [B, T-1, 8]
        ctx_tgt, ercp_tgt = ctx[:, 1:], ercp[:, 1:]
        z = self.enc(cur)
        z_pred = self._step_latent(z, ctx_tgt)
        z_tar = self.enc(nxt).detach()                  # stop-grad target representation
        loss_latent = F.mse_loss(z_pred, z_tar)
        # decode BOTH predicted and target latents to the true next state (anchor the decoder)
        pred_from_pred = self.head(self.dec(z_pred), cur, ercp_tgt)
        pred_from_tar = self.head(self.dec(z_tar), cur, ercp_tgt)
        loss_state = state_loss(pred_from_pred, nxt) + state_loss(pred_from_tar, nxt)
        loss_vic = vicreg(z.reshape(-1, self.dz))
        loss = loss_state + self.beta_latent * loss_latent + self.beta_vic * loss_vic
        return {"loss": loss, "state": loss_state.item(), "latent": loss_latent.item(),
                "vic": loss_vic.item()}

    @torch.no_grad()
    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        T = x.shape[1]
        for t in range(K, T - 1):
            z = self.enc(pred[:, t])
            z_pred = self._step_latent(z, ctx[:, t + 1])
            pred[:, t + 1] = self.head(self.dec(z_pred), pred[:, t], ercp[:, t + 1])
        return pred

    @torch.no_grad()
    def latent(self, batch):
        """Encoder outputs over a batch -> [N, dz], for effective-rank collapse diagnostics."""
        return self.enc(batch["x"]).reshape(-1, self.dz)
