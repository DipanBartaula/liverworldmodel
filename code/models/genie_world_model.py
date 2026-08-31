"""
GenieWorldModel -- a Genie-style GENERATIVE world model (deliberately NOT JEPA).

Following Genie's recipe adapted to a continuous 8-D state: (1) an encoder tokenises a state
into a latent z; (2) a Latent Action Model infers a discrete latent "action" between two
consecutive latents through a small VQ codebook (straight-through); (3) a dynamics model
predicts the next state from (z_t, action) and (4) an action PRIOR learns to predict which
action follows from the current latent + context, so at inference -- where the true next state
is unavailable -- we sample the action from the prior and generate forward. Output goes through
the ConstraintHead. Unlike JEPA it models the state itself (reconstruction), and its
controllability lives in the discrete latent-action bottleneck.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, mlp, state_loss, scale_dim


class GenieWorldModel(BaseModel):
    name = "genie_world_model"

    def __init__(self, T, dz=32, da=8, n_actions=8, hid=64, beta_commit=0.25, scale=1.0):
        super().__init__(T)
        dz, da, hid = scale_dim(dz, scale), scale_dim(da, scale), scale_dim(hid, scale)
        self.n_actions, self.beta_commit = n_actions, beta_commit
        self.enc = mlp([STATE_DIM, hid, dz])
        self.codebook = nn.Embedding(n_actions, da)
        self.lam = mlp([2 * dz, hid, da])                      # infer action from (z_t, z_{t+1})
        self.prior = mlp([dz + CTX_DIM, hid, n_actions])       # action prior for generation
        self.dyn = mlp([dz + da + CTX_DIM, hid, RAW_DIM])

    def _quantize(self, a_cont):
        d = torch.cdist(a_cont, self.codebook.weight)          # [..., n_actions]
        idx = d.argmin(-1)
        aq = self.codebook(idx)
        aq_st = a_cont + (aq - a_cont).detach()                # straight-through
        return aq_st, aq, idx

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        cur, nxt = x[:, :-1], x[:, 1:]
        z, zn = self.enc(cur), self.enc(nxt)
        a_cont = self.lam(torch.cat([z, zn], -1))
        aq_st, aq, idx = self._quantize(a_cont)
        raw = self.dyn(torch.cat([z, aq_st, ctx[:, 1:]], -1))
        pred = self.head(raw, cur, ercp[:, 1:])
        loss_rec = state_loss(pred, nxt)
        loss_vq = F.mse_loss(aq, a_cont.detach()) + self.beta_commit * F.mse_loss(a_cont, aq.detach())
        logits = self.prior(torch.cat([z, ctx[:, 1:]], -1))
        loss_prior = F.cross_entropy(logits.reshape(-1, self.n_actions), idx.reshape(-1))
        loss = loss_rec + loss_vq + loss_prior
        return {"loss": loss, "rec": loss_rec.item(), "vq": loss_vq.item(), "prior": loss_prior.item()}

    @torch.no_grad()
    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        T = x.shape[1]
        for t in range(K, T - 1):
            z = self.enc(pred[:, t])
            idx = self.prior(torch.cat([z, ctx[:, t + 1]], -1)).argmax(-1)   # sample action from prior
            a = self.codebook(idx)
            raw = self.dyn(torch.cat([z, a, ctx[:, t + 1]], -1))
            pred[:, t + 1] = self.head(raw, pred[:, t], ercp[:, t + 1])
        return pred
