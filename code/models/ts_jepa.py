"""
TimeSeriesJEPA -- a masked, time-series JEPA.

A Transformer ONLINE encoder reads the observed window; a momentum (EMA) TARGET encoder reads
the true future states with no gradient (the classic JEPA anti-collapse device -- the target
cannot follow the online net into a constant). A predictor takes the observed summary plus a
sinusoidal query for a future position and predicts that position's TARGET latent. A decoder,
anchored on BOTH the predicted and the target latent, maps latents to raw increments for the
ConstraintHead. Predicting in a learned latent (not raw x) is the JEPA bet; the EMA target and
a variance term are what keep it from collapsing.
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.le_world_model import vicreg
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss, scale_dim


class _Encoder(nn.Module):
    def __init__(self, dz, nhead=4, nlayers=2):
        super().__init__()
        self.inp = nn.Linear(STATE_DIM + CTX_DIM, dz)
        self.pos = SinusoidalPosEmb(dz)
        layer = nn.TransformerEncoderLayer(dz, nhead, dz * 2, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, nlayers)

    def forward(self, states, ctx):
        L = states.shape[1]
        h = self.inp(torch.cat([states, ctx], -1))
        h = h + self.pos(torch.arange(L, device=states.device)).unsqueeze(0)
        return self.tr(h)


class TimeSeriesJEPA(BaseModel):
    name = "ts_jepa"

    def __init__(self, T, dz=48, ema=0.99, beta_vic=1.0, scale=1.0):
        super().__init__(T)
        dz = scale_dim(dz, scale, 4)                           # keep divisible by nhead=4
        self.dz, self.ema = dz, ema
        self.online = _Encoder(dz)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.qpos = SinusoidalPosEmb(dz)
        self.pred = mlp([dz + dz + CTX_DIM, dz * 2, dz])       # (summary, posq, ctx_tgt) -> latent
        self.dec = mlp([dz, dz, RAW_DIM])
        self.beta_vic = beta_vic

    @torch.no_grad()
    def _ema_update(self):
        for po, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.mul_(self.ema).add_(po, alpha=1 - self.ema)

    def _predict_latents(self, summary, ctx, positions):
        # summary: [B, dz]; positions: [H] future absolute months; returns [B, H, dz]
        q = self.qpos(positions)                               # [H, dz]
        B, H = summary.shape[0], positions.shape[0]
        s = summary.unsqueeze(1).expand(B, H, -1)
        qe = q.unsqueeze(0).expand(B, H, -1)
        return self.pred(torch.cat([s, qe, ctx[:, positions]], -1))

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        B, T, _ = x.shape
        K = int(torch.randint(6, T // 2 + 1, (1,)).item())
        h_obs = self.online(x[:, : K + 1], ctx[:, : K + 1])
        summary = h_obs.mean(1)                                # observed-window summary
        positions = torch.arange(K + 1, T, device=x.device)
        z_pred = self._predict_latents(summary, ctx, positions)
        with torch.no_grad():
            h_tar = self.target(x, ctx)                        # target latents (no grad)
            z_tar = h_tar[:, positions]
        loss_latent = F.mse_loss(z_pred, z_tar)
        # decoder anchored on both spaces, chained from the last observed state
        prev = x[:, K]
        dp, dt, tgt = [], [], []
        for i, t in enumerate(positions.tolist()):
            dp.append(self.head(self.dec(z_pred[:, i]), prev, ercp[:, t]))
            dt.append(self.head(self.dec(z_tar[:, i]), prev, ercp[:, t]))
            prev = dt[-1]                                      # teacher-forced along target space
            tgt.append(x[:, t])
        loss_state = state_loss(torch.stack(dp, 1), torch.stack(tgt, 1)) \
            + state_loss(torch.stack(dt, 1), torch.stack(tgt, 1))
        loss_vic = vicreg(z_pred.reshape(-1, self.dz))
        loss = loss_state + loss_latent + self.beta_vic * loss_vic
        self._ema_update()
        return {"loss": loss, "state": loss_state.item(), "latent": loss_latent.item(),
                "vic": loss_vic.item()}

    @torch.no_grad()
    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        h_obs = self.online(x[:, : K + 1], ctx[:, : K + 1])
        summary = h_obs.mean(1)
        # sinusoidal query positions (no learned table) -> extrapolates past training T
        positions = torch.arange(K + 1, x.shape[1], device=x.device)
        z_pred = self._predict_latents(summary, ctx, positions)
        prev = x[:, K]
        for i, t in enumerate(positions.tolist()):
            prev = self.head(self.dec(z_pred[:, i]), prev, ercp[:, t])
            pred[:, t] = prev
        return pred

    @torch.no_grad()
    def latent(self, batch):
        h = self.online(batch["x"], batch["ctx"])
        return h.reshape(-1, self.dz)
