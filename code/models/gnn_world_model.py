"""
GNNTimeSeries -- a graph-over-patients world model.

Each patient's observed trajectory is encoded (GRU over state+ctx) into a patient embedding
that summarises "where this patient's disease has gone so far". Within a batch we build a
similarity graph over those embeddings (cosine attention over all patients) and message-pass,
so a patient's forecast can borrow dynamics from similar patients. The graph-refined embedding
then conditions an autoregressive predictor (with a sinusoidal time embedding of the target
month) that emits raw increments for the ConstraintHead.

Why it might be wrong: within one generator, patients are i.i.d. given their hidden params, so
the neighbour graph may add little over a good per-patient encoder -- the eval will show whether
the graph term actually helps or is decorative.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss, scale_dim


class GNNTimeSeries(BaseModel):
    name = "gnn_world_model"

    def __init__(self, T, demb=48, hid=64, t_dim=16, scale=1.0):
        super().__init__(T)
        demb, hid, t_dim = scale_dim(demb, scale), scale_dim(hid, scale), scale_dim(t_dim, scale)
        self.demb = demb
        self.enc = nn.GRU(STATE_DIM + CTX_DIM, demb, batch_first=True)
        self.q = nn.Linear(demb, demb)
        self.k = nn.Linear(demb, demb)
        self.v = nn.Linear(demb, demb)
        self.g_out = nn.Linear(demb, demb)
        self.temb = SinusoidalPosEmb(t_dim)
        self.pred = mlp([demb + STATE_DIM + CTX_DIM + t_dim, hid, hid, RAW_DIM])

    def _patient_emb(self, x_obs, ctx_obs):
        _, hN = self.enc(torch.cat([x_obs, ctx_obs], -1))
        return hN[-1]                                          # [B, demb]

    def _graph(self, e):
        # cosine self-attention message passing over the batch of patients
        q, k, v = self.q(e), self.k(e), self.v(e)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        att = torch.softmax(q @ k.T / (self.demb ** 0.5), dim=-1)   # [B, B]
        return e + self.g_out(att @ v)                        # residual graph update

    def _predict(self, e_ref, prev, ctx_t, t):
        te = self.temb(torch.full((prev.shape[0],), float(t), device=prev.device))
        return self.pred(torch.cat([e_ref, prev, ctx_t, te], -1))

    def _run(self, x, ctx, ercp, K, T):
        e = self._patient_emb(x[:, : K + 1], ctx[:, : K + 1])
        e_ref = self._graph(e)
        prev = x[:, K]
        outs = []
        for t in range(K, T - 1):
            raw = self._predict(e_ref, prev, ctx[:, t + 1], t + 1)
            prev = self.head(raw, prev, ercp[:, t + 1])
            outs.append(prev)
        return torch.stack(outs, 1)

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]
        K = int(torch.randint(6, T // 2 + 1, (1,)).item())
        preds = self._run(x, ctx, ercp, K, T)
        return {"loss": state_loss(preds, x[:, K + 1:])}

    @torch.no_grad()
    def rollout(self, x, ctx, ercp, K):
        pred = self._init_pred(x, K)
        pred[:, K + 1:] = self._run(x, ctx, ercp, K, x.shape[1])
        return pred
