"""
The Ada-JEPA family -- five variations on TimeSeriesJEPA that change ONE ingredient each, so the
sweep can attribute any gain to that ingredient.

Shared JEPA core (identical to `ts_jepa`): an ONLINE sequence encoder reads the observed window
and is pooled into a patient summary; a momentum (EMA) TARGET encoder reads the true future with
no gradient; a predictor maps (summary, sinusoidal query for a future month, that month's
context [, task code]) to the TARGET latent at that month; a decoder anchored on both the
predicted and the target latent emits raw increments for the shared ConstraintHead. The EMA
target plus a VICReg variance/covariance term are the anti-collapse devices.

What each variant changes:

  ada_jepa         ADAPTIVE JEPA. A per-patient task code z_task also conditions the predictor
                   and is found at test time by a few GRADIENT steps on a self-supervised
                   objective -- re-decode the held-in tail of the observed window from an
                   earlier anchor. Meta-trained: the outer JEPA loss on the future is
                   backpropagated THROUGH that inner search (MAML on the task code), so the
                   predictor is explicitly optimised to be steerable in a few test-time steps.
                   The JEPA bet is that searching in the LATENT the target encoder defines is a
                   better-conditioned search than searching in raw state space.

  fno_jepa         FOURIER NEURAL OPERATOR predictor. Instead of predicting each future month
                   independently (a pointwise MLP over the horizon axis), the predictor treats
                   the whole horizon as a FUNCTION and applies a neural operator: lift to a
                   channel space, then alternate a spectral convolution (rFFT along the horizon,
                   learned complex multiply on the lowest 8 modes, inverse rFFT) with a pointwise
                   linear path. The kernel is global in horizon and learned in frequency, so the
                   predictor can express slow ratchet drift and ERCP-driven oscillation as
                   separate modes, and is defined on any horizon grid (which is what the
                   beyond-training-horizon probe tests).

  gnn_ada_jepa     Ada-JEPA + a cosine-attention message pass over the patient summaries in the
                   batch (`GraphRefine`), so a patient's forecast can borrow dynamics from
                   similar patients before the task-code search runs.

  gnn_fno_jepa     FNO-JEPA + the same patient-graph refinement.

  mamba2_ada_jepa  Ada-JEPA with the Transformer encoder replaced by a stack of MAMBA-2 SSD
                   blocks (selective state-space; see models/mamba2.py). Linear rather than
                   quadratic in window length, and a recurrent state rather than attention --
                   the question is whether an explicit decaying state suits a monotone ratchet
                   process better than attention at equal parameter count.

Everything routes through the shared ConstraintHead, so the monotonicity/bounds guarantees hold
by construction for all five.
"""

from __future__ import annotations
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.composed import GraphRefine
from models.le_world_model import vicreg
from models.mamba2 import Mamba2Block, RMSNorm
from util import STATE_DIM, CTX_DIM, RAW_DIM, SinusoidalPosEmb, mlp, state_loss, scale_dim


# =========================================================================================
# Sequence encoders (online / EMA-target)
# =========================================================================================
class _TransformerEncoder(nn.Module):
    """ts_jepa's encoder, with nhead relaxed to the largest of 4/2/1 that divides dz -- the
    sweep's parameter search needs widths on a finer grid than multiples of 4."""

    def __init__(self, dz, nlayers=2):
        super().__init__()
        nhead = next(h for h in (4, 2, 1) if dz % h == 0)
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
class _Mamba2Encoder(nn.Module):
    """Same interface as ts_jepa's Transformer encoder: (states, ctx) -> [B, L, dz]."""

    def __init__(self, dz, nlayers=2, d_state=16, nheads=4):
        super().__init__()
        self.inp = nn.Linear(STATE_DIM + CTX_DIM, dz)
        self.pos = SinusoidalPosEmb(dz)
        self.blocks = nn.ModuleList(
            [Mamba2Block(dz, d_state=d_state, nheads=nheads) for _ in range(nlayers)])
        self.norms = nn.ModuleList([RMSNorm(dz) for _ in range(nlayers)])
        self.out_norm = RMSNorm(dz)

    def forward(self, states, ctx):
        L = states.shape[1]
        h = self.inp(torch.cat([states, ctx], -1))
        h = h + self.pos(torch.arange(L, device=states.device)).unsqueeze(0)
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + blk(nrm(h))                                   # pre-norm residual
        return self.out_norm(h)


# =========================================================================================
# Predictors: [B, H, din] horizon tokens -> [B, H, dz] predicted target latents
# =========================================================================================
class _PointwisePredictor(nn.Module):
    """The ts_jepa predictor: an MLP applied independently at each future month."""

    def __init__(self, din, dz, hid):
        super().__init__()
        self.net = mlp([din, hid, dz])

    def forward(self, tok):
        return self.net(tok)


class _SpectralConv1d(nn.Module):
    """FNO layer over the horizon axis: rFFT -> learned complex channel mixing on the lowest
    `modes` frequencies (higher modes are truncated to zero) -> inverse rFFT."""

    def __init__(self, ch, modes):
        super().__init__()
        self.modes = modes
        s = 1.0 / ch
        self.wr = nn.Parameter(torch.randn(modes, ch, ch) * s)
        self.wi = nn.Parameter(torch.randn(modes, ch, ch) * s)

    def forward(self, u):                                          # u: [B, H, ch]
        H = u.shape[1]
        uf = torch.fft.rfft(u, dim=1)                              # [B, H//2+1, ch]
        m = min(self.modes, uf.shape[1])
        w = torch.complex(self.wr[:m], self.wi[:m])
        low = torch.einsum("bmi,mio->bmo", uf[:, :m], w)
        if uf.shape[1] > m:                                        # truncate the high modes
            pad = low.new_zeros(low.shape[0], uf.shape[1] - m, low.shape[2])
            low = torch.cat([low, pad], dim=1)
        return torch.fft.irfft(low, n=H, dim=1)


class _FNOPredictor(nn.Module):
    """Lift -> nblocks x (spectral conv + pointwise linear, GELU) -> project to the latent."""

    def __init__(self, din, dz, ch, modes=8, nblocks=2):
        super().__init__()
        self.lift = nn.Linear(din, ch)
        self.spec = nn.ModuleList([_SpectralConv1d(ch, modes) for _ in range(nblocks)])
        self.pw = nn.ModuleList([nn.Linear(ch, ch) for _ in range(nblocks)])
        self.proj = mlp([ch, ch, dz])

    def forward(self, tok):
        u = self.lift(tok)
        for i, (sp, pw) in enumerate(zip(self.spec, self.pw)):
            v = sp(u) + pw(u)
            u = F.gelu(v) if i < len(self.spec) - 1 else v
        return self.proj(u)


# =========================================================================================
# Shared core
# =========================================================================================
class _JEPACore(BaseModel):
    def __init__(self, T, dz=44, hid=88, dtask=16, ema=0.99, beta_vic=1.0, encoder="transformer",
                 predictor="mlp", graph=False, adapt=False, val_window=6, inner_steps=2,
                 inner_lr=0.2, fno_width=40, fno_modes=8, scale=1.0):
        super().__init__(T)
        # dz moves on a coarse grid (attention heads / SSD heads must divide it); the predictor
        # width moves on a grid of 1, so the sweep's binary search can hit a param target between
        # two dz steps. Both grow with `scale`, so total params stay monotone in it.
        dz = scale_dim(dz, scale, 2)
        self.dz, self.ema, self.beta_vic = dz, ema, beta_vic
        self.adapt, self.val_window = adapt, val_window
        self.inner_steps, self.inner_lr = inner_steps, inner_lr
        self.dtask = scale_dim(dtask, scale) if adapt else 0

        self.online = _Mamba2Encoder(dz) if encoder == "mamba2" else _TransformerEncoder(dz)
        self.target = copy.deepcopy(self.online)                   # EMA momentum target
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.graph = GraphRefine(dz) if graph else None
        self.qpos = SinusoidalPosEmb(dz)
        din = dz + dz + CTX_DIM + self.dtask                       # summary | query | ctx | task
        self.pred = _FNOPredictor(din, dz, scale_dim(fno_width, scale), fno_modes) \
            if predictor == "fno" else _PointwisePredictor(din, dz, scale_dim(hid, scale))
        self.dec = mlp([dz, dz, RAW_DIM])

    # -- JEPA plumbing -------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self):
        for po, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.mul_(self.ema).add_(po, alpha=1 - self.ema)

    def _summary(self, x_obs, ctx_obs):
        s = self.online(x_obs, ctx_obs).mean(1)                    # observed-window summary
        return self.graph(s) if self.graph is not None else s

    def _predict_latents(self, summary, ctx, positions, z_task=None):
        """summary [B,dz], positions [H] absolute future months -> predicted latents [B,H,dz]."""
        B, H = summary.shape[0], positions.shape[0]
        parts = [summary.unsqueeze(1).expand(B, H, -1),
                 self.qpos(positions).unsqueeze(0).expand(B, H, -1),
                 ctx[:, positions]]
        if self.dtask:
            parts.append(z_task.unsqueeze(1).expand(B, H, -1))
        return self.pred(torch.cat(parts, -1))

    def _decode_chain(self, z, x_base, ercp, positions):
        """Chain latents through the decoder + ConstraintHead from the last observed state."""
        prev, outs = x_base, []
        for i, t in enumerate(positions.tolist()):
            prev = self.head(self.dec(z[:, i]), prev, ercp[:, t])
            outs.append(prev)
        return torch.stack(outs, 1)

    # -- test-time gradient search on the task code (ada variants) -----------------------
    def _adapt(self, x, ctx, ercp, Ks, K, create_graph):
        """Fit z_task so the model re-decodes the held-in tail Ks+1..K of the OBSERVED window
        from an anchor at Ks. The summary does not depend on z_task, so it is computed once and
        the outer loss still backprops through it."""
        summary = self._summary(x[:, : Ks + 1], ctx[:, : Ks + 1])
        positions = torch.arange(Ks + 1, K + 1, device=x.device)
        z_task = torch.zeros(x.shape[0], self.dtask, device=x.device, requires_grad=True)
        for _ in range(self.inner_steps):
            z = self._predict_latents(summary, ctx, positions, z_task)
            loss = state_loss(self._decode_chain(z, x[:, Ks], ercp, positions), x[:, Ks + 1:K + 1])
            (g,) = torch.autograd.grad(loss, z_task, create_graph=create_graph)
            z_task = z_task - self.inner_lr * g
        return z_task

    def _task_code(self, x, ctx, ercp, K, create_graph):
        if not self.adapt:
            return None
        if K < 3:
            return torch.zeros(x.shape[0], self.dtask, device=x.device)
        V = min(self.val_window, K - 1)
        return self._adapt(x, ctx, ercp, max(1, K - V), K, create_graph)

    # -- BaseModel contract ---------------------------------------------------------------
    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]
        lo = self.val_window + 4 if self.adapt else 6
        K = int(torch.randint(lo, T // 2 + 1, (1,)).item())
        z_task = self._task_code(x, ctx, ercp, K, create_graph=True)   # inner search (MAML)

        summary = self._summary(x[:, : K + 1], ctx[:, : K + 1])
        positions = torch.arange(K + 1, T, device=x.device)
        z_pred = self._predict_latents(summary, ctx, positions, z_task)
        with torch.no_grad():
            z_tar = self.target(x, ctx)[:, positions]                  # EMA target latents
        loss_latent = F.mse_loss(z_pred, z_tar)

        # decoder anchored on BOTH spaces, teacher-forced along the target space
        tgt = x[:, K + 1:]
        loss_state = state_loss(self._decode_chain(z_pred, x[:, K], ercp, positions), tgt) \
            + state_loss(self._decode_chain(z_tar, x[:, K], ercp, positions), tgt)
        loss_vic = vicreg(z_pred.reshape(-1, self.dz))
        loss = loss_state + loss_latent + self.beta_vic * loss_vic
        self._ema_update()
        return {"loss": loss, "state": loss_state.item(), "latent": loss_latent.item(),
                "vic": loss_vic.item()}

    def rollout(self, x, ctx, ercp, K):
        z_task = None
        if self.adapt:
            with torch.enable_grad():
                z_task = self._task_code(x, ctx, ercp, K, create_graph=False).detach()
        with torch.no_grad():
            pred = self._init_pred(x, K)
            summary = self._summary(x[:, : K + 1], ctx[:, : K + 1])
            positions = torch.arange(K + 1, x.shape[1], device=x.device)
            z = self._predict_latents(summary, ctx, positions, z_task)
            pred[:, K + 1:] = self._decode_chain(z, x[:, K], ercp, positions)
        return pred

    @torch.no_grad()
    def latent(self, batch):
        return self.online(batch["x"], batch["ctx"]).reshape(-1, self.dz)


# =========================================================================================
# The five registered variants
# =========================================================================================
class AdaJEPA(_JEPACore):
    name = "ada_jepa"

    def __init__(self, T, scale=1.0):
        super().__init__(T, encoder="transformer", predictor="mlp", adapt=True, scale=scale)


class FNOJEPA(_JEPACore):
    name = "fno_jepa"

    def __init__(self, T, scale=1.0):
        super().__init__(T, encoder="transformer", predictor="fno", adapt=False, scale=scale)


class GNNAdaJEPA(_JEPACore):
    name = "gnn_ada_jepa"

    def __init__(self, T, scale=1.0):
        super().__init__(T, encoder="transformer", predictor="mlp", adapt=True, graph=True,
                         scale=scale)


class GNNFNOJEPA(_JEPACore):
    name = "gnn_fno_jepa"

    def __init__(self, T, scale=1.0):
        super().__init__(T, encoder="transformer", predictor="fno", adapt=False, graph=True,
                         scale=scale)


class Mamba2AdaJEPA(_JEPACore):
    name = "mamba2_ada_jepa"

    def __init__(self, T, scale=1.0):
        super().__init__(T, encoder="mamba2", predictor="mlp", adapt=True, scale=scale)
