"""
Anti-collapse variants of the eight ingredient models, using the two devices the LeWorld and
TS-JEPA models already rely on, transplanted onto the Neural-ODE backbone.

Why. Measured like-for-like -- encoder output at three observation anchors, the same population
`_LatentPDE.latent()` reports -- the ODE family sits at effective rank ~3.5 against a data
intrinsic rank of 2.84, and `hazard_mono` falls to 2.41, BELOW the data's own rank. (The JEPA
family's headline 9-13 is not a richer latent: it counts one code per month over a 60-month
trajectory, so temporal spread inflates it. Under the anchor measurement `ts_jepa` is 3.69 and
`rate_anchor` is 4.06.) The ODE backbone encodes with a GRU final hidden state and nothing pushes
its dimensions apart, so a genuinely low-rank code is cheap for it to learn.

The two devices, both applied to the ENCODER output -- which is exactly the quantity the collapse
metric measures, so the lever matches the diagnostic:

  VICReg (the LeWorld device).      Variance term pushes every latent dimension's std up to gamma;
                                    covariance term drives the off-diagonals to zero. Attacks
                                    DIMENSIONAL collapse -- the failure where the code is live but
                                    confined to a low-rank subspace, which is what rank 2.4 means.

  EMA-target prediction (TS-JEPA).  An exponential-moving-average copy of the encoder encodes a
                                    LATER observation anchor with no gradient; a small predictor
                                    maps (code at anchor a, gap) to that target. The target cannot
                                    follow the online net into a constant, which is the classic
                                    guard against REPRESENTATIONAL collapse.

Both act only on encoder outputs, so every one of the eight models takes the ingredient unchanged
-- no surgery on the variants that override `_integrate` (time_warp, ude_hybrid, tpp_events,
quantile_head). The base objective is untouched and simply has the two terms added, so any change
in the other metrics is attributable to the regulariser and nothing else.

WHAT THE FULL SWEEP FOUND (8 pairs x 10 sizes, matched by parameter target -- the honest
comparison; a best-of-10 view flatters the variants by picking up favourable noise):

  effective rank   +4.02 mean, the variant wins 9-10 of 10 sizes on EVERY model.  Rank roughly
                   triples (3.3-5.5 -> 7.0-11.1). The device does exactly what it is for, and
                   hazard_mono's genuine collapse (2.41, below the data's 2.84) is gone.
  susc OOD         -6.4% mean, better in 6/8 models (quantile_head -20.2%, hazard_mono -14.8%,
                   ude_hybrid -11.2%).
  ratchet MAE      +4.6% mean, better in only 1/8.
  udca OOD        +12.9% mean, better in 0/8 -- the most consistent regression in the set.
  beyond horizon   +9.8% mean, better in 1/8.
  fast MAE         +0.3% -- a wash.

So this is NOT a free win, and the tension I first measured at a 6-epoch gate was real but
mislocated. It is not rank-vs-OOD in general: with full training the susceptibility axis IMPROVES.
What degrades is in-horizon accuracy, treatment-timing OOD, and long-horizon extrapolation. The
reading that fits: VICReg stops any single direction dominating the code, which helps when the
useful signal is spread (inferring an unseen progression SPEED) and hurts when the task needs one
sharp direction (an unseen treatment ONSET is a localised event in time, not a global rate).

Practical upshot: use these variants when the collapse diagnostic matters or the susceptibility
cohort is the target; keep the un-regularised base when ratchet MAE or treatment timing is. The
SOTA is unchanged -- rate_anchor_ac (0.0209) does not beat rate_anchor (0.0207) on anything but
rank.
"""

from __future__ import annotations
import copy

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from models.le_world_model import vicreg
from models.rate_family import RateAnchor, HazardMono, TimeWarp, UDEHybrid
from models.dist_family import QuantileHead, TPPEvents, NPEHead, CFPaired
from util import mlp


class _AntiCollapse:
    """Mixin: adds VICReg + an EMA-target latent-prediction term to any `_ODEBase` subclass."""

    # Tuned on a 6-epoch, ~100K-param gate (hazard_mono = the genuinely collapsed model,
    # rate_anchor = the current SOTA that must not regress). At the naive 0.5/0.5 the VICReg term
    # is ~0.46 against a main loss of ~0.01 -- 40x the objective -- and ratchet MAE degrades
    # 5-18%. At 0.02/0.05 the ratchet is retained (-0.1% / +0.9%) while hazard_mono's rank goes
    # 1.71 -> 5.02, i.e. from BELOW the data's intrinsic 2.84 to comfortably above it.
    BETA_VIC = 0.02
    BETA_JEPA = 0.05
    EMA = 0.99
    ANCHORS = (10, 16, 22, 28)

    def __init__(self, T, scale=1.0, **kw):
        super().__init__(T, scale=scale, **kw)
        # EMA target copy of exactly the modules the encoder path uses
        self.t_state_enc = copy.deepcopy(self.state_enc)
        self.t_enc_gru = copy.deepcopy(self.enc_gru)
        self.t_graph = copy.deepcopy(self.graph)
        for m in (self.t_state_enc, self.t_enc_gru, self.t_graph):
            for p in m.parameters():
                p.requires_grad_(False)
        hid = self.dec[0].out_features
        self.jepa_pred = mlp([self.dz + 1, hid, self.dz])     # (code, anchor gap) -> target code

    # -- encoder paths without the subclasses' side effects (rate_anchor sets head.gain,
    #    npe_head sets _theta), so the regulariser can never perturb the main objective ----
    def _plain_encode(self, x_obs, ctx_obs):
        seq = torch.cat([self.state_enc(x_obs), ctx_obs], -1)
        _, hN = self.enc_gru(seq)
        return self.graph(hN[-1])

    @torch.no_grad()
    def _target_encode(self, x_obs, ctx_obs):
        seq = torch.cat([self.t_state_enc(x_obs), ctx_obs], -1)
        _, hN = self.t_enc_gru(seq)
        return self.t_graph(hN[-1])

    @torch.no_grad()
    def _ema_update(self):
        pairs = [(self.state_enc, self.t_state_enc), (self.enc_gru, self.t_enc_gru),
                 (self.graph, self.t_graph)]
        for online, target in pairs:
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(self.EMA).add_(po, alpha=1 - self.EMA)

    def _collapse_loss(self, x, ctx):
        T = x.shape[1]
        anchors = [a for a in self.ANCHORS if a < T - 1]
        zs = [self._plain_encode(x[:, : a + 1], ctx[:, : a + 1]) for a in anchors]

        # VICReg over the pooled anchor codes: per-dim std -> gamma, off-diagonal cov -> 0
        vic = vicreg(torch.cat(zs, 0))

        # EMA-target prediction: code at anchor a must predict the target's code at a later anchor
        i = int(torch.randint(0, len(anchors) - 1, (1,)).item())
        j = int(torch.randint(i + 1, len(anchors), (1,)).item())
        gap = torch.full((x.shape[0], 1), (anchors[j] - anchors[i]) / float(T), device=x.device)
        pred = self.jepa_pred(torch.cat([zs[i], gap], -1))
        tgt = self._target_encode(x[:, : anchors[j] + 1], ctx[:, : anchors[j] + 1])
        return vic, Fn.mse_loss(pred, tgt)

    def training_step(self, batch):
        out = super().training_step(batch)
        vic, jep = self._collapse_loss(batch["x"], batch["ctx"])
        out["loss"] = out["loss"] + self.BETA_VIC * vic + self.BETA_JEPA * jep
        out["vic"], out["jepa"] = vic.item(), jep.item()
        self._ema_update()
        return out


class RateAnchorAC(_AntiCollapse, RateAnchor):
    name = "rate_anchor_ac"


class HazardMonoAC(_AntiCollapse, HazardMono):
    name = "hazard_mono_ac"


class TimeWarpAC(_AntiCollapse, TimeWarp):
    name = "time_warp_ac"


class UDEHybridAC(_AntiCollapse, UDEHybrid):
    name = "ude_hybrid_ac"


class QuantileHeadAC(_AntiCollapse, QuantileHead):
    name = "quantile_head_ac"


class TPPEventsAC(_AntiCollapse, TPPEvents):
    name = "tpp_events_ac"


class NPEHeadAC(_AntiCollapse, NPEHead):
    name = "npe_head_ac"


class CFPairedAC(_AntiCollapse, CFPaired):
    name = "cf_paired_ac"
