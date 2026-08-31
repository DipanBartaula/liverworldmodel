"""
Four more single-ingredient additions to `gnn_leworld_meta_ode`, attacking the three failures
that ratchet MAE cannot see.

  quantile_head  Train the metric you score. Every one of the 39 existing models minimises
                 field-weighted MSE (`util.state_loss`) but is REPORTED on MAE. MSE targets the
                 conditional mean; MAE is minimised by the conditional MEDIAN, and the ratchet's
                 conditional law is right-skewed (bounded below by `prev`), so the two differ.
                 Recomputing the "irreducible floor" about the median instead of the mean moves
                 it from 0.0193 to 0.0185 on the ratchets and from 0.1297 to 0.1010 on the fast
                 fields -- i.e. the current floor is loose by 4.5% / 22%, and part of that is
                 recoverable simply by predicting the right functional. This model emits three
                 quantiles per field, each routed through its own ConstraintHead chain (so each
                 is an independently valid monotone trajectory), trains with the pinball loss,
                 and forecasts with the median. Bonus: it is the only model in the zoo that
                 produces intervals, so coverage becomes measurable.

  tpp_events     The event channel is a point process, not a regression target. Flare onset in
                 the generator is Bernoulli with intensity `0.04 + 0.10*S`, and the FNO
                 experiment lost 10/10 on the long horizon precisely because this component is
                 sparse and broadband. Add an explicit conditional-intensity head trained by
                 Bernoulli log-likelihood, and feed its predicted intensity to the state decoder.
                 Deliberately NOT sampled at rollout: sampling would raise MAE (a sample is a
                 worse point estimate than a median), so the intensity is used as information,
                 not as a generator.

  npe_head       Amortized simulation-based inference. A posterior q(theta | x_0:K) over the
                 generator's own per-patient parameters -- theta = (log susc_eff, responder,
                 udca_start/T) -- trained on simulator draws from the FULL prior, which includes
                 the susceptibility > 2.0 band the forecasting CSV excludes. The drift is then
                 conditioned on the inferred theta. This is the mechanism by which SBI fixes OOD:
                 the held-out-susceptibility cohort is inside the posterior's training support
                 even though it is outside the forecaster's.

  cf_paired      Interventional training. The counterfactual magnitude ratio of the current SOTA
                 is 0.072 -- direction right, magnitude nil -- because the true UDCA effect on C
                 (~0.017) is far below C's aleatoric spread (~0.10) at a loss weight of 0.3, so
                 the observational MSE cannot see it. Fix the signal-to-noise instead of the
                 model: train on paired simulator arms under shifted treatment timing with COMMON
                 RANDOM NUMBERS (measured 6.0x variance reduction on the contrast) and add a loss
                 on the DIFFERENCE of predictions matching the difference of trajectories. Pure
                 objective change -- architecturally identical to the base model.
"""

from __future__ import annotations
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from models.rate_family import _ODEBase
from util import (STATE_DIM, CTX_DIM, RAW_DIM, FIELD_WEIGHTS, FLARE, mlp, state_loss, scale_dim)

SIM_CACHE = "sim_cache.npz"
THETA_DIM = 3
K_CF = 8                       # anchor month for the counterfactual contrast
_CACHE = {}
_WARNED = set()


def _sims():
    """Load the simulator cache once per process. Kept in a module global, NOT on the model, so
    it never enters state_dict or the parameter count."""
    if "loaded" in _CACHE:
        return _CACHE
    _CACHE["loaded"] = True
    if not os.path.exists(SIM_CACHE):
        _CACHE["ok"] = False
        return _CACHE
    z = np.load(SIM_CACHE)
    t = lambda k: torch.from_numpy(z[k])
    _CACHE["npe_x"], _CACHE["npe_ctx"], _CACHE["npe_theta"] = t("npe_x"), t("npe_ctx"), t("npe_theta")
    # counterfactual pairs are only usable where BOTH arms are still untreated at K_CF, so the
    # two arms share the observed prefix exactly and the contrast is the pure treatment effect
    ctxb = z["cf_ctxb"]
    start_b = np.argmax(ctxb[:, :, 6] > 0.5, axis=1)
    usable = np.where(start_b > K_CF)[0]
    for k in ("cf_xa", "cf_xb", "cf_ctxa", "cf_ctxb", "cf_ercp"):
        _CACHE[k] = torch.from_numpy(z[k][usable])
    _CACHE["ok"] = len(usable) > 0
    _CACHE["n_cf"] = len(usable)
    return _CACHE


def _warn_once(name):
    if name not in _WARNED:
        _WARNED.add(name)
        print(f"[{name}] {SIM_CACHE} not found -- auxiliary objective disabled "
              f"(run `python simprep.py`). The model still trains as the base ODE.", flush=True)


# =========================================================================================
# 1. quantile_head
# =========================================================================================
class QuantileHead(_ODEBase):
    name = "quantile_head"
    QUANTILES = (0.1, 0.5, 0.9)
    MEDIAN = 1

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale)
        self.nq = len(self.QUANTILES)
        hid = self.dec[0].out_features
        self.dec = mlp([self.dz, hid, self.nq * RAW_DIM])
        self.register_buffer("qs", torch.tensor(self.QUANTILES))

    def _integrate_q(self, z0, x_base, ctx, ercp, K, end, z_task=None):
        """Returns [B, H, Q, 8]: one independently-valid monotone trajectory per quantile."""
        B = x_base.shape[0]
        z = z0
        prev = x_base.unsqueeze(1).expand(B, self.nq, STATE_DIM)
        outs = []
        for t in range(K, end - 1):
            z = z + self._field(z, ctx[:, t + 1], t, z_task)
            raw = self.dec(z).view(B, self.nq, RAW_DIM)
            e = ercp[:, t + 1].unsqueeze(-1).expand(B, self.nq)
            prev = self.head(raw, prev, e)
            outs.append(prev)
        return torch.stack(outs, 1)

    def _integrate(self, z0, x_base, ctx, ercp, K, end, z_task=None):
        return self._integrate_q(z0, x_base, ctx, ercp, K, end, z_task)[:, :, self.MEDIAN]

    def _pinball(self, pred_q, true):
        w = FIELD_WEIGHTS.to(pred_q.device)
        e = true.unsqueeze(2) - pred_q                       # [B,H,Q,8]
        q = self.qs.view(1, 1, -1, 1)
        return (torch.maximum(q * e, (q - 1.0) * e) * w).mean()

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]
        V = self.val_window
        K = int(torch.randint(V + 4, T // 2 + 1, (1,)).item())
        z_task = self._adapt(x, ctx, ercp, K - V, K, create_graph=True)
        z0 = self._encode(x[:, : K + 1], ctx[:, : K + 1])
        pq = self._integrate_q(z0, x[:, K], ctx, ercp, K, T, z_task)
        loss = self._pinball(pq, x[:, K + 1:])
        return {"loss": loss, "pinball": loss.item()}

    @torch.no_grad()
    def intervals(self, x, ctx, ercp, K):
        """[B, H, Q, 8] quantile trajectories -- for coverage / interval-width diagnostics."""
        z_task = torch.zeros(x.shape[0], self.dtask, device=x.device)
        if K >= 3:
            V = min(self.val_window, K - 1)
            with torch.enable_grad():
                z_task = self._adapt(x, ctx, ercp, max(1, K - V), K, create_graph=False).detach()
        z0 = self._encode(x[:, : K + 1], ctx[:, : K + 1])
        return self._integrate_q(z0, x[:, K], ctx, ercp, K, x.shape[1], z_task)


# =========================================================================================
# 2. tpp_events
# =========================================================================================
class TPPEvents(_ODEBase):
    name = "tpp_events"
    BETA = 0.2                 # keeps the state loss primary once the event NLL nears its ~0.20 floor

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale)
        hid = self.dec[0].out_features
        self.dec = mlp([self.dz + 1, hid, RAW_DIM])           # + predicted intensity
        self.lam = mlp([self.dz + CTX_DIM, hid, 1])           # conditional intensity (logit)

    def _integrate(self, z0, x_base, ctx, ercp, K, end, z_task=None):
        z, prev = z0, x_base
        outs, logits = [], []
        for t in range(K, end - 1):
            z = z + self._field(z, ctx[:, t + 1], t, z_task)
            lg = self.lam(torch.cat([z, ctx[:, t + 1]], -1))  # [B,1] onset logit for month t+1
            logits.append(lg)
            prev = self.head(self.dec(torch.cat([z, torch.sigmoid(lg)], -1)),
                             prev, ercp[:, t + 1])
            outs.append(prev)
        self._logits = torch.cat(logits, 1) if logits else None   # [B, H]
        return torch.stack(outs, 1)

    def training_step(self, batch):
        x, ctx, ercp = batch["x"], batch["ctx"], batch["ercp"]
        T = x.shape[1]
        V = self.val_window
        K = int(torch.randint(V + 4, T // 2 + 1, (1,)).item())
        z_task = self._adapt(x, ctx, ercp, K - V, K, create_graph=True)
        z0 = self._encode(x[:, : K + 1], ctx[:, : K + 1])
        preds = self._integrate(z0, x[:, K], ctx, ercp, K, T, z_task)
        main = state_loss(preds, x[:, K + 1:])
        # generator sets flare = max(onset, prev*0.4), and onset in {0,1}, so flare == 1 <=> onset
        onset = (x[:, K + 1:, FLARE] >= 0.999).float()
        nll = Fn.binary_cross_entropy_with_logits(self._logits, onset)
        return {"loss": main + self.BETA * nll, "state": main.item(), "event_nll": nll.item()}


# =========================================================================================
# 3. npe_head
# =========================================================================================
class NPEHead(_ODEBase):
    name = "npe_head"
    BETA = 1.0
    NPE_BATCH = 96

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale, extra_in=THETA_DIM)
        dq = scale_dim(24, scale)
        self.npe_gru = nn.GRU(STATE_DIM + CTX_DIM, dq, batch_first=True)
        self.npe_out = mlp([dq, dq, 2 * THETA_DIM])            # mean, log-std
        _sims()

    def _posterior(self, x_obs, ctx_obs):
        _, hN = self.npe_gru(torch.cat([x_obs, ctx_obs], -1))
        o = self.npe_out(hN[-1])
        mu, logstd = o[:, :THETA_DIM], o[:, THETA_DIM:].clamp(-5, 3)
        return mu, logstd

    def _encode(self, x_obs, ctx_obs):
        mu, _ = self._posterior(x_obs, ctx_obs)
        # DETACHED: the posterior is trained only by its own NPE objective, as amortized
        # inference requires. Letting the forecasting loss reshape it would make q(theta|x) some
        # other useful code rather than a posterior, and would forfeit the OOD argument -- the
        # whole point is that q was fit on the prior's full support, including susc > 2.
        self._theta = mu.detach()
        return super()._encode(x_obs, ctx_obs)

    def _field(self, z, ctx_t, t, z_task=None):
        te = self.temb(torch.full((z.shape[0],), float(t), device=z.device))
        return self.reaction(torch.cat([z, ctx_t, te, z_task, self._theta], -1))

    def _npe_loss(self, T):
        c = _sims()
        if not c.get("ok"):
            _warn_once(self.name)
            return None
        n = c["npe_x"].shape[0]
        i = torch.randint(0, n, (self.NPE_BATCH,))
        K = int(torch.randint(6, T // 2 + 1, (1,)).item())
        mu, logstd = self._posterior(c["npe_x"][i, : K + 1], c["npe_ctx"][i, : K + 1])
        th = c["npe_theta"][i]
        std = logstd.exp()
        return (0.5 * ((th - mu) / std) ** 2 + logstd).mean()   # Gaussian NLL

    def training_step(self, batch):
        out = super().training_step(batch)
        nll = self._npe_loss(batch["x"].shape[1])
        if nll is not None:
            out["loss"] = out["loss"] + self.BETA * nll
            out["npe_nll"] = nll.item()
        return out


# =========================================================================================
# 4. cf_paired
# =========================================================================================
class CFPaired(_ODEBase):
    name = "cf_paired"
    BETA = 0.1                 # relative-contrast term, so ~2x the main loss's scale at init
    CF_BATCH = 48

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale)
        _sims()

    def _rollout_arm(self, x, ctx, ercp, K, T):
        z_task = self._adapt(x, ctx, ercp, K - self.val_window, K, create_graph=True)
        z0 = self._encode(x[:, : K + 1], ctx[:, : K + 1])
        return self._integrate(z0, x[:, K], ctx, ercp, K, T, z_task)

    def _cf_loss(self, T):
        c = _sims()
        if not c.get("ok"):
            _warn_once(self.name)
            return None
        n = c["n_cf"]
        i = torch.randint(0, n, (self.CF_BATCH,))
        xa, xb = c["cf_xa"][i], c["cf_xb"][i]
        ca, cb = c["cf_ctxa"][i], c["cf_ctxb"][i]
        e = c["cf_ercp"][i]
        K = K_CF
        # both arms are untreated through K, so they share the observed prefix exactly and the
        # difference below is the pure effect of shifting treatment onset
        pa = self._rollout_arm(xa, ca, e, K, T)
        pb = self._rollout_arm(xa, cb, e, K, T)
        dtrue = xb[:, K + 1:] - xa[:, K + 1:]
        # RELATIVE contrast error. The absolute contrast is tiny (dC ~ 0.017), so an absolute
        # loss here is ~0.6% of the total and changes nothing -- exactly the signal-to-noise
        # problem this experiment exists to fix. Normalising by the contrast's own magnitude
        # makes the term scale-free and directly interpretable: 1.0 = "predicts no effect at
        # all" (the current SOTA, magnitude ratio 0.072), 0.0 = exact.
        num = state_loss(pb - pa, dtrue)
        den = state_loss(torch.zeros_like(dtrue), dtrue) + 1e-8
        return num / den

    def training_step(self, batch):
        out = super().training_step(batch)
        cf = self._cf_loss(batch["x"].shape[1])
        if cf is not None:
            out["loss"] = out["loss"] + self.BETA * cf
            out["cf"] = cf.item()
        return out
