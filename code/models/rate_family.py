"""
The rate-invariance family: four single-ingredient additions to `gnn_leworld_meta_ode` (the
sweep's best model), each attacking the SAME diagnosed failure from a different direction.

The diagnosis. The preregistered falsification harness found the SOTA is 3.7x worse (floor-
relative) under generator B1, a pure COEFFICIENT shift, but only 1.26x worse under B2, a
MECHANISM rewiring. That is backwards for a world model, and `ConstraintHead` says why:
`INC_SHIFT = 4.0` makes `softplus(0 - 4) = 0.0181`, which sits squarely inside generator A's
true monthly creep (F 0.022, D 0.015, P 0.011, S 0.018). The *architecture* supplies the rate;
the network only learns a small correction around it, and nothing can re-infer the rate when
the generator's rates move.

The generator also tells us the rate is identifiable. Every ratchet increment is proportional to
one scalar: `drive = susc_eff * (0.6*A + 0.4*C)` with `susc_eff = susceptibility*(0.8+0.4*age)`,
then `dF = 0.022*drive`, `dD = 0.015*drive`, `dS = 0.018*susc_eff*A`, `dM = 0.05*F*C`. So a
patient's pace is recoverable from their own observed increments, and the per-field ratios are
fixed. Four ways to exploit that:

  rate_anchor   Non-parametric. Measure the patient's REALISED per-field creep over the observed
                window and multiply the head's increment by it. gain = 1 reproduces the base
                model exactly, so this is a strict generalisation. Under a coefficient shift the
                measured gain moves with the generator, for free, with no retraining.

  hazard_mono   Structural. Factor the increment as `pace(z) x shape(...)` with `pace > 0`
                learned from the observed window and the INC_SHIFT prior removed, so scale and
                shape are separated. Makes the forecast PROVABLY monotone in the inferred pace:
                a patient judged faster can never be predicted slower. This is the same move the
                repo made for time-monotonicity, applied to the hidden parameter.

  time_warp     Reparameterise the clock. Integrate the latent ODE in "disease time"
                tau = pace * t rather than months, and embed tau instead of t. A global rate
                rescaling (exactly what B1 is) leaves the trajectory unchanged in tau-space and
                only changes the t -> tau map, so the learned dynamics are rate-invariant by
                construction rather than by fitting.

  ude_hybrid    Universal differential equation. A differentiable re-implementation of the
                generator's FUNCTIONAL FORM with LEARNABLE coefficients (initialised generically
                at 0.02, NOT at the true values) supplies a physics increment; the network adds a
                residual on top. Tests directly how much knowing the form is worth.
                *** This model is given privileged structural information the other 39 are not.
                It is an upper bound on what mechanism knowledge buys, not a fair competitor. ***
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from models.neural_pde import _LatentPDE
from models.composed import GraphRefine
from util import (ConstraintHead, STATE_DIM, CTX_DIM, RAW_DIM, RATCHETS, F, D, S, P, A, C, M,
                  SinusoidalPosEmb, mlp, state_loss, scale_dim)

BASE_INC = 0.0181          # softplus(-INC_SHIFT): the rate the base architecture hard-codes


# =========================================================================================
# Heads
# =========================================================================================
class GainHead(ConstraintHead):
    """ConstraintHead whose ratchet increment is scaled by a per-patient, per-field gain.

    `gain` is set by the model before each rollout (broadcast [B, 8]); gain == 1 everywhere
    reproduces ConstraintHead exactly, so anything built on this is a strict generalisation of
    the base model. Monotonicity is preserved for any gain >= 0.
    """

    def __init__(self, couple_m: bool = True, use_shift: bool = True):
        super().__init__(couple_m)
        self.use_shift = use_shift
        self.gain = None                                   # plain tensor, not a parameter

    def forward(self, raw, prev_x, is_ercp):
        raw_fields = raw[..., :STATE_DIM]
        raw_relief = raw[..., STATE_DIM]
        shift = self.INC_SHIFT if self.use_shift else 0.0
        inc = Fn.softplus(raw_fields - shift)
        if self.gain is not None:
            inc = inc * self.gain
        if self.couple_m:
            fc = (prev_x[..., F] * prev_x[..., C]).unsqueeze(-1)
            inc = inc * (1.0 - self.m_onehot + self.m_onehot * fc)
        val = torch.sigmoid(raw_fields)
        nxt = torch.where(self.free_mask, val, prev_x + inc)
        relief = Fn.softplus(raw_relief) * is_ercp
        nxt = nxt - relief.unsqueeze(-1) * self.s_onehot
        return torch.minimum(torch.clamp(nxt, min=0.0), self.fmax)


class PhysicsHead(ConstraintHead):
    """ConstraintHead that adds a non-negative PHYSICS increment to the learned residual.

    ratchets : next = prev + phys_inc + softplus(raw - 4)      (both terms >= 0 -> monotone)
    free     : next = sigmoid(raw + phys_logit)                (physics as a logit offset)
    With phys_inc = 0 and phys_logit = 0 this is exactly ConstraintHead.
    """

    def __init__(self, couple_m: bool = True):
        super().__init__(couple_m)
        self.phys_inc = None
        self.phys_logit = None

    def forward(self, raw, prev_x, is_ercp):
        raw_fields = raw[..., :STATE_DIM]
        raw_relief = raw[..., STATE_DIM]
        inc = Fn.softplus(raw_fields - self.INC_SHIFT)
        if self.phys_inc is not None:
            inc = inc + self.phys_inc.clamp(min=0.0)
        if self.couple_m:
            fc = (prev_x[..., F] * prev_x[..., C]).unsqueeze(-1)
            inc = inc * (1.0 - self.m_onehot + self.m_onehot * fc)
        logit = raw_fields if self.phys_logit is None else raw_fields + self.phys_logit
        val = torch.sigmoid(logit)
        nxt = torch.where(self.free_mask, val, prev_x + inc)
        relief = Fn.softplus(raw_relief) * is_ercp
        nxt = nxt - relief.unsqueeze(-1) * self.s_onehot
        return torch.minimum(torch.clamp(nxt, min=0.0), self.fmax)


# =========================================================================================
# Shared base: exactly `gnn_leworld_meta_ode`, with hooks
# =========================================================================================
class _ODEBase(_LatentPDE):
    """The sweep-best config: LeWorld state encoder -> GRU -> GraphRefine -> meta task code ->
    drift-only latent ODE (no diffusion) -> decode -> ConstraintHead."""

    def __init__(self, T, scale=1.0, extra_in=0, **kw):
        super().__init__(T, drift_kind="mlp", leworld_enc=True, use_task=True,
                         use_diffusion=False, scale=scale, **kw)
        self.graph = GraphRefine(self.dz)
        if extra_in:                                        # widen the drift net's input
            din = self.dz + CTX_DIM + 16 + self.dtask + extra_in
            self.reaction = mlp([din, self.reaction[0].out_features,
                                 self.reaction[0].out_features, self.dz])
        self.extra_in = extra_in

    def _encode(self, x_obs, ctx_obs):
        return self.graph(_LatentPDE._encode(self, x_obs, ctx_obs))


# =========================================================================================
# 1. rate_anchor
# =========================================================================================
class RateAnchor(_ODEBase):
    name = "rate_anchor"

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale, extra_in=len(RATCHETS))
        self.head = GainHead()

    @staticmethod
    def _realised_gain(x_obs):
        """Per-field realised creep over the observed window, as a multiple of the rate the base
        architecture hard-codes. Uses only observed increments -- no learned parameters."""
        if x_obs.shape[1] < 2:
            return None
        dx = (x_obs[:, 1:] - x_obs[:, :-1]).clamp(min=0.0)   # [B, K, 8]
        r = dx.mean(1)                                       # [B, 8]
        gain = torch.ones_like(r)
        idx = torch.tensor(RATCHETS, device=x_obs.device)
        gain[:, idx] = (r[:, idx] / BASE_INC).clamp(0.05, 20.0)
        return gain

    def _encode(self, x_obs, ctx_obs):
        g = self._realised_gain(x_obs)
        self.head.gain = g                                   # consumed by GainHead
        idx = torch.tensor(RATCHETS, device=x_obs.device)
        self._gfeat = torch.zeros(x_obs.shape[0], len(RATCHETS), device=x_obs.device) \
            if g is None else torch.log(g[:, idx])
        return super()._encode(x_obs, ctx_obs)

    def _field(self, z, ctx_t, t, z_task=None):
        te = self.temb(torch.full((z.shape[0],), float(t), device=z.device))
        parts = [z, ctx_t, te, z_task, self._gfeat]
        return self.reaction(torch.cat(parts, -1))


# =========================================================================================
# 2. hazard_mono
# =========================================================================================
class HazardMono(_ODEBase):
    name = "hazard_mono"

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale, extra_in=1)
        self.head = GainHead(use_shift=False)                # pace carries the scale now
        self.pace = nn.Linear(self.dz, 1)
        # With INC_SHIFT removed, pace alone sets the increment scale. Bias-init it so that
        # pace * softplus(0) == BASE_INC, i.e. the model STARTS at the same near-stationary
        # ratchet the base architecture hard-codes. Without this the increments begin at ~0.5/mo
        # and the ratchets saturate in three steps (the exact failure INC_SHIFT exists to avoid).
        nn.init.zeros_(self.pace.weight)
        with torch.no_grad():
            p0 = BASE_INC / Fn.softplus(torch.zeros(1))
            self.pace.bias.fill_(float(torch.log(torch.expm1(p0))))

    def _encode(self, x_obs, ctx_obs):
        z0 = super()._encode(x_obs, ctx_obs)
        s = Fn.softplus(self.pace(z0)) + 1e-3                # [B,1], strictly positive
        gain = torch.ones(z0.shape[0], STATE_DIM, device=z0.device)
        idx = torch.tensor(RATCHETS, device=z0.device)
        gain = gain.clone()
        gain[:, idx] = s.expand(-1, len(RATCHETS))
        self.head.gain = gain
        self._pace_feat = torch.log(s)
        return z0

    def _field(self, z, ctx_t, t, z_task=None):
        te = self.temb(torch.full((z.shape[0],), float(t), device=z.device))
        return self.reaction(torch.cat([z, ctx_t, te, z_task, self._pace_feat], -1))


# =========================================================================================
# 3. time_warp
# =========================================================================================
class TimeWarp(_ODEBase):
    name = "time_warp"

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale)
        self.pace = nn.Linear(self.dz, 1)

    def _encode(self, x_obs, ctx_obs):
        z0 = super()._encode(x_obs, ctx_obs)
        # pace ~ 1 at init so the model starts as the base ODE (dt = 1 month)
        self._pace = Fn.softplus(self.pace(z0) + 0.5413) + 1e-3
        self._tau0 = 0.0
        return z0

    def _field_tau(self, z, ctx_t, tau, z_task):
        te = self.temb(tau.squeeze(-1))                      # embed DISEASE time, not months
        return self.reaction(torch.cat([z, ctx_t, te, z_task], -1))

    def _integrate(self, z0, x_base, ctx, ercp, K, end, z_task=None):
        z, prev = z0, x_base
        s = self._pace                                       # [B,1]
        tau = s * float(K)
        outs = []
        for t in range(K, end - 1):
            z = z + s * self._field_tau(z, ctx[:, t + 1], tau, z_task)   # dtau = pace * dt
            tau = tau + s
            prev = self.head(self.dec(z), prev, ercp[:, t + 1])
            outs.append(prev)
        return torch.stack(outs, 1)


# =========================================================================================
# 4. ude_hybrid  (privileged: knows the generator's functional form)
# =========================================================================================
class UDEHybrid(_ODEBase):
    name = "ude_hybrid"

    # generic init -- deliberately NOT the generator's true coefficients
    INIT = 0.02

    def __init__(self, T, scale=1.0):
        super().__init__(T, scale=scale, extra_in=1)
        self.head = PhysicsHead()
        self.pace = nn.Linear(self.dz, 1)
        # learnable mechanism coefficients (the FORM is given, the RATES are not)
        self.c_ratchet = nn.Parameter(torch.full((4,), self.INIT))    # F, D, P, S
        self.c_m = nn.Parameter(torch.tensor(self.INIT))
        self.c_pdrive = nn.Parameter(torch.tensor(0.5))               # P's extra F-drive
        self.w_ac = nn.Parameter(torch.tensor([0.5, 0.5]))            # drive = w_a*A + w_c*C
        self.c_rev = nn.Parameter(torch.tensor([0.5, 0.4]))           # A, C reversion rates
        self.c_base = nn.Parameter(torch.tensor([0.15, 0.10]))        # set point = b0 + b1*class
        self.c_supp = nn.Parameter(torch.tensor(0.6))                 # treatment knock-down
        self.c_flare = nn.Parameter(torch.tensor(0.5))

    def _encode(self, x_obs, ctx_obs):
        z0 = super()._encode(x_obs, ctx_obs)
        self._pace = Fn.softplus(self.pace(z0) + 0.5413) + 1e-3       # ~1 at init
        self._pace_feat = torch.log(self._pace)
        return z0

    def _physics(self, prev, ctx_t):
        """Differentiable, batched re-implementation of Data_Generator._advance's deterministic
        skeleton. Returns (nonneg ratchet increments [B,8], logit offsets for the free fields)."""
        pace = self._pace                                             # [B,1]
        dclass = ctx_t[:, :3] @ torch.arange(3.0, device=prev.device)
        base = self.c_base[0] + self.c_base[1] * dclass                # [B]
        supp = self.c_supp * ctx_t[:, 6] * ctx_t[:, 5]                 # on_udca AND responder
        a_prev, c_prev = prev[:, A], prev[:, C]
        flare = prev[:, 7] * 0.4                                       # deterministic decay part

        drive = pace.squeeze(-1) * (self.w_ac[0] * a_prev + self.w_ac[1] * c_prev)
        inc = torch.zeros_like(prev)
        inc[:, F] = self.c_ratchet[0].abs() * drive
        inc[:, D] = self.c_ratchet[1].abs() * drive
        inc[:, P] = self.c_ratchet[2].abs() * (drive + self.c_pdrive * prev[:, F])
        inc[:, S] = self.c_ratchet[3].abs() * pace.squeeze(-1) * a_prev
        inc[:, M] = self.c_m.abs() * prev[:, F] * c_prev

        # free fields: physics predicts a target level; pass it in as a logit offset
        a_set = base * (1 - supp)
        a_next = (a_prev + self.c_rev[0] * (a_set - a_prev) + self.c_flare * flare).clamp(1e-4, 1 - 1e-4)
        c_next = (c_prev + self.c_rev[1] * (a_set - c_prev) + self.c_flare * flare).clamp(1e-4, 1 - 1e-4)
        f_next = flare.clamp(1e-4, 1 - 1e-4)
        logit = torch.zeros_like(prev)
        logit[:, A] = torch.log(a_next / (1 - a_next))
        logit[:, C] = torch.log(c_next / (1 - c_next))
        logit[:, 7] = torch.log(f_next / (1 - f_next))
        return inc, logit

    def _field(self, z, ctx_t, t, z_task=None):
        te = self.temb(torch.full((z.shape[0],), float(t), device=z.device))
        return self.reaction(torch.cat([z, ctx_t, te, z_task, self._pace_feat], -1))

    def _integrate(self, z0, x_base, ctx, ercp, K, end, z_task=None):
        z, prev = z0, x_base
        outs = []
        for t in range(K, end - 1):
            z = z + self._field(z, ctx[:, t + 1], t, z_task)
            pi, pl = self._physics(prev, ctx[:, t + 1])
            self.head.phys_inc, self.head.phys_logit = pi, pl
            prev = self.head(self.dec(z), prev, ercp[:, t + 1])
            outs.append(prev)
        return torch.stack(outs, 1)
