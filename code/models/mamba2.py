"""
Mamba-2 -- a selective state-space (SSD) sequence block, in plain PyTorch.

The Mamba-2 layer is a linear-time state-space model whose transition is *input-dependent*
(selective) but whose state matrix is a SCALAR per head -- the "state-space duality" (SSD)
restriction that makes Mamba-2 both a linear RNN and a masked attention. Per head h:

    h_t = exp(dt_t * A_h) * h_{t-1} + dt_t * (x_t (x) B_t)          h_t : [P, N] outer product
    y_t = h_t @ C_t + D_h * x_t

with dt_t = softplus(linear(u_t) + dt_bias) the *selection* (how much of step t to admit) and
A_h < 0 the per-head decay. B_t, C_t are input-dependent (data-dependent kernel), which is what
separates this from a linear-time-invariant SSM (S4). Around the scan sit Mamba-2's other
ingredients: one fused input projection, a short causal depthwise conv over (x, B, C), a SiLU
gate z, and a gated RMSNorm before the output projection.

The recurrence is run sequentially. The official implementation uses a chunked matmul form for
GPU throughput; on CPU at T=60 the sequential scan is the same arithmetic, needs no chunk
padding, and stays numerically exact (no cumulative-log rescaling), so it is what we use.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Mamba2Block(nn.Module):
    """One Mamba-2 SSD block: d_model -> d_model (pre-norm residual is applied by the caller).

    nheads must divide d_inner = expand * d_model; ngroups is fixed at 1 (B, C shared across
    heads), as in the reference Mamba-2 configuration.
    """

    def __init__(self, d_model, d_state=16, expand=2, nheads=4, d_conv=4):
        super().__init__()
        d_inner = expand * d_model
        nheads = max(1, min(nheads, d_inner))
        while d_inner % nheads:                       # keep heads exact for any swept width
            nheads -= 1
        self.d_inner, self.d_state, self.nheads = d_inner, d_state, nheads
        self.dhead = d_inner // nheads
        self.d_conv = d_conv

        # z (gate) | x | B | C | dt  -- one fused projection, as in Mamba-2
        self.in_proj = nn.Linear(d_model, 2 * d_inner + 2 * d_state + nheads, bias=False)
        conv_ch = d_inner + 2 * d_state
        self.conv = nn.Conv1d(conv_ch, conv_ch, d_conv, groups=conv_ch, padding=d_conv - 1)
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(torch.full((nheads,), 0.05))))
        # A_h = -exp(A_log_h): initialised to a spread of per-head decay time-scales (1..nheads)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, nheads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(nheads))
        self.norm = RMSNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _scan(self, x, dt, B, C):
        """x:[b,L,H,P]  dt:[b,L,H]  B,C:[b,L,N] -> y:[b,L,H,P]."""
        A = -torch.exp(self.A_log)                                  # [H], negative
        decay = torch.exp(dt * A)                                   # [b,L,H]
        h = x.new_zeros(x.shape[0], self.nheads, self.dhead, self.d_state)
        ys = []
        for t in range(x.shape[1]):
            # dt-scaled outer product x_t (x) B_t is the input written into the state
            dBx = torch.einsum("bhp,bn->bhpn", dt[:, t].unsqueeze(-1) * x[:, t], B[:, t])
            h = decay[:, t].unsqueeze(-1).unsqueeze(-1) * h + dBx
            ys.append(torch.einsum("bhpn,bn->bhp", h, C[:, t]) + self.D.unsqueeze(-1) * x[:, t])
        return torch.stack(ys, 1)

    def forward(self, u):                                           # u: [b, L, d_model]
        b, L, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        # short causal depthwise conv over (x, B, C) -- local mixing before the scan
        xBC = F.silu(self.conv(xBC.transpose(1, 2))[..., :L].transpose(1, 2))
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)                          # selection
        y = self._scan(x.view(b, L, self.nheads, self.dhead), dt, B, C)
        y = self.norm(y.reshape(b, L, self.d_inner) * F.silu(z))    # gated RMSNorm
        return self.out_proj(y)
