"""
A compact, from-scratch Kolmogorov-Arnold Network (KAN) layer (efficient B-spline form).

A KAN replaces the affine weight + fixed nonlinearity of an MLP edge with a LEARNABLE univariate
function on every edge:

    y_o = sum_i  phi_{o,i}(x_i),   phi_{o,i}(x) = w^base_{o,i}*silu(x) + sum_g c_{o,i,g} B_g(x)

where {B_g} are order-k B-spline bases on a fixed grid. The learned edge functions are the reason
KANs are interpretable: each phi can be plotted / read as a 1-D response curve, so "which input
drives the output, and through what shape" is directly inspectable (unlike an MLP's entangled
weights).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.grid_size, self.spline_order = grid_size, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid.expand(in_dim, -1).contiguous())   # [in, G+2k+1]
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.spline_weight = nn.Parameter(torch.empty(out_dim, in_dim, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1)

    def b_splines(self, x):
        # x: [N, in] -> bases [N, in, G+k]  (Cox-de Boor recursion)
        grid = self.grid
        x = x.unsqueeze(-1)
        b = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            b = left * b[:, :, :-1] + right * b[:, :, 1:]
        return b

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_dim)
        base = F.silu(x) @ self.base_weight.t()                       # [N, out]
        spline = torch.einsum("nig,oig->no", self.b_splines(x), self.spline_weight)
        return (base + spline).reshape(*shape[:-1], self.out_dim)
