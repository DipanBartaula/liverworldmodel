"""
Common interface every LiverModel model implements, so `training.py` and `eval.py` can drive
any of them uniformly.

Contract:
    training_step(batch) -> {"loss": tensor, ...aux}   model-specific objective over the batch
    rollout(x, ctx, ercp, K) -> pred [B, T, 8]         observe months 0..K, predict K+1..T-1
                                                        (observed prefix is copied into pred)

Every model routes its final per-step output through the shared ConstraintHead (self.head),
so the one-directional / bounds / coupling constraints hold BY CONSTRUCTION for all of them.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from util import ConstraintHead, STATE_DIM


class BaseModel(nn.Module):
    name = "base"

    def __init__(self, T: int, couple_m: bool = True):
        super().__init__()
        self.T = T
        self.head = ConstraintHead(couple_m=couple_m)

    # -- to implement --------------------------------------------------------------------
    def training_step(self, batch: dict) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def rollout(self, x, ctx, ercp, K: int) -> torch.Tensor:
        raise NotImplementedError

    # -- shared helpers ------------------------------------------------------------------
    def apply_head(self, raw, prev_x, ercp_t):
        return self.head(raw, prev_x, ercp_t)

    def _init_pred(self, x, K):
        """Allocate a prediction buffer with the observed prefix 0..K copied in."""
        pred = torch.zeros_like(x)
        pred[:, : K + 1] = x[:, : K + 1]
        return pred
