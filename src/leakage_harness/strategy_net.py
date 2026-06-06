"""StrategyNet: the per-timestep position emitter shared across spikes.

Architecture is deliberately non-trivial (not a single nn.Linear) so that
Spike 6's planted-signal recovery is a real test of end-to-end learning,
not a one-line linear regression.

Shape contract:
    Input:  x of shape [T, S, F]  (time, stock, feature)
    Output: positions of shape [T, S], cross-sectionally mean-zero
            (market-neutral by construction at each timestep).
"""

from __future__ import annotations

import torch
from torch import nn


class StrategyNet(nn.Module):
    """Per-stock MLP head plus cross-sectional de-mean (market-neutral)."""

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [T, S, F] -> [T, S]
        z = self.net(x).squeeze(-1)
        return z - z.mean(dim=-1, keepdim=True)
