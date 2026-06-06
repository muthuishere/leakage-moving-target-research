"""Differentiable backtest primitive.

Trades-and-holds convention: position at time t (decided from features at t)
earns the return from t -> t+1. Therefore the per-step P&L vector has length
T-1.

Pure torch, fully autograd-friendly. Used by Spike 5 (gradient flow) and
Spike 6 (planted-signal recovery).
"""

from __future__ import annotations

import torch


def differentiable_backtest(positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Compute portfolio P&L per timestep, end-to-end differentiable.

    Args:
        positions: Tensor of shape [T, S]. positions[t, s] is the position in
            stock s decided at time t. Should typically be market-neutral
            (zero cross-sectional mean) but that is the caller's choice.
        returns:   Tensor of shape [T, S]. returns[t, s] is the simple return
            of stock s from t-1 to t.

    Returns:
        pnl: Tensor of shape [T-1]. pnl[t] = sum_s positions[t, s] * returns[t+1, s].
    """
    if positions.dim() != 2 or returns.dim() != 2:
        raise ValueError(
            f"expected 2-D tensors, got positions {tuple(positions.shape)} "
            f"and returns {tuple(returns.shape)}"
        )
    if positions.shape != returns.shape:
        raise ValueError(
            f"positions {tuple(positions.shape)} != returns {tuple(returns.shape)}"
        )
    return (positions[:-1] * returns[1:]).sum(dim=-1)
