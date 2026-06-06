"""Loss functions for differentiable backtesting.

Three losses → three personalities (per Spike 7):
- `negative_mean_pnl`        — return-max
- `negative_sharpe`          — risk-adjusted-return-max
- `drawdown_aware`           — drawdown-averse

`max_drawdown` is the shared per-step P&L drawdown primitive, fully
differentiable in torch via `cummax`.
"""

from __future__ import annotations

import math

import torch


def negative_mean_pnl(pnl: torch.Tensor) -> torch.Tensor:
    """Loss equivalent to maximising mean P&L."""
    return -pnl.mean()


def max_drawdown(pnl: torch.Tensor) -> torch.Tensor:
    """Max drawdown of the cumulative P&L (equity) curve.

    equity[t] = sum_{i<=t} pnl[i]
    drawdown[t] = max(equity[0..t]) - equity[t]
    returns max(drawdown) — a non-negative scalar tensor.

    Differentiable: `cummax` carries gradients through the max op via the
    argmax-of-prefix selector; `pnl.cumsum` is a linear combination.
    """
    if pnl.numel() == 0:
        return torch.zeros((), dtype=pnl.dtype, device=pnl.device)
    equity = pnl.cumsum(dim=0)
    peak = equity.cummax(dim=0).values
    return (peak - equity).max()


def negative_sharpe(
    pnl: torch.Tensor, periods_per_year: int = 252, eps: float = 1e-8
) -> torch.Tensor:
    """Loss = -annualised Sharpe. eps guards against zero-variance P&L."""
    mu = pnl.mean()
    sigma = pnl.std(unbiased=False)
    sharpe = mu / (sigma + eps) * math.sqrt(periods_per_year)
    return -sharpe


def drawdown_aware(pnl: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Loss = -mean(pnl) + lam * max_drawdown(cumsum(pnl)).

    lam balances return-pursuit against drawdown control. Default 1.0 is
    a reasonable starting point on per-step P&L of order 1e-2; tune to
    move the strategy on the comparison plot.
    """
    return -pnl.mean() + lam * max_drawdown(pnl)

def sharpe_with_position_floor(
    pnl: torch.Tensor,
    pos: torch.Tensor,
    floor: float = 0.05,
    alpha: float = 0.5,
    periods_per_year: int = 252,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Loss = -Sharpe + alpha * ReLU(floor - mean|pos|).

    The position-floor penalty kicks in when the model collapses to
    near-zero positions (the negative_sharpe scale-invariance pathology).
    Unlike pnl-only floor variants, this directly sees position magnitude,
    so the gradient pushes positions away from zero even when pnl is
    momentarily zero.

    floor=0.05: each stock should carry at least 5% absolute weight on average.
    alpha=5.0: tuned so the penalty dominates loss when positions are degenerate.
    """
    mu = pnl.mean()
    sigma = pnl.std(unbiased=False)
    sharpe = mu / (sigma + eps) * math.sqrt(periods_per_year)
    mean_abs_pos = pos.abs().mean()
    penalty = alpha * torch.relu(
        torch.tensor(floor, device=pos.device, dtype=pos.dtype) - mean_abs_pos
    )
    return -sharpe + penalty

