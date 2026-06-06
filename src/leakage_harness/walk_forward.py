"""Walk-forward training + backtest harness.

For each holdout day t (in `[n_train, n_train + n_holdout)`):
  - Retrain every `retrain_every` days using `features[t - n_train : t]`
    and `returns[t - n_train : t]`. Default: every 5 days.
  - Predict positions at day t by feeding `features[t:t+1]` to the
    most-recent model.
  - Realise per-step P&L = `positions[t] · returns[t+1]`.

Returns per-step P&L + final model + summary metrics, including the
**normalised DD/return** ratio (L7 carry-forward from Spike 7) so
strategies of different absolute magnitudes can be compared honestly.

Deterministic: re-seeded before every training call, so two runs with
the same `(features, returns, seed)` produce bit-exact identical P&L
arrays. This is the primitive the S9 reproducibility leakage test
exercises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from leakage_harness.backtest import differentiable_backtest
from leakage_harness.losses import max_drawdown as md_torch
from leakage_harness.strategy_net import StrategyNet


class LinearStrategy(nn.Module):
    """Pure-linear cross-sectional model: positions = (features @ W) - mean.

    No hidden layer, no nonlinearity. Simplest expressive model that can
    extract time-edge from features — sanity floor for any claim that an
    MLP is doing real work. If LinearStrategy can't find an edge on
    z-scored features, StrategyNet won't either."""

    def __init__(self, n_features: int):
        super().__init__()
        self.lin = nn.Linear(n_features, 1, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [T, S, F] -> raw: [T, S]; cross-sectionally de-mean.
        raw = self.lin(features).squeeze(-1)
        return raw - raw.mean(dim=-1, keepdim=True)


@dataclass
class WalkForwardResult:
    pnl: torch.Tensor             # [n_holdout]
    positions: torch.Tensor       # [n_holdout, S]
    sharpe: float
    max_drawdown: float
    mean_pnl: float
    turnover_avg: float           # avg |Δpositions| / |positions|
    norm_dd_over_return: float    # max_dd / |mean_pnl| (inf if mean_pnl ~ 0)
    final_model: StrategyNet | LinearStrategy | list[StrategyNet]


# ----- internals -----


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _call_loss(loss_fn, pnl, pos):
    """Loss functions historically took only `pnl`. Newer position-aware
    losses declare a `pos` parameter. Inspect the signature to dispatch
    correctly — try/except on TypeError is unsafe because Python binds
    extra positional args to defaulted kwargs (e.g. `periods_per_year`)
    instead of raising."""
    import inspect
    sig = inspect.signature(loss_fn)
    if "pos" in sig.parameters:
        return loss_fn(pnl, pos=pos)
    return loss_fn(pnl)


def _train_one(
    features: torch.Tensor,
    returns: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    hp: dict,
    seed: int,
) -> StrategyNet:
    _seed_all(seed)
    model = StrategyNet(
        n_features=features.shape[-1],
        hidden=hp.get("hidden", 16),
        dropout=hp.get("dropout", 0.2),
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=hp.get("lr", 3e-3),
        weight_decay=hp.get("weight_decay", 1e-2),
    )
    # Ticker-dropout (L43 identity-stripping): on each step, mask out a
    # random ~p fraction of tickers and re-de-mean so the masked positions
    # don't shift the zero-sum invariant. Generator is seeded off the
    # outer training seed so dropout is reproducible per (window, seed)
    # — bit_exact_reproducibility test still holds.
    ticker_dropout = float(hp.get("ticker_dropout", 0.0))
    dropout_gen = torch.Generator().manual_seed(seed + 1) if ticker_dropout > 0 else None
    for _ in range(hp.get("n_steps", 200)):
        model.train()
        opt.zero_grad()
        pos = model(features)
        if ticker_dropout > 0:
            mask = (torch.rand(pos.shape[-1], generator=dropout_gen) > ticker_dropout).float()
            pos = pos * mask
            pos = pos - pos.mean(dim=-1, keepdim=True)
        pnl = differentiable_backtest(pos, returns)
        loss = _call_loss(loss_fn, pnl, pos)
        loss.backward()
        opt.step()
    return model


def annualised_sharpe(pnl: torch.Tensor, periods_per_year: int = 252) -> float:
    if pnl.numel() < 2:
        return 0.0
    mu = pnl.mean()
    sigma = pnl.std(unbiased=False)
    if sigma.item() == 0.0:
        return 0.0
    return float((mu / sigma * math.sqrt(periods_per_year)).item())


def _avg_turnover(positions: torch.Tensor) -> float:
    if positions.shape[0] < 2:
        return 0.0
    diff = (positions[1:] - positions[:-1]).abs().sum(dim=-1)
    gross = positions[1:].abs().sum(dim=-1).clamp(min=1e-9)
    return float((diff / gross).mean().item())


# ----- public -----


def walk_forward(
    features: torch.Tensor,  # [T, S, F]
    returns: torch.Tensor,   # [T, S]
    *,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int = 5,
    seed: int = 20260514,
) -> WalkForwardResult:
    T = features.shape[0]
    max_holdout = T - n_train - 1
    if max_holdout < 1:
        raise ValueError(
            f"need T >= n_train + 2 for at least one holdout step; "
            f"have T={T}, n_train={n_train}"
        )
    n_holdout = min(n_holdout, max_holdout)

    pnl_list: list[torch.Tensor] = []
    pos_list: list[torch.Tensor] = []
    model: StrategyNet | None = None
    last_train_t = -1

    for offset in range(n_holdout):
        t = n_train + offset
        if model is None or (t - last_train_t) >= retrain_every:
            model = _train_one(
                features[t - n_train : t],
                returns[t - n_train : t],
                loss_fn,
                hp=hp,
                seed=seed,
            )
            last_train_t = t

        model.eval()
        with torch.no_grad():
            pos_t = model(features[t : t + 1])[0]  # [S]
        pos_list.append(pos_t)
        pnl_list.append((pos_t * returns[t + 1]).sum())

    pnl = torch.stack(pnl_list)
    positions = torch.stack(pos_list)

    sharpe = annualised_sharpe(pnl)
    max_dd = float(md_torch(pnl).item())
    mean_pnl = float(pnl.mean().item())
    norm_dd = max_dd / abs(mean_pnl) if abs(mean_pnl) > 1e-9 else float("inf")
    turnover = _avg_turnover(positions)

    assert model is not None
    return WalkForwardResult(
        pnl=pnl,
        positions=positions,
        sharpe=sharpe,
        max_drawdown=max_dd,
        mean_pnl=mean_pnl,
        turnover_avg=turnover,
        norm_dd_over_return=norm_dd,
        final_model=model,
    )


def _train_one_per_stock(
    features_s: torch.Tensor,   # [n_train, F]
    returns_s: torch.Tensor,    # [n_train]
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    hp: dict,
    seed: int,
) -> StrategyNet:
    """Train a StrategyNet on one stock's window.

    Wrapped as a (T, 1, F) -> (T, 1) call so StrategyNet's cross-sectional
    de-mean is a no-op (mean of one element is itself ⇒ output is always 0).
    To make the model produce a useful scalar position we bypass the
    de-mean by reading the pre-demean activation. Done here by replicating
    `_train_one`'s body but reading `model.net(x).squeeze(-1)` directly."""
    _seed_all(seed)
    model = StrategyNet(
        n_features=features_s.shape[-1],
        hidden=hp.get("hidden", 8),
        dropout=hp.get("dropout", 0.2),
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=hp.get("lr", 3e-3),
        weight_decay=hp.get("weight_decay", 1e-2),
    )
    x = features_s.unsqueeze(1)  # [T, 1, F]
    for _ in range(hp.get("n_steps", 100)):
        model.train()
        opt.zero_grad()
        pos = model.net(x).squeeze(-1).squeeze(-1)  # [T]
        # Mirror differentiable_backtest's t/t+1 alignment: position at t
        # earns return at t+1. The unlagged form would train the model
        # against the wrong direction.
        pnl = pos[:-1] * returns_s[1:]
        loss = _call_loss(loss_fn, pnl, pos[:-1])
        loss.backward()
        opt.step()
    return model


def single_stock_walk_forward(
    features: torch.Tensor,   # [T, S, F]
    returns: torch.Tensor,    # [T, S]
    *,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int = 5,
    seed: int = 20260516,
) -> WalkForwardResult:
    """Per-stock independent walk-forward — one StrategyNet per ticker.

    Each model sees only its own stock's features and trains on its own
    returns. Positions are STACKED (no cross-sectional de-mean) — the
    whole point is to eliminate the cross-stock channel and test whether
    any individual stock has time-edge at all.

    Returns the same `WalkForwardResult` shape as `walk_forward` so the
    existing leakage tests work unchanged.
    """
    T, S, _ = features.shape
    max_holdout = T - n_train - 1
    if max_holdout < 1:
        raise ValueError(
            f"need T >= n_train + 2 for at least one holdout step; "
            f"have T={T}, n_train={n_train}"
        )
    n_holdout = min(n_holdout, max_holdout)

    pnl_list: list[torch.Tensor] = []
    pos_list: list[torch.Tensor] = []
    models: list[StrategyNet] = [None] * S  # type: ignore[list-item]
    last_train_t = -1

    for offset in range(n_holdout):
        t = n_train + offset
        if models[0] is None or (t - last_train_t) >= retrain_every:
            for s in range(S):
                # Per-stock seed offset keeps training deterministic per
                # ticker while staying reproducible run-to-run.
                models[s] = _train_one_per_stock(
                    features[t - n_train : t, s, :],
                    returns[t - n_train : t, s],
                    loss_fn,
                    hp=hp,
                    seed=seed + s,
                )
            last_train_t = t

        pos_t = torch.empty(S)
        for s in range(S):
            models[s].eval()
            with torch.no_grad():
                pos_t[s] = models[s].net(
                    features[t : t + 1, s, :].unsqueeze(1)
                ).squeeze()
        pos_list.append(pos_t)
        pnl_list.append((pos_t * returns[t + 1]).sum())

    pnl = torch.stack(pnl_list)
    positions = torch.stack(pos_list)

    sharpe = annualised_sharpe(pnl)
    max_dd = float(md_torch(pnl).item())
    mean_pnl = float(pnl.mean().item())
    norm_dd = max_dd / abs(mean_pnl) if abs(mean_pnl) > 1e-9 else float("inf")
    turnover = _avg_turnover(positions)

    return WalkForwardResult(
        pnl=pnl,
        positions=positions,
        sharpe=sharpe,
        max_drawdown=max_dd,
        mean_pnl=mean_pnl,
        turnover_avg=turnover,
        norm_dd_over_return=norm_dd,
        final_model=models,
    )


def _train_one_linear(
    features: torch.Tensor,
    returns: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    hp: dict,
    seed: int,
) -> LinearStrategy:
    _seed_all(seed)
    model = LinearStrategy(n_features=features.shape[-1])
    opt = torch.optim.Adam(
        model.parameters(),
        lr=hp.get("lr", 3e-3),
        weight_decay=hp.get("weight_decay", 1e-2),
    )
    for _ in range(hp.get("n_steps", 200)):
        model.train()
        opt.zero_grad()
        pos = model(features)
        pnl = differentiable_backtest(pos, returns)
        loss = _call_loss(loss_fn, pnl, pos)
        loss.backward()
        opt.step()
    return model


def linear_walk_forward(
    features: torch.Tensor,  # [T, S, F]
    returns: torch.Tensor,   # [T, S]
    *,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int = 5,
    seed: int = 20260517,
) -> WalkForwardResult:
    """Walk-forward harness using `LinearStrategy` instead of `StrategyNet`.

    Identical to `walk_forward` in semantics and output shape; `hp` keys
    `hidden` and `dropout` are ignored (linear has neither). Used by the
    `linear_baseline` runner as the sanity-floor model.
    """
    T = features.shape[0]
    max_holdout = T - n_train - 1
    if max_holdout < 1:
        raise ValueError(
            f"need T >= n_train + 2 for at least one holdout step; "
            f"have T={T}, n_train={n_train}"
        )
    n_holdout = min(n_holdout, max_holdout)

    pnl_list: list[torch.Tensor] = []
    pos_list: list[torch.Tensor] = []
    model: LinearStrategy | None = None
    last_train_t = -1

    for offset in range(n_holdout):
        t = n_train + offset
        if model is None or (t - last_train_t) >= retrain_every:
            model = _train_one_linear(
                features[t - n_train : t],
                returns[t - n_train : t],
                loss_fn,
                hp=hp,
                seed=seed,
            )
            last_train_t = t

        model.eval()
        with torch.no_grad():
            pos_t = model(features[t : t + 1])[0]  # [S]
        pos_list.append(pos_t)
        pnl_list.append((pos_t * returns[t + 1]).sum())

    pnl = torch.stack(pnl_list)
    positions = torch.stack(pos_list)

    sharpe = annualised_sharpe(pnl)
    max_dd = float(md_torch(pnl).item())
    mean_pnl = float(pnl.mean().item())
    norm_dd = max_dd / abs(mean_pnl) if abs(mean_pnl) > 1e-9 else float("inf")
    turnover = _avg_turnover(positions)

    assert model is not None
    return WalkForwardResult(
        pnl=pnl,
        positions=positions,
        sharpe=sharpe,
        max_drawdown=max_dd,
        mean_pnl=mean_pnl,
        turnover_avg=turnover,
        norm_dd_over_return=norm_dd,
        final_model=model,
    )
