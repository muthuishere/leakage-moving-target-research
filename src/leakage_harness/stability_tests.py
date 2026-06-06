"""Stability tests for the walk-forward harness — different contract from
leakage tests. A strategy that passes leakage but fails stability is a
window-position artifact, not a real edge."""

from __future__ import annotations

import math
from typing import Callable

import torch

from leakage_harness.walk_forward import walk_forward
from leakage_harness.leakage_tests import (
    LeakageResult,
    _sharpe_se,
)


def window_robustness_test(
    features: torch.Tensor,
    returns: torch.Tensor,
    *,
    n_train: int,
    n_holdout_short: int = 21,
    n_holdout_long: int = 180,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int,
    seed: int = 20260515,
) -> LeakageResult:
    """Run walk_forward on two holdout sizes against the same data.
    A real, robust edge must produce Sharpe values consistent within
    2 * sqrt(SE_short^2 + SE_long^2)."""
    r_short = walk_forward(
        features, returns,
        n_train=n_train, n_holdout=n_holdout_short,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )
    r_long = walk_forward(
        features, returns,
        n_train=n_train, n_holdout=n_holdout_long,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )

    n_short = int(r_short.pnl.numel())
    n_long = int(r_long.pnl.numel())
    n_stocks = returns.shape[1]
    se_short = _sharpe_se(n_holdout=n_short, n_stocks=n_stocks)
    se_long = _sharpe_se(n_holdout=n_long, n_stocks=n_stocks)
    combined_se = math.sqrt(se_short * se_short + se_long * se_long)

    diff = abs(r_short.sharpe - r_long.sharpe)
    passed = diff < 2.0 * combined_se

    return LeakageResult(
        name="window_robustness",
        passed=passed,
        value=diff,
        expected=f"|sharpe_short - sharpe_long| < {2.0 * combined_se:.2f}",
        notes=(
            f"sharpe_short={r_short.sharpe:.3f} (SE={se_short:.2f}, n={n_short}), "
            f"sharpe_long={r_long.sharpe:.3f} (SE={se_long:.2f}, n={n_long})"
        ),
    )

def static_features_test(
    features: torch.Tensor,
    returns: torch.Tensor,
    *,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int,
    seed: int = 20260516,
) -> LeakageResult:
    """Replace each feature with its per-stock mean. Time-variation is
    destroyed but stock-identity is preserved. A model that uses
    time-edge should produce ~zero positions (no time signal); a model
    that memorised stock-identity will still emit its baseline
    positions and post a non-zero Sharpe — exposing the L43 leakage
    mechanism that shuffled_target alone could not catch.

    Pass iff |sharpe| < 2 * SE (model is NOT using static stock
    structure to generate fake edge)."""
    static_feat = features.mean(dim=0, keepdim=True).expand_as(features).contiguous()
    r = walk_forward(
        static_feat, returns,
        n_train=n_train, n_holdout=n_holdout,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )
    se = _sharpe_se(n_holdout=int(r.pnl.numel()), n_stocks=returns.shape[1])
    bound = max(1.0, 2.0 * se)
    passed = abs(r.sharpe) < bound
    return LeakageResult(
        name="static_features",
        passed=passed,
        value=r.sharpe,
        expected=f"|sharpe| < {bound:.2f} (2·SE)",
        notes="features replaced with per-stock mean; only ticker-identity signal remains.",
    )


def permutation_invariance_test(
    features: torch.Tensor,
    returns: torch.Tensor,
    *,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int,
    seed: int = 20260517,
) -> LeakageResult:
    """Train on un-permuted, then retrain a fresh model on
    (perm(features), perm(returns)) with the same permutation. Pass iff
    the two Sharpes are statistically consistent (real time-edge survives
    ticker renaming; ticker-memorisation does not).

    Mechanism: physics from feature→return doesn't care about ticker
    labels. A model that found genuine time-edge retrains on the
    permuted data to the same Sharpe. A model that memorised
    (ticker_label → constant position) learns the wrong mapping on
    permuted training data and collapses when evaluated against the
    un-permuted returns at eval time (i.e. the harness's own next
    walk_forward call on permuted (features, returns))."""
    r_base = walk_forward(
        features, returns,
        n_train=n_train, n_holdout=n_holdout,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )

    S = features.shape[1]
    perm = torch.randperm(S, generator=torch.Generator().manual_seed(seed))
    feat_p = features[:, perm, :].contiguous()
    rets_p = returns[:, perm].contiguous()

    r_perm = walk_forward(
        feat_p, rets_p,
        n_train=n_train, n_holdout=n_holdout,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )

    n_stocks = returns.shape[1]
    se_a = _sharpe_se(n_holdout=int(r_base.pnl.numel()), n_stocks=n_stocks)
    se_b = _sharpe_se(n_holdout=int(r_perm.pnl.numel()), n_stocks=n_stocks)
    combined_se = math.sqrt(se_a * se_a + se_b * se_b)

    diff = abs(r_base.sharpe - r_perm.sharpe)
    passed = diff < 2.0 * combined_se

    return LeakageResult(
        name="permutation_invariance",
        passed=passed,
        value=diff,
        expected=f"|Δsharpe| < {2.0 * combined_se:.2f}",
        notes=(
            f"sharpe_base={r_base.sharpe:.3f}, "
            f"sharpe_permuted={r_perm.sharpe:.3f}"
        ),
    )

