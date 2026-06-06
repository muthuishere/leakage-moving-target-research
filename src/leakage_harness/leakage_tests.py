"""Four mandatory leakage tests for the walk-forward harness.

Per `docs/specs/in-progress/003-phase-3.md`: a passing S9 with positive
strategy Sharpes is meaningless if any of these tests fails — the
harness either cannot detect a leak, has a real leak, or is
non-deterministic. Each test returns a uniform `LeakageResult`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from leakage_harness.walk_forward import WalkForwardResult, walk_forward


@dataclass
class LeakageResult:
    name: str
    passed: bool
    value: float
    expected: str
    notes: str = ""


# ---------------- helpers ----------------


def _sharpe_se(n_holdout: int, n_stocks: int) -> float:
    """Standard error on annualised Sharpe with N independent samples."""
    n = max(n_holdout * n_stocks, 1)
    return math.sqrt(252.0 / n)


# ---------------- tests ----------------


def _run_shuffled_once(
    features: torch.Tensor,
    returns: torch.Tensor,
    *,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int,
    seed: int,
) -> WalkForwardResult:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(returns.shape[0], generator=gen)
    shuffled = returns[perm]
    return walk_forward(
        features, shuffled,
        n_train=n_train, n_holdout=n_holdout,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )


def shuffled_target_test(
    features: torch.Tensor,
    returns: torch.Tensor,
    *,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int,
    seed: int = 20260514,
    bootstrap_k: int = 0,
) -> LeakageResult:
    """Permute the returns along the time axis. Train + eval. Expected:
    Sharpe collapses to within the bound of zero. If not, there's a leak.

    `bootstrap_k`: when 0 (default) use the closed-form `2·SE` bound — the
    legacy behaviour, preserved bit-exact. When ≥ 5, run shuffled_target K
    times with seeds `[seed, seed+1, …, seed+K-1]`, compute the empirical
    mean + std of those K Sharpes, and use `2·max(empirical_std, SE)` as the
    bound (floored at the closed-form SE so we never accept a less-tight
    bound). The headline value is `sharpes[0]` for seed reproducibility.

    Per L53: the closed-form SE underestimates the true null distribution
    by ~6× at small n_holdout × n_stocks regimes; the bootstrap path is
    the correctly-calibrated bound at the cost of K walk_forward runs.
    """
    if bootstrap_k == 0:
        r = _run_shuffled_once(
            features, returns,
            n_train=n_train, n_holdout=n_holdout,
            loss_fn=loss_fn, hp=hp,
            retrain_every=retrain_every, seed=seed,
        )
        se = _sharpe_se(n_holdout=int(r.pnl.numel()), n_stocks=returns.shape[1])
        bound = max(1.0, 2.0 * se)
        passed = abs(r.sharpe) < bound
        return LeakageResult(
            name="shuffled_target",
            passed=passed,
            value=r.sharpe,
            expected=f"|sharpe| < {bound:.2f} (2·SE)",
            notes="Returns permuted along time; any signal is leakage.",
        )

    if bootstrap_k < 5:
        raise ValueError(
            f"bootstrap_k must be 0 or >= 5 (got {bootstrap_k}); K<5 gives "
            "a uselessly wide CI"
        )

    sharpes: list[float] = []
    n_holdout_eff = 0
    for k in range(bootstrap_k):
        r = _run_shuffled_once(
            features, returns,
            n_train=n_train, n_holdout=n_holdout,
            loss_fn=loss_fn, hp=hp,
            retrain_every=retrain_every, seed=seed + k,
        )
        sharpes.append(float(r.sharpe))
        if k == 0:
            n_holdout_eff = int(r.pnl.numel())

    headline = sharpes[0]
    mean_s = sum(sharpes) / len(sharpes)
    var_s = sum((s - mean_s) ** 2 for s in sharpes) / len(sharpes)
    std_s = math.sqrt(var_s)
    se = _sharpe_se(n_holdout=n_holdout_eff, n_stocks=returns.shape[1])
    bound = max(1.0, 2.0 * max(std_s, se))
    passed = abs(headline - mean_s) < bound
    return LeakageResult(
        name="shuffled_target",
        passed=passed,
        value=headline,
        expected=(
            f"|sharpe - empirical_mean| < {bound:.2f} "
            f"(2·max(emp_std={std_s:.3f}, SE={se:.3f}))"
        ),
        notes=(
            f"bootstrap K={bootstrap_k}: empirical_mean={mean_s:+.3f}, "
            f"empirical_std={std_s:.3f}, closed_form_SE={se:.3f}, "
            f"ratio_emp_to_se={std_s / max(se, 1e-12):.2f}. "
            "Returns permuted along time; any signal is leakage."
        ),
    )


def _augment_with_lookahead(features: torch.Tensor, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Concat returns[t+1] as an extra feature on day t. Returns truncated
    arrays of length T-1."""
    T = features.shape[0]
    next_ret = returns[1:].unsqueeze(-1)  # [T-1, S, 1]
    aug = torch.cat([features[:-1], next_ret], dim=-1)  # [T-1, S, F+1]
    return aug, returns[:-1]


def look_ahead_cheat_test(
    features: torch.Tensor,
    returns: torch.Tensor,
    *,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int,
    seed: int = 20260514,
    transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    mean_pnl_honest: float | None = None,
) -> LeakageResult:
    """Plant a future-return feature in the input. Expected: Sharpe rockets
    (≥ 5) AND the cheat run's |mean_pnl| dwarfs the honest run's by ≥ 10x.

    `transform_fn` (Phase 6.A): if the honest pipeline standardised its
    features, the same transform must be applied to the augmented tensor
    so the planted cheat column lands on a comparable z-score footing
    instead of being invisibly small at raw returns-scale.

    `mean_pnl_honest`: enables the magnitude co-condition. Sharpe alone is
    scale-invariant — a tiny-position cheat can post a high Sharpe without
    actually exploiting the leak. The ratio check forces the cheat to
    *materially* dominate the honest signal in absolute P&L."""
    aug_features, aug_returns = _augment_with_lookahead(features, returns)
    if transform_fn is not None:
        aug_features = transform_fn(aug_features)
    r = walk_forward(
        aug_features, aug_returns,
        n_train=n_train, n_holdout=n_holdout,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )
    sharpe_ok = r.sharpe >= 5.0
    if mean_pnl_honest is None:
        passed = sharpe_ok
        ratio = float("nan")
        expected = "sharpe >= 5.0"
        notes = "returns[t+1] concatenated as input feature at time t."
    else:
        ratio = abs(r.mean_pnl) / max(abs(mean_pnl_honest), 1e-8)
        passed = sharpe_ok and ratio >= 10.0
        expected = "sharpe >= 5.0 AND |mean_pnl_cheat|/|mean_pnl_honest| >= 10"
        notes = (f"returns[t+1] concatenated as input feature at time t. "
                 f"mean_pnl_ratio={ratio:.2f}")
    return LeakageResult(
        name="look_ahead_cheat",
        passed=passed,
        value=r.sharpe,
        expected=expected,
        notes=notes,
    )


def _shift_news_forward(features: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Shift the news-embedding portion of features +1 day along time.
    `features[t, :, :embed_dim]` becomes the news that was actually
    published at t+1. The last row's news goes to zero."""
    shifted = features.clone()
    shifted[:-1, :, :embed_dim] = features[1:, :, :embed_dim]
    shifted[-1, :, :embed_dim] = 0.0
    return shifted


def future_news_cheat_test(
    features: torch.Tensor,
    returns: torch.Tensor,
    *,
    embed_dim: int,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int,
    seed: int = 20260514,
    transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    mean_pnl_honest: float | None = None,
) -> LeakageResult:
    """Shift news features +1 day so the model sees tomorrow's news today.
    Expected: Sharpe rockets (≥ 5) AND |mean_pnl_cheat|/|mean_pnl_honest|
    ≥ 10. Different channel from look_ahead_cheat on purpose — news-side
    leak vs price-side leak.

    `transform_fn` / `mean_pnl_honest`: see `look_ahead_cheat_test`."""
    shifted = _shift_news_forward(features, embed_dim)
    if transform_fn is not None:
        shifted = transform_fn(shifted)
    r = walk_forward(
        shifted, returns,
        n_train=n_train, n_holdout=n_holdout,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )
    sharpe_ok = r.sharpe >= 5.0
    if mean_pnl_honest is None:
        passed = sharpe_ok
        ratio = float("nan")
        expected = "sharpe >= 5.0"
        notes = "News embeddings shifted +1 day; features at t carry t+1's news."
    else:
        ratio = abs(r.mean_pnl) / max(abs(mean_pnl_honest), 1e-8)
        passed = sharpe_ok and ratio >= 10.0
        expected = "sharpe >= 5.0 AND |mean_pnl_cheat|/|mean_pnl_honest| >= 10"
        notes = (f"News embeddings shifted +1 day; features at t carry t+1's news. "
                 f"mean_pnl_ratio={ratio:.2f}")
    return LeakageResult(
        name="future_news_cheat",
        passed=passed,
        value=r.sharpe,
        expected=expected,
        notes=notes,
    )


def bit_exact_reproducibility_test(
    features: torch.Tensor,
    returns: torch.Tensor,
    *,
    n_train: int,
    n_holdout: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    hp: dict,
    retrain_every: int,
    seed: int = 20260514,
) -> LeakageResult:
    """Two runs with the same seed must produce identical per-step P&L.
    If not, the harness has non-determinism that invalidates everything
    else."""
    r1 = walk_forward(
        features, returns,
        n_train=n_train, n_holdout=n_holdout,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )
    r2 = walk_forward(
        features, returns,
        n_train=n_train, n_holdout=n_holdout,
        loss_fn=loss_fn, hp=hp,
        retrain_every=retrain_every, seed=seed,
    )
    bit_exact = bool(torch.equal(r1.pnl, r2.pnl))
    if bit_exact:
        max_diff = 0.0
    else:
        max_diff = float((r1.pnl - r2.pnl).abs().max().item())
    return LeakageResult(
        name="bit_exact_reproducibility",
        passed=bit_exact,
        value=max_diff,
        expected="max |pnl_run1 - pnl_run2| == 0.0",
        notes="Two runs of walk_forward with identical inputs and seed.",
    )
