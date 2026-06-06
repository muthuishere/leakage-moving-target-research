"""Synthetic planted-signal panels — zero market data, fully seeded.

Every demo and gate in this repo runs against these generators, so the
whole repository reproduces from a clean clone with no NSE feed, no
credentials, no BigQuery. The two knobs are deliberately the two kinds of
"signal" a cross-sectional model can latch onto:

- ``time_edge``      — a genuine time-varying edge: feature 0 at day t
  predicts the cross-section of returns at day t+1. A real edge. Survives
  ``static_features`` (it needs time variation) and dies under
  ``shuffled_target`` (permuting the time axis breaks the t -> t+1 link).
- ``identity_drift`` — a persistent per-stock drift, exposed to the model
  through a constant-over-time identity feature. NOT a time edge: a model
  can memorise ``ticker -> position`` and post a Sharpe that survives
  ``static_features`` and ``permutation_invariance``. This is the
  self-deception the negative-control gates exist to catch.
"""

from __future__ import annotations

import torch


def planted_panel(
    *,
    T: int = 180,
    S: int = 12,
    F: int = 4,
    time_edge: float = 0.0,
    identity_drift: float = 0.0,
    noise: float = 1.0,
    seed: int = 20260606,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(features [T, S, F], returns [T, S])``.

    Convention matches the harness: the position decided from
    ``features[t]`` earns ``returns[t + 1]``.

    feature 0 is the (time-varying) predictive driver; feature 1 is a
    constant-over-time per-stock identity tag; the rest are pure noise.
    """
    gen = torch.Generator().manual_seed(seed)

    # feature 0: the honest predictive driver, known at day t.
    driver = torch.randn(T, S, generator=gen)
    # per-stock identity: a constant drift mu_s and its constant feature tag.
    mu = torch.randn(S, generator=gen)
    identity_feat = mu.unsqueeze(0).expand(T, S)  # constant over time
    other = torch.randn(T, S, max(F - 2, 0), generator=gen)
    features = torch.cat(
        [driver.unsqueeze(-1), identity_feat.unsqueeze(-1), other], dim=-1
    )[:, :, :F].contiguous()

    eps = torch.randn(T, S, generator=gen)
    returns = noise * eps.clone()
    # time edge: driver at t-1 drives return at t (so features[t] -> returns[t+1]).
    if time_edge != 0.0:
        returns[1:] += time_edge * driver[:-1]
    # identity drift: persistent per-stock return, recoverable from the
    # constant identity feature without any time information.
    if identity_drift != 0.0:
        returns += identity_drift * mu.unsqueeze(0)

    return features, returns
