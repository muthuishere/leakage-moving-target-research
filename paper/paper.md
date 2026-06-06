---
title: "Leakage Is a Moving Target: A Field Guide to Self-Deception in Differentiable Cross-Sectional Trading"
author:
  - name: Muthukumaran Navaneethakrishnan
    orcid: 0009-0004-3577-1953
    affiliation: TODO
  - name: M. K. Haribalaji
    orcid: 0009-0002-8964-2212
    affiliation: TODO
date: TODO
abstract: |
  TODO — one paragraph. End-to-end differentiable models trained against a
  P&L objective find no tradeable edge on Indian large-caps (NSE; 5-ticker,
  NIFTY 50, NIFTY 100) across several signal families. The contribution is
  not the null but the negative-control gates that earn it: an escape-route
  taxonomy of how the optimiser manufactures fake edges (scale-invariance
  collapse, cross-stock identity memorisation, regime-dependent detector
  blindness) and the small, runnable probes that catch each. Positioned
  against Nikolopoulos (arXiv:2604.15531, concurrent) and the selection-bias
  line of López de Prado & Bailey.
---

> **Status: outline.** The manuscript is written in the next pass. Every
> claim below resolves to a runnable file in `../src/leakage_harness/` and a
> demo in `../demos/`. This file is the skeleton the huddle agreed on:
> narrative spine = escape-route taxonomy; under each escape route, a small
> code excerpt (the gate that caught it) + the before/after number.

## 1. Introduction — the fake-Sharpe problem
TODO. Why differentiable cross-sectional trading is uniquely good at fooling
you. Honest null as a contribution. What is and isn't novel vs Nikolopoulos.

## 2. The harness
TODO. Walk-forward + bit-exact reproducibility + the gate contract
(`LeakageResult`). Excerpt: `strategy_net.py` market-neutral `z - z.mean()`.

## 3. Escape route I — scale-invariance collapse
TODO. Differentiable Sharpe is scale-invariant; the optimiser drives
mean_pnl -> 0. Excerpt: signature-dispatch loss wiring + `sharpe_with_position_floor`.

## 4. Escape route II — cross-stock identity memorisation
TODO. Per-stock structure memorised as identity, invisible to
shuffled_target. Excerpts: `static_features_test`, `permutation_invariance_test`.
Demo: `02_identity_leakage.py`.

## 5. Escape route III — regime-dependent detector blindness
TODO. A leakage detector's own power depends on the preprocessing regime
under test; no single configuration passes all gates.

## 6. Calibrating the null
TODO. Closed-form Sharpe SE under-states the null ~6x at small N; the
bootstrap-calibrated bound. Demo: `01_structural_vs_time_leakage.py`.

## 7. Results — the honest null
TODO. Leakage-clean Sharpes sit in the noise band across signal families at
N50/N100.

## 8. Limitations & threats
TODO.

## 9. Conclusion
TODO. The methodology generalises; the data edge does not.

## References
See `references.bib` (TODO).
