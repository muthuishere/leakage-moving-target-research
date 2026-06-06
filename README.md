# Leakage Is a Moving Target

**A field guide to self-deception in differentiable cross-sectional trading.**

End-to-end differentiable models trained against a P&L objective are
unusually good at fooling you. This repository is the runnable companion to
the paper: a small set of **negative-control gates** that each catch a
specific way the optimiser manufactures a fake edge, plus the synthetic
fixtures that reproduce every number with no market data, no credentials,
and no network.

The honest result on Indian large-caps (NSE; 5-ticker, NIFTY 50, NIFTY 100)
was a **null** — no tradeable edge survived the gates. The contribution is
the gates and the escape-route taxonomy, not an edge.

> Positioned alongside Nikolopoulos, *Spurious Predictability in Financial
> Machine Learning* (arXiv:2604.15531, concurrent) and the
> selection-bias line of López de Prado & Bailey. See `paper/`.

## Quick start

```sh
uv sync
uv run demos/01_structural_vs_time_leakage.py
uv run demos/02_identity_leakage.py
uv run demos/03_look_ahead_cheat.py
```

Everything runs on CPU against seeded synthetic panels — each demo takes
from ~40 s to a couple of minutes (demo 1 runs a bootstrap null).

## What each gate catches

| Finding | Gate (file) | Demo | Paper § |
|---|---|---|---|
| A real edge is a *time* edge; permuting time must kill it | `shuffled_target_test` — `src/leakage_harness/leakage_tests.py` | `demos/01_structural_vs_time_leakage.py` | §6 |
| Per-stock identity memorisation (survives shuffled_target) | `static_features_test` — `src/leakage_harness/stability_tests.py` | `demos/02_identity_leakage.py` | §4 |
| Ticker labels carry no physics; renaming must not change Sharpe | `permutation_invariance_test` — `stability_tests.py` | TODO | §4 |
| Window-position artifact (Sharpe must hold across holdout sizes) | `window_robustness_test` — `stability_tests.py` | TODO | §7 |
| Positive control: a planted future feature MUST be detected | `look_ahead_cheat_test` — `leakage_tests.py` | `demos/03_look_ahead_cheat.py` | §3 |
| Closed-form Sharpe SE under-states the null ~6x at small N | `shuffled_target_test(bootstrap_k=...)` — `leakage_tests.py` | `demos/01_structural_vs_time_leakage.py` | §6 |
| Scale-invariance of differentiable Sharpe (positions → 0) | `sharpe_with_position_floor` — `losses.py`; signature-dispatch in `walk_forward.py` | TODO | §3 |

## Layout

```
src/leakage_harness/   # the lifted, self-contained harness
  backtest.py          # differentiable_backtest primitive
  losses.py            # P&L / Sharpe / drawdown losses (+ position-floor)
  strategy_net.py      # market-neutral position emitter (z - z.mean())
  walk_forward.py      # walk-forward train+backtest engine
  leakage_tests.py     # shuffled_target, look_ahead_cheat, future_news_cheat, bit_exact
  stability_tests.py   # static_features, permutation_invariance, window_robustness
  fixtures.py          # synthetic planted-signal panels (seeded)
demos/                 # each script reproduces one paper number
paper/                 # manuscript + Zenodo metadata
```

The harness was extracted from the `stock-core` research repo and made
import-standalone; the methodology and findings are unchanged.

## Authors

Muthukumaran Navaneethakrishnan · M. K. Haribalaji
