"""leakage_harness — negative-control gates for differentiable cross-sectional trading.

Lifted and made self-contained from the stock-core research repo. Each gate
returns a uniform ``LeakageResult(name, passed, value, expected, notes)``.
"""

from leakage_harness.backtest import differentiable_backtest
from leakage_harness.fixtures import planted_panel
from leakage_harness.leakage_tests import (
    LeakageResult,
    bit_exact_reproducibility_test,
    future_news_cheat_test,
    look_ahead_cheat_test,
    shuffled_target_test,
)
from leakage_harness.losses import negative_sharpe
from leakage_harness.stability_tests import (
    permutation_invariance_test,
    static_features_test,
    window_robustness_test,
)
from leakage_harness.strategy_net import StrategyNet
from leakage_harness.walk_forward import walk_forward

__all__ = [
    "LeakageResult",
    "StrategyNet",
    "differentiable_backtest",
    "planted_panel",
    "walk_forward",
    "negative_sharpe",
    "shuffled_target_test",
    "look_ahead_cheat_test",
    "future_news_cheat_test",
    "bit_exact_reproducibility_test",
    "static_features_test",
    "permutation_invariance_test",
    "window_robustness_test",
]
