"""Demo 5 — window_robustness catches a window-position artifact.

A real edge holds up across holdout sizes. We compare a short holdout (21
days) against a long one (180 days) on the same data and seed; a robust
edge keeps a consistent Sharpe, while an edge that only existed in one
stretch of the series shows up strong in a short window and dilutes in a
long one.

- Stationary panel (edge across the whole series): Sharpe consistent
  across window sizes -> gate PASSES.
- Regime-shifted panel (edge only in the early days, noise afterward): the
  short holdout sits in the signal regime and looks strong, the long
  holdout averages it away -> large Sharpe gap -> gate FAILS.

Run: uv run demos/05_window_robustness.py  (~1-2 min)
"""

from leakage_harness.losses import negative_sharpe
from leakage_harness.fixtures import planted_panel
from leakage_harness.stability_tests import window_robustness_test

HP = {"hidden": 16, "dropout": 0.1, "lr": 3e-3, "n_steps": 150, "weight_decay": 5e-2}
N_TRAIN = 60
SHORT, LONG = 21, 180


def run(label: str, **panel_kwargs) -> None:
    features, returns = planted_panel(T=280, S=16, **panel_kwargs)
    gate = window_robustness_test(
        features, returns,
        n_train=N_TRAIN, n_holdout_short=SHORT, n_holdout_long=LONG,
        loss_fn=negative_sharpe, hp=HP, retrain_every=5,
    )
    verdict = "PASS (window-robust)" if gate.passed else "FAIL (window artifact)"
    print(f"{label:<26} |Δsharpe|={gate.value:.3f}  ({gate.notes})  -> {verdict}")


def main() -> None:
    print("=== Demo 5: window robustness ===")
    run("stationary edge",   time_edge=0.15, edge_end=None)
    run("regime-shifted edge", time_edge=0.15, edge_end=100)


if __name__ == "__main__":
    main()
