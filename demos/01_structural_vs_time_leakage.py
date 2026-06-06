"""Demo 1 — shuffled_target separates a real time-edge from structural leakage.

A genuine time-edge (feature 0 at t predicts returns at t+1) earns a
positive walk-forward Sharpe. Permuting the return series along the time
axis breaks the t -> t+1 link, so the leftover Sharpe must be consistent
with the null distribution of shuffles. We use the bootstrap-calibrated
bound (L53: the closed-form Sharpe SE under-states the true null by ~6x at
small holdout sizes, so a naive 2*SE bound spuriously fails).

Run: uv run demos/01_structural_vs_time_leakage.py  (~1-2 min)
"""

from leakage_harness import negative_sharpe, planted_panel, walk_forward
from leakage_harness.leakage_tests import shuffled_target_test

HP = {"hidden": 16, "dropout": 0.1, "lr": 3e-3, "n_steps": 150, "weight_decay": 5e-2}
N_TRAIN, N_HOLDOUT = 90, 180


def main() -> None:
    features, returns = planted_panel(T=320, S=16, time_edge=0.12, noise=1.0)

    honest = walk_forward(
        features, returns,
        n_train=N_TRAIN, n_holdout=N_HOLDOUT,
        loss_fn=negative_sharpe, hp=HP, retrain_every=5,
    )
    gate = shuffled_target_test(
        features, returns,
        n_train=N_TRAIN, n_holdout=N_HOLDOUT,
        loss_fn=negative_sharpe, hp=HP, retrain_every=5,
        bootstrap_k=5,
    )

    print("=== Demo 1: structural vs time leakage ===")
    print(f"honest walk-forward Sharpe : {honest.sharpe:+.3f}")
    print(f"shuffled-target Sharpe     : {gate.value:+.3f}")
    print(f"expected                   : {gate.expected}")
    print(f"gate passed (no structural leak): {gate.passed}")
    print(f"notes: {gate.notes}")


if __name__ == "__main__":
    main()
