"""Demo 4 — permutation_invariance: a structural control satisfied by construction.

The physics from features to returns does not care about ticker names, so
relabelling the tickers (permuting the stock axis of BOTH features and
returns) must not change the Sharpe. This gate is the negative control for
models that carry *per-stock parameters* — those memorise a
`ticker_label -> position` map and collapse when the labels are shuffled.

The harness's StrategyNet is a SHARED-WEIGHT, permutation-equivariant net
(one MLP applied to every stock, then a cross-sectional de-mean), so it
satisfies this control essentially by construction: base and permuted runs
produce the same Sharpe (Δ ~ 0) and the gate passes. We show that here as a
sanity check — and note honestly that this gate only bites against a
per-stock-parameterised model, which this architecture is not.

Run: uv run demos/04_permutation_invariance.py  (~40 s)
"""

from leakage_harness import planted_panel
from leakage_harness.losses import negative_sharpe
from leakage_harness.stability_tests import permutation_invariance_test

HP = {"hidden": 16, "dropout": 0.1, "lr": 3e-3, "n_steps": 150, "weight_decay": 5e-2}
N_TRAIN, N_HOLDOUT = 60, 120


def main() -> None:
    features, returns = planted_panel(T=240, S=16, time_edge=0.3, noise=1.0)
    gate = permutation_invariance_test(
        features, returns,
        n_train=N_TRAIN, n_holdout=N_HOLDOUT,
        loss_fn=negative_sharpe, hp=HP, retrain_every=5,
    )

    print("=== Demo 4: permutation invariance (structural control) ===")
    print(f"|Δsharpe| base vs relabelled : {gate.value:+.4f}")
    print(f"expected                     : {gate.expected}")
    print(f"gate passed (relabel-invariant): {gate.passed}")
    print(f"notes: {gate.notes}")
    print("\nA shared-weight net is permutation-equivariant by construction;")
    print("this control bites against per-stock-parameterised models instead.")


if __name__ == "__main__":
    main()
