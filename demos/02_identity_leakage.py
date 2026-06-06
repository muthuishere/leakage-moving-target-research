"""Demo 2 — static_features catches per-stock identity memorisation.

shuffled_target alone cannot see this leak: the model isn't using future
information, it is memorising which ticker is which. We feed it a panel
with a persistent per-stock drift exposed through a constant identity
feature, then replace every feature with its per-stock time-mean. A real
time-edge goes flat (no time variation left, Sharpe within the null bound);
an identity-memoriser keeps its positions and keeps its Sharpe — which is
the failure the gate reports.

Run: uv run demos/02_identity_leakage.py  (~1 min)
"""

from leakage_harness import planted_panel
from leakage_harness.losses import negative_sharpe
from leakage_harness.stability_tests import static_features_test

HP = {"hidden": 16, "dropout": 0.1, "lr": 3e-3, "n_steps": 150, "weight_decay": 5e-2}
N_TRAIN, N_HOLDOUT = 90, 180


def run(label: str, **panel_kwargs) -> None:
    features, returns = planted_panel(T=320, S=16, **panel_kwargs)
    gate = static_features_test(
        features, returns,
        n_train=N_TRAIN, n_holdout=N_HOLDOUT,
        loss_fn=negative_sharpe, hp=HP, retrain_every=5,
    )
    verdict = "PASS (no identity leak)" if gate.passed else "FAIL (identity memorised)"
    print(f"{label:<28} static-features Sharpe={gate.value:+.3f}  -> {verdict}")


def main() -> None:
    print("=== Demo 2: per-stock identity leakage ===")
    run("pure time-edge panel",  time_edge=0.12, identity_drift=0.0)
    run("identity-drift panel",  time_edge=0.0, identity_drift=0.4)


if __name__ == "__main__":
    main()
