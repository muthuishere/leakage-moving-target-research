"""Demo 3 — look_ahead_cheat is a positive control: it MUST fire.

A leakage harness you cannot trust is one that never catches a real leak.
We plant returns[t+1] directly into the day-t feature vector on an honest
panel that has no real edge. A working harness must light up: Sharpe
rockets past 5 AND the cheat run's absolute P&L dwarfs the honest run's by
>= 10x (the magnitude co-condition that defeats the scale-invariance of
Sharpe alone).

Run: uv run demos/03_look_ahead_cheat.py  (~40 s)
"""

from leakage_harness import negative_sharpe, planted_panel, walk_forward
from leakage_harness.leakage_tests import look_ahead_cheat_test

HP = {"hidden": 16, "dropout": 0.1, "lr": 3e-3, "n_steps": 150, "weight_decay": 5e-2}
N_TRAIN, N_HOLDOUT = 90, 180


def main() -> None:
    features, returns = planted_panel(T=320, S=16, time_edge=0.0, noise=1.0)

    honest = walk_forward(
        features, returns,
        n_train=N_TRAIN, n_holdout=N_HOLDOUT,
        loss_fn=negative_sharpe, hp=HP, retrain_every=5,
    )
    gate = look_ahead_cheat_test(
        features, returns,
        n_train=N_TRAIN, n_holdout=N_HOLDOUT,
        loss_fn=negative_sharpe, hp=HP, retrain_every=5,
        mean_pnl_honest=honest.mean_pnl,
    )

    print("=== Demo 3: look-ahead cheat (positive control) ===")
    print(f"honest Sharpe   : {honest.sharpe:+.3f}")
    print(f"cheat Sharpe    : {gate.value:+.3f}")
    print(f"expected        : {gate.expected}")
    print(f"gate fired (detected the planted leak): {gate.passed}")
    print(f"notes: {gate.notes}")


if __name__ == "__main__":
    main()
