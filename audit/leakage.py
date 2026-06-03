"""
Automated leakage audit. Three independent checks:

1) TEMPORAL ORDER  -- train dates strictly precede test dates.
2) TARGET PERTURBATION -- the strong one. Perturb y at the LAST row of each
   series, rebuild features, and assert no feature value AT THAT ROW changed.
   A leakage-free feature for row t depends only on rows < t, so perturbing y[t]
   cannot move it. If it moves, the feature is reading the target. This catches
   the same-row-rolling-mean bug regardless of how it's disguised.
3) BASELINE SANITY -- model must beat seasonal-naive but a near-perfect R^2 on
   noisy demand is flagged as suspicious rather than celebrated.

Run: python -m src.audit.leakage
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import GROUP, TARGET


def check_temporal_order(train: pd.DataFrame, test: pd.DataFrame) -> tuple[bool, str]:
    ok = train["date"].max() < test["date"].min()
    return ok, (
        "OK: train precedes test"
        if ok
        else f"LEAK: train max {train['date'].max()} >= test min {test['date'].min()}"
    )


def _candidate_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column that could be fed to a model -- independent of the
    production name filter, so a leak in a forgotten column is still caught."""
    exclude = set(GROUP) | {TARGET, "date"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def check_target_perturbation(raw: pd.DataFrame, build_fn) -> tuple[bool, list[str]]:
    """Returns (passed, leaking_feature_names)."""
    base = build_fn(raw.copy())
    cols = _candidate_columns(base)

    # last row per series -> features there depend only on earlier same-series rows
    last_idx = base.groupby(GROUP, sort=False).tail(1).index
    before = base.loc[last_idx, cols].copy()

    perturbed = raw.copy()
    pidx = perturbed.groupby(GROUP, sort=False).tail(1).index
    perturbed.loc[pidx, TARGET] = perturbed.loc[pidx, TARGET] + 10_000

    after_full = build_fn(perturbed)
    after = after_full.loc[last_idx, cols]

    leaking = []
    for c in cols:
        b = before[c].to_numpy(dtype=float)
        a = after[c].to_numpy(dtype=float)
        # ignore positions that are NaN in both
        mask = ~(np.isnan(a) & np.isnan(b))
        if not np.allclose(np.nan_to_num(a[mask]), np.nan_to_num(b[mask])):
            leaking.append(c)
    return (len(leaking) == 0), leaking


def check_baseline_sanity(model_r2: float, baseline_wmape: float,
                          model_wmape: float) -> tuple[bool, str]:
    if model_r2 > 0.97:
        return False, f"SUSPICIOUS: R^2={model_r2:.4f} too high for noisy demand -> audit features"
    if model_wmape >= baseline_wmape:
        return False, f"WEAK: model WMAPE {model_wmape:.1f} does not beat naive {baseline_wmape:.1f}"
    return True, f"OK: beats naive ({model_wmape:.1f} < {baseline_wmape:.1f}), R^2 plausible"


def run_audit(raw: pd.DataFrame, build_fn) -> dict:
    from src.eval.validation import temporal_split

    train, test, _ = temporal_split(build_fn(raw.copy()))
    t_ok, t_msg = check_temporal_order(train, test)
    p_ok, leaking = check_target_perturbation(raw, build_fn)
    report = {
        "temporal_order": {"passed": t_ok, "detail": t_msg},
        "target_perturbation": {
            "passed": p_ok,
            "detail": "OK: no feature uses y[t]" if p_ok else f"LEAK via: {leaking}",
        },
    }
    report["passed"] = t_ok and p_ok
    return report


if __name__ == "__main__":
    from src.data.generate import generate_demand
    from src.features.engineering import build_features, build_features_LEAKY

    raw = generate_demand(n_stores=3, n_products=4, n_days=200)

    print("=== Clean pipeline (build_features) ===")
    print(run_audit(raw, build_features))
    print("\n=== Leaky pipeline (build_features_LEAKY) ===")
    print(run_audit(raw, build_features_LEAKY))
