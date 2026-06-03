"""
CI tests. pytest-compatible; also runnable directly (`python tests/test_all.py`)
since pytest may not be installed everywhere.

The performance-regression test is the important one: it fails the build if the
model stops beating the baseline or if WMAPE degrades past a threshold. That is
how you stop a silent model regression from shipping.
"""
from __future__ import annotations

import numpy as np

from src.audit.leakage import run_audit
from src.data.generate import generate_demand
from src.data.validation import DataValidationError, validate
from src.eval.metrics import wmape, r2
from src.eval.validation import temporal_split, walk_forward
from src.features.engineering import (
    TARGET,
    build_features,
    build_features_LEAKY,
    feature_columns,
)
from src.models.baselines import seasonal_naive
from src.models.train import DemandModel

# Performance gate. The robust invariant is "beats naive by a clear margin";
# the absolute ceiling is set for THIS small/fast test config (the full pipeline
# on the complete dataset lands ~26%). Tighten per real dataset.
MAX_WMAPE = 40.0
MIN_REL_IMPROVEMENT = 0.20  # model must be >=20% better than seasonal-naive


def _data(n_days=260):
    return generate_demand(n_stores=4, n_products=6, n_days=n_days)


def test_clean_pipeline_has_no_leakage():
    audit = run_audit(_data(), build_features)
    assert audit["passed"], audit


def test_audit_catches_injected_leak():
    audit = run_audit(_data(), build_features_LEAKY)
    assert not audit["passed"]
    assert "roll_mean_7_LEAK" in audit["target_perturbation"]["detail"]


def test_no_random_split_temporal_order():
    feat = build_features(_data())
    train, test, _ = temporal_split(feat)
    assert train["date"].max() < test["date"].min()


def test_validation_rejects_negative_units():
    bad = _data().copy()
    bad.loc[bad.index[0], "units_sold"] = -5
    try:
        validate(bad)
        assert False, "should have raised"
    except DataValidationError:
        pass


def test_metrics_known_values():
    y = np.array([10.0, 20.0, 30.0])
    assert wmape(y, y) == 0.0
    assert r2(y, y) == 1.0
    # WMAPE of a constant-1-off forecast = 3 / 60 * 100 = 5%
    assert abs(wmape(y, y + 1) - 5.0) < 1e-9


def test_model_beats_baseline_and_meets_threshold():
    feat = build_features(_data(n_days=320))
    cols = feature_columns(feat)
    scores, base = [], []
    for _, tr, te in walk_forward(feat, n_folds=3, horizon=28, min_train_days=120):
        m = DemandModel("hgb").fit(tr[cols], tr[TARGET])
        scores.append(wmape(te[TARGET], m.predict(te[cols])))
        base.append(wmape(te[TARGET], seasonal_naive(te)))
    model_wmape, base_wmape = float(np.mean(scores)), float(np.mean(base))
    assert model_wmape < base_wmape, (model_wmape, base_wmape)
    assert model_wmape < (1 - MIN_REL_IMPROVEMENT) * base_wmape, (model_wmape, base_wmape)
    assert model_wmape < MAX_WMAPE, model_wmape


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
