"""
Training. HistGradientBoostingRegressor is the default engine: fast, handles NaN
natively, no external deps. XGBoost / LightGBM are used automatically IF installed
-- same interface, so `pip install xgboost lightgbm` lights them up with no code
change.

We model log1p(units) and invert on predict: demand is non-negative and
right-skewed, and this stabilises variance.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _make_model(kind: str):
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
    if kind == "xgb":
        from xgboost import XGBRegressor  # optional

        return XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
        )
    if kind == "lgbm":
        from lightgbm import LGBMRegressor  # optional

        return LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    raise ValueError(kind)


def available_models() -> list[str]:
    models = ["hgb"]
    try:
        import xgboost  # noqa: F401

        models.append("xgb")
    except ImportError:
        pass
    try:
        import lightgbm  # noqa: F401

        models.append("lgbm")
    except ImportError:
        pass
    return models


class DemandModel:
    """Thin wrapper: log1p target transform + chosen GBM engine."""

    def __init__(self, kind: str = "hgb"):
        self.kind = kind
        self.model = _make_model(kind)
        self.features: list[str] | None = None

    def fit(self, X, y):
        self.features = list(X.columns)
        self.model.fit(X.to_numpy(), np.log1p(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        if self.features is not None:
            X = X[self.features]
        pred = np.expm1(self.model.predict(X.to_numpy()))
        return np.clip(pred, 0.0, None)  # demand can't be negative
