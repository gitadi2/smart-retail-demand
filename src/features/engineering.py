"""
Feature engineering for demand forecasting -- leakage-safe by construction.

THE RULE: a feature for row t may only use information available strictly before
the target at t is realized.

  * Calendar (dow, month, week, sin/cos)         -> known in advance. Safe.
  * Planned price / promo / holiday flags for t  -> known in advance. Safe.
  * Lags  y[t-k]                                 -> past only. Safe.
  * Rolling stats over y                         -> MUST exclude y[t].

The classic bug is `df.groupby(g)['y'].rolling(7).mean()` which INCLUDES y[t] in
its own 7-day window. We always `.shift(1)` first so the window ends at t-1.

`build_features` is the production path. `build_features_LEAKY` exists only so the
audit/tests can demonstrate detection -- never call it for real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "units_sold"
GROUP = ["store_id", "product_id"]
LAGS = [1, 7, 14, 28]
WINDOWS = [7, 14, 28]


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["dow"] = d.dayofweek
    df["month"] = d.month
    df["day"] = d.day
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * d.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * d.dayofyear / 365.25)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Production feature builder. Leakage-safe.

    Returns the frame with feature columns added. Early rows per series will have
    NaN lag/rolling features; HistGradientBoosting (and XGB/LGBM) handle NaN
    natively, so we keep them rather than dropping data.
    """
    df = df.sort_values(GROUP + ["date"]).copy()
    df["is_promotion"] = df["is_promotion"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)
    df = _calendar(df)

    g = df.groupby(GROUP, sort=False)[TARGET]

    # Lags: pure past values.
    for k in LAGS:
        df[f"lag_{k}"] = g.shift(k)

    # Rolling stats on the SHIFTED series so the window ends at t-1 (excludes y[t]).
    # We materialise the shifted target, then roll WITHIN each series via transform
    # so windows never cross series boundaries and the index always realigns.
    df["_y_shift1"] = g.shift(1)
    sgrp = df.groupby(GROUP, sort=False)["_y_shift1"]
    for w in WINDOWS:
        mp = max(2, w // 2)
        df[f"roll_mean_{w}"] = sgrp.transform(
            lambda s, w=w, mp=mp: s.rolling(w, min_periods=mp).mean()
        )
        df[f"roll_std_{w}"] = sgrp.transform(
            lambda s, w=w, mp=mp: s.rolling(w, min_periods=mp).std()
        )
    df["expanding_mean"] = sgrp.transform(
        lambda s: s.expanding(min_periods=3).mean()
    )

    # Price ratio vs the series' trailing average price (past only).
    df["_price_shift1"] = df.groupby(GROUP, sort=False)["price"].shift(1)
    df["_price_roll28"] = df.groupby(GROUP, sort=False)["_price_shift1"].transform(
        lambda s: s.rolling(28, min_periods=3).mean()
    )
    df["price_vs_roll"] = df["price"] / df["_price_roll28"]

    df = df.drop(columns=["_y_shift1", "_price_shift1", "_price_roll28"])
    return df


def build_features_LEAKY(df: pd.DataFrame) -> pd.DataFrame:
    """DO NOT USE. Intentionally leaks the target via a same-row rolling mean.

    Only here so the leakage audit/tests have something to catch. This is the
    single most common real-world mistake and the reason for absurd R^2 values.
    """
    df = build_features(df)
    # No shift -> window includes y[t]. Leak.
    df["roll_mean_7_LEAK"] = df.groupby(GROUP, sort=False)[TARGET].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All engineered + planned-known feature columns (excludes ids/target/date)."""
    exclude = set(GROUP) | {TARGET, "date", "category"}
    base_known = ["price", "is_promotion", "is_holiday"]
    engineered = [
        c
        for c in df.columns
        if c not in exclude
        and c not in base_known
        and not c.endswith("_LEAK")
        and df[c].dtype != "object"
    ]
    return base_known + engineered


if __name__ == "__main__":
    from src.data.generate import generate_demand

    df = generate_demand(n_stores=2, n_products=3, n_days=90)
    feats = build_features(df)
    cols = feature_columns(feats)
    print("n features:", len(cols))
    print(cols)
    print(feats[["date", TARGET, "lag_1", "roll_mean_7"]].head(10).to_string())
