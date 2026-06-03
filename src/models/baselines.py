"""
Baselines. If your fancy model can't beat seasonal-naive, it has no business
shipping. Quoting "R^2 = 0.99" without a baseline is meaningless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(test_df: pd.DataFrame, lag_col: str = "lag_7") -> np.ndarray:
    """Predict y[t] = y[t-7] (last week, same weekday).

    Uses the leakage-safe lag_7 feature already built. NaNs (series start)
    fall back to lag_1, then to 0.
    """
    pred = test_df[lag_col].to_numpy(dtype=float)
    if "lag_1" in test_df:
        pred = np.where(np.isnan(pred), test_df["lag_1"].to_numpy(dtype=float), pred)
    return np.nan_to_num(pred, nan=0.0)
