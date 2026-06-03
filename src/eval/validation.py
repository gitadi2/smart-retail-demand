"""
Time-series validation. NO random splits anywhere -- that is itself leakage.

`temporal_split` : single chronological train/test cut by date.
`walk_forward`   : rolling-origin CV. Train on [start, cutoff], test the next
                   `horizon` days, advance the cutoff, repeat. Reports the
                   distribution of WMAPE across folds, which is what you quote
                   ("WMAPE 21.4% +/- 1.8 over 5 folds"), not a single lucky split.
"""
from __future__ import annotations

import pandas as pd


def temporal_split(df: pd.DataFrame, test_days: int = 56, date_col: str = "date"):
    cutoff = df[date_col].max() - pd.Timedelta(days=test_days)
    train = df[df[date_col] <= cutoff]
    test = df[df[date_col] > cutoff]
    return train, test, cutoff


def walk_forward(
    df: pd.DataFrame,
    n_folds: int = 5,
    horizon: int = 28,
    min_train_days: int = 180,
    date_col: str = "date",
):
    """Yield (fold_idx, train_df, test_df) for rolling-origin CV.

    The last `n_folds * horizon` days are carved into consecutive test blocks;
    each fold trains only on data strictly before its test block.
    """
    dates = pd.Index(sorted(df[date_col].unique()))
    total_test = n_folds * horizon
    if len(dates) < min_train_days + total_test:
        raise ValueError(
            f"Need >= {min_train_days + total_test} distinct days, got {len(dates)}"
        )

    first_test_pos = len(dates) - total_test
    for k in range(n_folds):
        test_start = dates[first_test_pos + k * horizon]
        test_end_pos = first_test_pos + (k + 1) * horizon - 1
        test_end = dates[test_end_pos]
        train = df[df[date_col] < test_start]
        test = df[(df[date_col] >= test_start) & (df[date_col] <= test_end)]
        yield k, train, test
