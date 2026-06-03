"""
Input data validation. Catches the bad data that silently poisons models.

This is intentionally dependency-free (pure pandas) so it runs in CI everywhere.
In a real stack you'd express the same contract with pandera or Great
Expectations; the checks are identical, the framework just gives nicer reports.
"""
from __future__ import annotations

import pandas as pd

SCHEMA = {
    "date": "datetime64[ns]",
    "store_id": "object",
    "product_id": "object",
    "category": "object",
    "price": "float",
    "is_promotion": "bool",
    "is_holiday": "bool",
    "units_sold": "int",
}


class DataValidationError(AssertionError):
    pass


def validate(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Validate the demand panel. Raises DataValidationError on hard failures."""
    problems: list[str] = []

    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        problems.append("`date` is not datetime")
    if (df["units_sold"] < 0).any():
        problems.append("negative `units_sold` present")
    if (df["price"] <= 0).any():
        problems.append("non-positive `price` present")
    if df["units_sold"].isna().any():
        problems.append("null `units_sold` present")

    # Duplicate (store, product, date) keys would corrupt lag/rolling features
    dupes = df.duplicated(subset=["store_id", "product_id", "date"]).sum()
    if dupes:
        problems.append(f"{dupes} duplicate (store, product, date) rows")

    # Each series should be contiguous daily; gaps break lag alignment
    gap_series = 0
    for _, g in df.groupby(["store_id", "product_id"], sort=False):
        d = g["date"].sort_values()
        if len(d) > 1 and (d.diff().dropna().dt.days != 1).any():
            gap_series += 1
    if gap_series:
        problems.append(f"{gap_series} series have non-daily gaps (reindex first)")

    if problems and strict:
        raise DataValidationError("; ".join(problems))
    return df


if __name__ == "__main__":
    from src.data.generate import generate_demand

    df = generate_demand(n_stores=3, n_products=4, n_days=60)
    validate(df)
    print("validation passed:", df.shape)
