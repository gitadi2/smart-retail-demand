"""
feature_engineering.py - Engineer demand forecasting features
==============================================================
Creates lag features, rolling statistics, seasonality components,
price features, and promotion effects for time-series modeling.

Usage: python src/feature_engineering.py
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


def load_cleaned_data(path="data/processed/cleaned_sales.csv"):
    """Load cleaned sales data."""
    if not os.path.exists(path):
        print("ERROR: cleaned_sales.csv not found. Run data_cleaning.py first.")
        exit(1)
    df = pd.read_csv(path, parse_dates=["sale_date"])
    print(f"Loaded {len(df):,} rows for feature engineering")
    return df


def aggregate_daily(df):
    """Aggregate to daily store-product level for time-series modeling."""
    group_cols = ["sale_date", "store_id", "product_id"]
    agg_df = df.groupby(group_cols).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        avg_price=("unit_price", "mean"),
        avg_competitor_price=("competitor_price", "mean"),
        max_discount=("discount_pct", "max"),
        had_promotion=("is_promotion", "max"),
        is_holiday=("is_holiday", "max"),
    ).reset_index()
    agg_df = agg_df.sort_values(["store_id", "product_id", "sale_date"]).reset_index(drop=True)
    print(f"  Aggregated to {len(agg_df):,} daily store-product records")
    return agg_df


def create_lag_features(df, target="units_sold", lags=[1, 3, 7, 14, 30]):
    """
    Create lag features for time-series prediction.
    Business justification: Past demand is the strongest predictor of future demand.
    """
    df = df.sort_values(["store_id", "product_id", "sale_date"])
    for lag in lags:
        df[f"lag_{lag}d"] = df.groupby(["store_id", "product_id"])[target].shift(lag)
    print(f"  Created {len(lags)} lag features: {lags}")
    return df


def create_rolling_features(df, target="units_sold", windows=[7, 14, 30]):
    """
    Create rolling window statistics.
    Business justification: Smoothed averages capture medium-term demand trends.
    """
    df = df.sort_values(["store_id", "product_id", "sale_date"])
    for w in windows:
        grp = df.groupby(["store_id", "product_id"])[target]
        df[f"rolling_{w}d_mean"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"rolling_{w}d_std"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        df[f"rolling_{w}d_max"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
    print(f"  Created rolling features for windows: {windows}")
    return df


def create_price_features(df):
    """
    Create price-based features.
    Business justification: Price elasticity directly impacts demand volume.
    """
    # Price ratio (our price vs competitor)
    df["price_ratio"] = np.where(
        df["avg_competitor_price"] > 0,
        df["avg_price"] / df["avg_competitor_price"],
        1.0
    )
    # Price change from lag
    df["price_change_pct"] = df.groupby(["store_id", "product_id"])["avg_price"].pct_change()
    df["price_change_pct"] = df["price_change_pct"].fillna(0).clip(-1, 1)

    # Discount impact
    df["discount_flag"] = (df["max_discount"] > 0).astype(int)

    print("  Created price features: price_ratio, price_change_pct, discount_flag")
    return df


def create_seasonality_features(df):
    """
    Create time-based seasonality features.
    Business justification: Retail demand follows strong weekly, monthly, yearly cycles.
    Fourier components capture smooth seasonal patterns (MATLAB-inspired approach).
    """
    df["day_of_week"] = df["sale_date"].dt.dayofweek
    df["day_of_month"] = df["sale_date"].dt.day
    df["month"] = df["sale_date"].dt.month
    df["quarter"] = df["sale_date"].dt.quarter
    df["week_of_year"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)

    # Fourier seasonality components (inspired by MATLAB signal processing)
    day_of_year = df["sale_date"].dt.dayofyear
    # Annual cycle
    df["sin_annual"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_annual"] = np.cos(2 * np.pi * day_of_year / 365.25)
    # Weekly cycle
    df["sin_weekly"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_weekly"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # Monthly cycle
    df["sin_monthly"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_monthly"] = np.cos(2 * np.pi * df["month"] / 12)

    print("  Created seasonality features: calendar + 6 Fourier components")
    return df


def create_demand_features(df):
    """
    Create demand-derived features.
    Business justification: Demand velocity and volatility indicate inventory urgency.
    """
    df = df.sort_values(["store_id", "product_id", "sale_date"])

    # Demand velocity (change rate)
    df["demand_velocity"] = df.groupby(["store_id", "product_id"])["units_sold"].diff().fillna(0)

    # Demand volatility band
    rolling_std = df.get("rolling_7d_std", pd.Series(0, index=df.index))
    rolling_mean = df.get("rolling_7d_mean", pd.Series(1, index=df.index))
    cv = np.where(rolling_mean > 0, rolling_std / rolling_mean, 0)
    df["demand_volatility"] = np.clip(cv, 0, 5)
    df["volatility_band"] = pd.cut(
        df["demand_volatility"],
        bins=[-0.01, 0.3, 0.7, 1.5, 999],
        labels=["Stable", "Moderate", "Volatile", "Erratic"]
    ).astype(str)

    # Cumulative sales (year-to-date)
    df["ytd_cumulative"] = df.groupby(
        ["store_id", "product_id", df["sale_date"].dt.year]
    )["units_sold"].cumsum()

    print("  Created demand features: velocity, volatility_band, ytd_cumulative")
    return df


def drop_warmup_period(df, warmup_days=30):
    """Drop first N days where lag/rolling features are NaN."""
    min_date = df["sale_date"].min()
    cutoff = min_date + pd.Timedelta(days=warmup_days)
    before = len(df)
    df = df[df["sale_date"] >= cutoff].copy()
    print(f"  Dropped {before - len(df):,} warmup rows (first {warmup_days} days)")
    return df


def fill_remaining_nulls(df):
    """Fill any remaining NaN values from feature engineering."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
    return df


def select_model_features(df):
    """Select final features for modeling and encode categoricals."""
    # One-hot encode categorical columns
    cat_cols = ["volatility_band"]
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    # Drop non-feature columns (IDs, raw dates, strings)
    drop_cols = ["sale_date", "store_id", "product_id"]
    df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Drop any remaining object columns
    obj_cols = df_features.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df_features = df_features.drop(columns=obj_cols)

    print(f"  Final feature set: {df_features.shape[1]} columns")
    return df_features


def run_full_pipeline():
    """Execute the full feature engineering pipeline."""
    print("=" * 60 + "\nFEATURE ENGINEERING PIPELINE\n" + "=" * 60)

    df = load_cleaned_data()
    df = aggregate_daily(df)

    print("\nEngineering features...")
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_price_features(df)
    df = create_seasonality_features(df)
    df = create_demand_features(df)
    df = drop_warmup_period(df)
    df = fill_remaining_nulls(df)

    # Save full featured dataset (with IDs for dashboard)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/dashboard_data.csv", index=False)
    print(f"\nSaved: data/processed/dashboard_data.csv ({df.shape})")

    # Save encoded features for modeling
    df_encoded = select_model_features(df)
    df_encoded.to_csv("data/processed/features_encoded.csv", index=False)
    print(f"Saved: data/processed/features_encoded.csv ({df_encoded.shape})")

    return df


if __name__ == "__main__":
    run_full_pipeline()
