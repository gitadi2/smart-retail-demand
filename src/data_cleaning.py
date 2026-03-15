"""
data_cleaning.py - Clean and preprocess retail sales data
==========================================================
Handles CSV-based data loading, merging, deduplication,
null handling, outlier capping, and validation.

Usage: python src/data_cleaning.py
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv()


def load_sales_data(path=None):
    """Load raw sales CSV data."""
    if path is None:
        for p in ["data/raw/retail_sales.csv", "data/raw/sales.csv"]:
            if os.path.exists(p):
                path = p
                break
    if path is None or not os.path.exists(path):
        print("ERROR: No sales data found. Run: python scripts/generate_sample_data.py")
        exit(1)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {path}")
    return df


def load_stores(path="data/raw/stores.csv"):
    """Load store metadata."""
    if not os.path.exists(path):
        print("WARNING: stores.csv not found, skipping store merge")
        return None
    return pd.read_csv(path)


def load_products(path="data/raw/products.csv"):
    """Load product catalog."""
    if not os.path.exists(path):
        print("WARNING: products.csv not found, skipping product merge")
        return None
    return pd.read_csv(path)


def merge_metadata(sales_df, stores_df, products_df):
    """Merge store and product metadata into sales data."""
    if stores_df is not None:
        sales_df = sales_df.merge(stores_df, on="store_id", how="left")
        print(f"  Merged store metadata ({len(stores_df)} stores)")
    if products_df is not None:
        # Avoid duplicate columns
        prod_cols = [c for c in products_df.columns if c not in sales_df.columns or c == "product_id"]
        sales_df = sales_df.merge(products_df[prod_cols], on="product_id", how="left")
        print(f"  Merged product metadata ({len(products_df)} products)")
    return sales_df


def parse_dates(df):
    """Parse sale_date to datetime and extract components."""
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df.dropna(subset=["sale_date"])
    df["day_of_week"] = df["sale_date"].dt.dayofweek
    df["day_of_month"] = df["sale_date"].dt.day
    df["week_of_year"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["month"] = df["sale_date"].dt.month
    df["quarter"] = df["sale_date"].dt.quarter
    df["year"] = df["sale_date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    print(f"  Date range: {df['sale_date'].min().date()} to {df['sale_date'].max().date()}")
    return df


def remove_duplicates(df):
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} duplicate rows")
    return df


def handle_missing(df):
    """Fill missing values intelligently."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    null_count = df.isnull().sum().sum()
    print(f"  Remaining nulls after fill: {null_count}")
    return df


def cap_outliers(df):
    """Cap numeric outliers at 1st/99th percentiles."""
    skip = ["is_promotion", "is_holiday", "is_weekend", "is_perishable",
            "day_of_week", "month", "year", "quarter", "day_of_month", "week_of_year"]
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in skip or col.endswith("_id"):
            continue
        q01, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
        if q01 != q99:
            df[col] = df[col].clip(q01, q99)
    print("  Outliers capped at 1st/99th percentiles")
    return df


def remove_negative_sales(df):
    """Remove rows with negative units or revenue."""
    before = len(df)
    df = df[df["units_sold"] >= 0]
    if "revenue" in df.columns:
        df = df[df["revenue"] >= 0]
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} rows with negative sales/revenue")
    return df


def validate(df):
    """Validate cleaned data quality."""
    checks = {
        "No nulls": df.isnull().sum().sum() == 0,
        "Rows > 100K": len(df) > 100000,
        "No negative units": (df["units_sold"] >= 0).all(),
        "Has date range": df["sale_date"].nunique() > 100,
        "Has multiple stores": df["store_id"].nunique() > 1,
    }
    print("\nValidation:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks.values())


def run_full_pipeline(source="csv"):
    """Execute the full data cleaning pipeline."""
    print("=" * 60 + "\nDATA CLEANING PIPELINE\n" + "=" * 60)

    # Load data
    sales_df = load_sales_data()
    stores_df = load_stores()
    products_df = load_products()

    # Process
    print("\nProcessing...")
    sales_df = merge_metadata(sales_df, stores_df, products_df)
    sales_df = parse_dates(sales_df)
    sales_df = remove_duplicates(sales_df)
    sales_df = remove_negative_sales(sales_df)
    sales_df = handle_missing(sales_df)
    sales_df = cap_outliers(sales_df)

    # Validate & save
    validate(sales_df)
    os.makedirs("data/processed", exist_ok=True)
    sales_df.to_csv("data/processed/cleaned_sales.csv", index=False)
    print(f"\nSaved: data/processed/cleaned_sales.csv ({sales_df.shape})")
    return sales_df


if __name__ == "__main__":
    run_full_pipeline()
