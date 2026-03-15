"""
generate_sample_data.py - Generate Synthetic Retail Sales Data
===============================================================
Generates 400K+ realistic retail sales records across 50 stores,
500 products, and 3 years of daily sales data.

Usage: python scripts/generate_sample_data.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────
NUM_STORES = 50
NUM_PRODUCTS = 500
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2023, 12, 31)
NUM_DAYS = (END_DATE - START_DATE).days + 1

# ── Store metadata ─────────────────────────────────────────────
REGIONS = ["North", "South", "East", "West", "Central"]
STORE_TYPES = ["Supermarket", "Express", "Warehouse", "Online"]

# ── Product metadata ───────────────────────────────────────────
CATEGORIES = [
    "Groceries", "Dairy", "Beverages", "Snacks", "Frozen",
    "Personal Care", "Household", "Electronics", "Clothing", "Toys"
]
SUB_CATEGORIES = {
    "Groceries": ["Rice", "Flour", "Oil", "Spices", "Pulses"],
    "Dairy": ["Milk", "Cheese", "Yogurt", "Butter", "Cream"],
    "Beverages": ["Juice", "Soda", "Water", "Tea", "Coffee"],
    "Snacks": ["Chips", "Cookies", "Crackers", "Nuts", "Bars"],
    "Frozen": ["Ice Cream", "Pizza", "Vegetables", "Meals", "Seafood"],
    "Personal Care": ["Shampoo", "Soap", "Lotion", "Toothpaste", "Deodorant"],
    "Household": ["Detergent", "Cleaner", "Trash Bags", "Paper Towels", "Sponges"],
    "Electronics": ["Batteries", "Cables", "Chargers", "Bulbs", "Adapters"],
    "Clothing": ["T-Shirts", "Socks", "Caps", "Gloves", "Scarves"],
    "Toys": ["Board Games", "Puzzles", "Action Figures", "Dolls", "Cars"],
}


def generate_stores():
    """Generate store metadata."""
    stores = []
    for i in range(1, NUM_STORES + 1):
        stores.append({
            "store_id": f"S{i:03d}",
            "region": np.random.choice(REGIONS),
            "store_type": np.random.choice(STORE_TYPES),
            "store_size_sqft": np.random.randint(2000, 50000),
            "opening_year": np.random.randint(2005, 2021),
        })
    return pd.DataFrame(stores)


def generate_products():
    """Generate product catalog with pricing."""
    products = []
    for i in range(1, NUM_PRODUCTS + 1):
        cat = np.random.choice(CATEGORIES)
        sub_cat = np.random.choice(SUB_CATEGORIES[cat])
        base_price = np.round(np.random.lognormal(2.5, 0.8), 2)
        base_price = np.clip(base_price, 1.0, 500.0)
        products.append({
            "product_id": f"P{i:04d}",
            "category": cat,
            "sub_category": sub_cat,
            "base_price": base_price,
            "cost_price": np.round(base_price * np.random.uniform(0.4, 0.75), 2),
            "is_perishable": int(cat in ["Groceries", "Dairy", "Frozen"]),
        })
    return pd.DataFrame(products)


def generate_sales(stores_df, products_df):
    """Generate 400K+ daily sales transactions with realistic patterns."""
    print("Generating sales data (this may take a moment)...")

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    # Pre-compute seasonal patterns
    day_of_year = np.arange(1, 367)
    # Annual seasonality: peaks in Nov-Dec (holiday), dip in Jan-Feb
    annual_pattern = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
    # Weekly pattern: weekends are 30% higher
    weekly_boost = {5: 1.3, 6: 1.35}  # Sat, Sun

    # Store-level base demand multiplier
    store_demand = {}
    for _, row in stores_df.iterrows():
        type_mult = {"Supermarket": 1.2, "Warehouse": 1.5, "Express": 0.7, "Online": 1.0}
        store_demand[row["store_id"]] = type_mult.get(row["store_type"], 1.0)

    # Product-level base demand
    product_demand = {}
    for _, row in products_df.iterrows():
        cat_mult = {
            "Groceries": 2.0, "Dairy": 1.8, "Beverages": 1.5, "Snacks": 1.3,
            "Frozen": 1.0, "Personal Care": 0.8, "Household": 0.7,
            "Electronics": 0.4, "Clothing": 0.3, "Toys": 0.3,
        }
        product_demand[row["product_id"]] = cat_mult.get(row["category"], 1.0)

    # Sample a subset of store-product-date combos (not all combos — too large)
    num_records = 420000
    sampled_stores = np.random.choice(stores_df["store_id"].values, num_records)
    sampled_products = np.random.choice(products_df["product_id"].values, num_records)
    sampled_dates = np.random.choice(dates, num_records)

    product_prices = products_df.set_index("product_id")["base_price"].to_dict()
    product_cats = products_df.set_index("product_id")["category"].to_dict()

    for i in range(num_records):
        dt = pd.Timestamp(sampled_dates[i])
        sid = sampled_stores[i]
        pid = sampled_products[i]

        # Base demand
        base = 10.0

        # Seasonal effect
        doy = min(dt.dayofyear, 365)
        seasonal = annual_pattern[doy - 1]

        # Weekly effect
        dow = dt.dayofweek
        weekly = weekly_boost.get(dow, 1.0)

        # Store & product multiplier
        s_mult = store_demand.get(sid, 1.0)
        p_mult = product_demand.get(pid, 1.0)

        # Random promotion (10% chance)
        is_promo = int(np.random.random() < 0.10)
        promo_mult = 1.5 if is_promo else 1.0

        # Holiday boost
        is_holiday = int(dt.month == 12 and dt.day >= 20) or int(dt.month == 11 and dt.day >= 25)
        holiday_mult = 1.8 if is_holiday else 1.0

        # Compute units sold
        mean_units = base * seasonal * weekly * s_mult * p_mult * promo_mult * holiday_mult
        units_sold = max(0, int(np.random.poisson(max(1, mean_units))))

        # Price with possible discount
        price = product_prices.get(pid, 10.0)
        discount_pct = np.random.choice([0, 5, 10, 15, 20, 25]) if is_promo else 0
        sell_price = round(price * (1 - discount_pct / 100), 2)

        # Competitor price (±15% of our price)
        competitor_price = round(price * np.random.uniform(0.85, 1.15), 2)

        records.append({
            "sale_date": dt.strftime("%Y-%m-%d"),
            "store_id": sid,
            "product_id": pid,
            "category": product_cats.get(pid, "Unknown"),
            "units_sold": units_sold,
            "unit_price": sell_price,
            "revenue": round(units_sold * sell_price, 2),
            "discount_pct": discount_pct,
            "is_promotion": is_promo,
            "is_holiday": is_holiday,
            "competitor_price": competitor_price,
            "day_of_week": dow,
            "month": dt.month,
            "year": dt.year,
        })

        if (i + 1) % 100000 == 0:
            print(f"  Generated {i+1:,}/{num_records:,} records...")

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("RETAIL SAMPLE DATA GENERATOR")
    print("=" * 60)

    stores_df = generate_stores()
    print(f"Generated {len(stores_df)} stores")

    products_df = generate_products()
    print(f"Generated {len(products_df)} products")

    sales_df = generate_sales(stores_df, products_df)
    print(f"Generated {len(sales_df):,} sales records")

    # Save
    os.makedirs("data/raw", exist_ok=True)
    stores_df.to_csv("data/raw/stores.csv", index=False)
    products_df.to_csv("data/raw/products.csv", index=False)
    sales_df.to_csv("data/raw/retail_sales.csv", index=False)

    print(f"\nSaved:")
    print(f"  data/raw/stores.csv        ({stores_df.shape})")
    print(f"  data/raw/products.csv      ({products_df.shape})")
    print(f"  data/raw/retail_sales.csv  ({sales_df.shape})")
    print(f"\nDate range: {sales_df['sale_date'].min()} to {sales_df['sale_date'].max()}")
    print(f"Avg units/record: {sales_df['units_sold'].mean():.1f}")
    print(f"Total revenue: ${sales_df['revenue'].sum():,.2f}")


if __name__ == "__main__":
    main()
