"""
Synthetic demand generator + pluggable real-data loader.

Why synthetic at all? Because public datasets (M5, Rossmann, Favorita) can't be
shipped in this repo. The generator below is deliberately *honest*: demand is a
structured mean signal corrupted by irreducible Poisson noise. That noise caps
the achievable R^2/WMAPE at realistic levels, so a correct pipeline lands around
R^2 ~ 0.6-0.85 and WMAPE ~ 15-30% -- the range you actually see in production.
A model reporting R^2 = 0.998 here is leaking, full stop.

To use a real dataset, implement `load_real_dataset()` to return a frame with the
same schema (see SCHEMA in src/data/validation.py) and the rest of the pipeline
is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42


def generate_demand(
    n_stores: int = 15,
    n_products: int = 40,
    n_days: int = 600,
    start_date: str = "2023-01-01",
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Generate a daily (store, product) demand panel.

    Mean demand = base * trend * weekly_season * yearly_season * promo_lift
                  * price_elasticity * holiday_lift
    Realized demand ~ Poisson(mean)  <-- irreducible noise.

    Every driver here (price, promo flag, holiday flag, calendar) is KNOWN at
    forecast time, so using them as features is not leakage. The only thing the
    model cannot know is the Poisson draw -- which is the point.
    """
    rng = np.random.default_rng(seed)

    stores = [f"S{ i:03d}" for i in range(1, n_stores + 1)]
    products = [f"P{ i:04d}" for i in range(1, n_products + 1)]
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # Per-entity random effects (stable across time)
    store_mult = dict(zip(stores, rng.lognormal(0.0, 0.35, n_stores)))
    product_base = dict(zip(products, rng.lognormal(1.6, 0.5, n_products)))
    product_price = dict(zip(products, rng.uniform(2.0, 40.0, n_products)))
    product_elasticity = dict(zip(products, rng.uniform(-2.2, -0.6, n_products)))
    categories = rng.choice(
        ["Snacks", "Beverages", "Dairy", "Produce", "Household", "Frozen"],
        size=n_products,
    )
    product_cat = dict(zip(products, categories))
    # Category-level demand multipliers: in real retail, category + price are
    # genuinely informative about a product's demand level. This is exactly why
    # forecasting a brand-new product from similar existing products works.
    cat_mult = {
        "Beverages": 1.45, "Snacks": 1.20, "Produce": 1.10,
        "Dairy": 1.00, "Frozen": 0.85, "Household": 0.65,
    }

    # Calendar effects
    dow = dates.dayofweek.to_numpy()  # 0=Mon
    weekly = np.array([0.95, 0.90, 0.92, 1.00, 1.20, 1.55, 1.40])[dow]  # weekend lift
    doy = dates.dayofyear.to_numpy()
    yearly = 1.0 + 0.20 * np.sin(2 * np.pi * (doy - 80) / 365.25)  # spring/summer peak
    t = np.arange(n_days)
    trend = 1.0 + 0.0004 * t  # mild upward drift

    # Holidays (fixed set, known in advance)
    holiday_days = set(
        pd.to_datetime(
            [
                "2023-01-01", "2023-12-25", "2023-11-23", "2023-07-04",
                "2024-01-01", "2024-12-25", "2024-11-28", "2024-07-04",
            ]
        ).dayofyear.tolist()
    )

    frames = []
    for s in stores:
        for p in products:
            # Promotions: random ~8% of days, planned in advance
            promo = rng.random(n_days) < 0.08
            promo_lift = np.where(promo, rng.uniform(1.3, 2.2, n_days), 1.0)

            # Price = base, discounted on promo days
            price = np.where(promo, product_price[p] * rng.uniform(0.7, 0.85, n_days),
                             product_price[p])
            ref_price = product_price[p]
            # Constant-elasticity price response
            price_effect = (price / ref_price) ** product_elasticity[p]

            is_holiday = np.isin(pd.DatetimeIndex(dates).dayofyear, list(holiday_days))
            holiday_lift = np.where(is_holiday, 1.35, 1.0)

            mean = (
                product_base[p]
                * store_mult[s]
                * cat_mult[product_cat[p]]
                * trend
                * weekly
                * yearly
                * promo_lift
                * price_effect
                * holiday_lift
            )
            mean = np.clip(mean, 0.05, None)
            units = rng.poisson(mean)  # irreducible noise

            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store_id": s,
                        "product_id": p,
                        "category": product_cat[p],
                        "price": np.round(price, 2),
                        "is_promotion": promo,
                        "is_holiday": is_holiday,
                        "units_sold": units.astype(np.int32),
                    }
                )
            )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    return df


def load_real_dataset(path: str) -> pd.DataFrame:
    """Stub for swapping in M5 / Rossmann / Favorita.

    Return a frame matching SCHEMA in src/data/validation.py:
        date, store_id, product_id, category, price, is_promotion,
        is_holiday, units_sold
    Then run the same pipeline -- features, validation, and audit are agnostic.
    """
    raise NotImplementedError(
        "Map your real dataset columns to the project SCHEMA here, then return it."
    )


if __name__ == "__main__":
    d = generate_demand()
    print(d.shape)
    print(d.head())
    print("units_sold describe:\n", d["units_sold"].describe())
