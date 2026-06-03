"""
Cold-start experiment: does the neighbour prior actually help forecast NEW
products that have no sales history?

Setup:
  * Temporal train/test split.
  * Pick a set of "new" products and DELETE their training rows (they launch at
    test time). Their test rows remain, to be forecast.
  * Catalogue = products that still have training history.
  * Train two models on the catalogue's training rows:
       (a) WITHOUT prior  -- standard features.
       (b) WITH prior     -- standard features + neighbour_prior.
  * Forecast the NEW products' test rows with each, compare WMAPE.

A model with no history for a product is basically guessing its level; the prior
supplies that level from similar products. The gap is the value of the feature.

Run: python -m experiments.cold_start
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.cold_start.neighbor_prior import attach_prior, build_prior_table
from src.cold_start.vector_index import backend_name
from src.data.generate import generate_demand
from src.eval.metrics import wmape
from src.eval.validation import temporal_split
from src.features.engineering import TARGET, build_features, feature_columns
from src.models.train import DemandModel

ART = Path("artifacts")


def main(n_products: int = 40, n_new: int = 10, seed: int = 7):
    rng = np.random.default_rng(seed)
    print("index backend:", backend_name())

    raw = generate_demand(n_stores=8, n_products=n_products, n_days=540)
    all_products = sorted(raw["product_id"].unique())
    new_products = set(rng.choice(all_products, size=n_new, replace=False).tolist())
    print(f"products={n_products}  held out as NEW={n_new}")

    # temporal split
    _, _, cutoff = temporal_split(raw, test_days=70)
    is_train_period = raw["date"] <= cutoff
    is_new = raw["product_id"].isin(new_products)

    # NEW products have NO training history (their train rows are deleted)
    train_mask = is_train_period & ~is_new
    test_mask = ~is_train_period

    # features on the full panel (leakage-safe); new products' train rows removed first
    panel = raw[~(is_train_period & is_new)].copy()
    feat = build_features(panel).reset_index(drop=True)
    cols = feature_columns(feat)

    # prior table from catalogue training data only
    t0 = time.perf_counter()
    train_df = raw[train_mask]
    prior_table = build_prior_table(raw, train_df, k=6)
    feat_p = attach_prior(feat, prior_table)
    prior_build_ms = (time.perf_counter() - t0) * 1000

    # Cold-start regime: a new product has NO history, so a cold-start model must
    # not depend on lag/rolling features. Restrict to features known at launch.
    history_prefixes = ("lag_", "roll_mean_", "roll_std_", "expanding_mean", "price_vs_roll")
    cold_cols = [c for c in cols if not c.startswith(history_prefixes)]
    cold_cols_p = cold_cols + ["neighbour_prior"]

    # row masks aligned to feat / feat_p
    f_is_new = feat["product_id"].isin(new_products)
    f_is_test = feat["date"] > cutoff
    f_train = (feat["date"] <= cutoff) & ~f_is_new  # catalogue train rows
    eval_mask = f_is_new & f_is_test                # NEW products, test period

    # ---- cold-start model WITHOUT prior (price + calendar only) ----
    m0 = DemandModel("hgb").fit(feat.loc[f_train, cold_cols], feat.loc[f_train, TARGET])
    pred0 = m0.predict(feat.loc[eval_mask, cold_cols])
    w0 = wmape(feat.loc[eval_mask, TARGET], pred0)

    # ---- cold-start model WITH neighbour prior ----
    m1 = DemandModel("hgb").fit(feat_p.loc[f_train, cold_cols_p], feat_p.loc[f_train, TARGET])
    pred1 = m1.predict(feat_p.loc[eval_mask, cold_cols_p])
    w1 = wmape(feat_p.loc[eval_mask, TARGET], pred1)

    result = {
        "index_backend": backend_name(),
        "n_products": n_products,
        "n_new_products": n_new,
        "new_product_eval_rows": int(eval_mask.sum()),
        "cold_start_wmape_without_prior": round(w0, 2),
        "cold_start_wmape_with_prior": round(w1, 2),
        "relative_improvement_pct": round((w0 - w1) / w0 * 100, 1),
        "prior_table_build_ms": round(prior_build_ms, 1),
    }
    print(json.dumps(result, indent=2))
    ART.mkdir(exist_ok=True)
    (ART / "cold_start_result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
