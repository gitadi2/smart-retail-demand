"""
Human-readable cold-start demo: for a few "new" products, show which existing
products the vector index matched, and the demand level it borrowed vs reality.

Run: python -m experiments.cold_start_demo
"""
from __future__ import annotations

import numpy as np

from src.cold_start.embeddings import embed_products, product_metadata
from src.cold_start.vector_index import VectorIndex, backend_name
from src.data.generate import generate_demand
from src.eval.validation import temporal_split


def main(seed: int = 7):
    print("vector index backend:", backend_name(), "\n")
    raw = generate_demand(n_stores=8, n_products=40, n_days=540)
    train, test, _ = temporal_split(raw, test_days=70)

    meta = product_metadata(raw).set_index("product_id")
    pids, vecs = embed_products(raw, backend="metadata")
    row_of = {p: i for i, p in enumerate(pids)}

    # catalogue = products with history; index them
    catalogue = sorted(train["product_id"].unique())
    idx = VectorIndex(vecs.shape[1])
    idx.add(np.array([row_of[p] for p in catalogue], dtype=np.int64),
            vecs[[row_of[p] for p in catalogue]])

    train_level = train.groupby("product_id")["units_sold"].mean()
    test_level = test.groupby("product_id")["units_sold"].mean()

    rng = np.random.default_rng(seed)
    new_products = rng.choice(catalogue, size=3, replace=False)

    for p in new_products:
        cat, price = meta.loc[p, "category"], meta.loc[p, "price"]
        nbr_rows, sims = idx.search(vecs[row_of[p]], k=4)
        nbrs = [(pids[r], s) for r, s in zip(nbr_rows, sims) if pids[r] != p][:3]
        prior = np.average([train_level[n] for n, _ in nbrs],
                           weights=[s for _, s in nbrs])
        print(f"NEW product {p}  [{cat}, ${price:.2f}]")
        print("  nearest existing products (cosine sim):")
        for n, s in nbrs:
            print(f"    {n}  [{meta.loc[n,'category']}, ${meta.loc[n,'price']:.2f}]"
                  f"  sim={s:.2f}  avg demand={train_level[n]:.1f}")
        print(f"  --> borrowed demand level (prior): {prior:.1f}")
        print(f"      actual avg demand of {p}:        {test_level[p]:.1f}")
        print(f"      (naive global-mean guess would be: {train_level.mean():.1f})\n")


if __name__ == "__main__":
    main()
