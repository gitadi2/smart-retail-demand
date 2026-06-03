"""
Cold-start / vector-index tests. pytest-compatible and runnable directly.

The leakage test is the important one: the neighbour prior for a product must NOT
depend on that product's own demand, only on its neighbours' training demand.
"""
from __future__ import annotations

import numpy as np

from src.cold_start.neighbor_prior import build_prior_table
from src.cold_start.vector_index import VectorIndex, backend_name
from src.data.generate import generate_demand
from src.eval.validation import temporal_split


def test_index_nearest_is_self():
    rng = np.random.default_rng(0)
    idx = VectorIndex(8)
    vecs = rng.normal(size=(50, 8)).astype(np.float32)
    idx.add(np.arange(50), vecs)
    ids, sims = idx.search(vecs[10], k=3)
    assert ids[0] == 10
    assert sims[0] > 0.999  # cosine with itself ~ 1.0


def test_prior_table_covers_all_products():
    df = generate_demand(n_stores=3, n_products=20, n_days=300)
    train, _, _ = temporal_split(df, test_days=56)
    table = build_prior_table(df, train, k=5)
    assert set(table["product_id"].unique()) == set(df["product_id"].unique())
    assert (table["neighbour_prior"] > 0).all()


def test_prior_does_not_use_own_demand():
    """Perturbing a product's OWN demand must not change ITS prior (the prior is
    built from neighbours' training demand only)."""
    df = generate_demand(n_stores=3, n_products=20, n_days=300)
    train, _, _ = temporal_split(df, test_days=56)

    target = sorted(df["product_id"].unique())[0]
    base = build_prior_table(df, train, k=5)
    base_target = base[base["product_id"] == target]["neighbour_prior"].to_numpy()

    # 10x the target product's demand everywhere
    df2 = df.copy()
    df2.loc[df2["product_id"] == target, "units_sold"] *= 10
    train2, _, _ = temporal_split(df2, test_days=56)
    pert = build_prior_table(df2, train2, k=5)
    pert_target = pert[pert["product_id"] == target]["neighbour_prior"].to_numpy()

    assert np.allclose(base_target, pert_target), "prior leaked the product's own demand"


if __name__ == "__main__":
    print("backend:", backend_name())
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
