"""
Neighbour demand prior for cold-start forecasting.

For each product, find its k most similar products (via the C++ vector index) and
borrow their TRAINING-period weekday demand profile. For a brand-new product with
no history, this is the only level signal available.

Leakage-safety (why the audit stays green):
  * The prior for product P is built from OTHER products' demand (neighbours),
    never P's own target -- perturbing P's y can't move P's prior.
  * Only rows with date <= cutoff feed the prior. No future information.
The prior is computed once from a frozen training snapshot and merged as a static
[product_id, dow] -> value lookup; the feature builder does not recompute it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cold_start.embeddings import embed_products
from src.cold_start.vector_index import VectorIndex, backend_name


def _train_profiles(train_df: pd.DataFrame) -> dict:
    """Per catalogue product: mean train demand by day-of-week (7 values)."""
    t = train_df.copy()
    t["dow"] = t["date"].dt.dayofweek
    prof = (
        t.groupby(["product_id", "dow"])["units_sold"].mean().unstack("dow")
        .reindex(columns=range(7))
    )
    # fill any missing dow with the product's own row mean
    prof = prof.apply(lambda r: r.fillna(r.mean()), axis=1)
    return {pid: prof.loc[pid].to_numpy(dtype=float) for pid in prof.index}


def build_prior_table(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 5,
    backend: str = "metadata",
) -> pd.DataFrame:
    """Return a long table [product_id, dow, neighbour_prior].

    Index (catalogue) = products that HAVE training history. Every product in
    `full_df` (including new ones) queries the index for its k nearest catalogue
    neighbours; the prior is the similarity-weighted mean of their train weekday
    profiles. New products borrow entirely; existing products get a mild extra
    level signal.
    """
    pids_all, vecs_all = embed_products(full_df, backend=backend)
    vec_of = {int_pid: vecs_all[i] for i, int_pid in enumerate(range(len(pids_all)))}
    pid_to_row = {pid: i for i, pid in enumerate(pids_all)}

    catalogue = list(train_df["product_id"].unique())
    cat_rows = [pid_to_row[p] for p in catalogue]
    profiles = _train_profiles(train_df)

    dim = vecs_all.shape[1]
    index = VectorIndex(dim)
    index.add(np.array(cat_rows, dtype=np.int64), vecs_all[cat_rows])

    records = []
    for pid in pids_all:
        qvec = vecs_all[pid_to_row[pid]]
        nbr_rows, sims = index.search(qvec, k=k + 1)  # +1 in case self is in catalogue
        # map rows back to product_ids, drop self
        nbr_pids, nbr_sims = [], []
        for row, sim in zip(nbr_rows, sims):
            npid = pids_all[row]
            if npid == pid:
                continue
            nbr_pids.append(npid)
            nbr_sims.append(max(float(sim), 1e-6))
            if len(nbr_pids) == k:
                break
        if not nbr_pids:
            continue
        w = np.array(nbr_sims)
        w = w / w.sum()
        prior = np.zeros(7)
        for wi, npid in zip(w, nbr_pids):
            prior += wi * profiles[npid]
        for d in range(7):
            records.append({"product_id": pid, "dow": d, "neighbour_prior": prior[d]})

    return pd.DataFrame(records)


def attach_prior(df: pd.DataFrame, prior_table: pd.DataFrame) -> pd.DataFrame:
    """Merge the [product_id, dow] prior onto a panel as `neighbour_prior`."""
    out = df.copy()
    if "dow" not in out.columns:
        out["dow"] = out["date"].dt.dayofweek
    out = out.merge(prior_table, on=["product_id", "dow"], how="left")
    return out


if __name__ == "__main__":
    from src.data.generate import generate_demand
    from src.eval.validation import temporal_split

    df = generate_demand(n_stores=4, n_products=30, n_days=400)
    train, _, cutoff = temporal_split(df, test_days=56)
    table = build_prior_table(df, train, k=5)
    print("index backend:", backend_name())
    print("prior rows:", len(table), "products:", table["product_id"].nunique())
    print(table.head(8).to_string(index=False))
