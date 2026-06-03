"""
Product embedding pipeline -- turns product metadata into vectors for the index.

Two backends, SAME downstream:
  * "metadata"  : deterministic vector from category + price band. Runs offline,
                  no model needed. Good enough to demonstrate cold-start lift.
  * "llm"       : production path -- embed a product text/description with a
                  sentence-transformer or embedding API. Identical interface;
                  only the vector source changes. Stubbed here (needs a model).

The point: the C++ index, the neighbour prior, and the forecasting pipeline are
all backend-agnostic. Swapping metadata -> LLM embeddings is a one-line change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["Snacks", "Beverages", "Dairy", "Produce", "Household", "Frozen"]


def product_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the panel to one row per product: category + typical price."""
    meta = (
        df.groupby("product_id")
        .agg(category=("category", "first"), price=("price", "median"))
        .reset_index()
    )
    return meta


def embed_metadata(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic embedding: one-hot category (weighted) + log-price scalar.

    Returns (product_index_array, vectors[n, dim]). Category dominates so
    same-category products cluster; price separates within a category.
    """
    cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
    n = len(meta)
    dim = len(CATEGORIES) + 1
    vecs = np.zeros((n, dim), dtype=np.float32)
    for r, (_, row) in enumerate(meta.iterrows()):
        vecs[r, cat_idx[row["category"]]] = 3.0  # category weight
        vecs[r, -1] = np.log1p(row["price"])      # price dimension
    # standardise the price dim so it's comparable to the one-hot scale
    vecs[:, -1] = (vecs[:, -1] - vecs[:, -1].mean()) / (vecs[:, -1].std() + 1e-9)
    return meta["product_id"].to_numpy(), vecs


def embed_llm(meta: pd.DataFrame, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Production embedding path. Requires a model + (usually) network.

    Build a short text per product ("<category> priced around <price>", or a real
    catalogue description) and embed it:

        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(model)
        texts = [f"{r.category} product priced ${r.price:.2f}" for r in meta.itertuples()]
        return meta["product_id"].to_numpy(), enc.encode(texts, normalize_embeddings=True)
    """
    raise NotImplementedError(
        "Install sentence-transformers (and allow model download) to use LLM "
        "embeddings; the metadata backend is the offline default."
    )


def embed_products(df: pd.DataFrame, backend: str = "metadata"):
    meta = product_metadata(df)
    if backend == "metadata":
        return embed_metadata(meta)
    if backend == "llm":
        return embed_llm(meta)
    raise ValueError(backend)
