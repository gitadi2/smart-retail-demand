"""
Drift detection via Population Stability Index (PSI).

PSI compares a reference distribution (training window) to a current window.
Rule of thumb:  PSI < 0.1 stable | 0.1-0.25 moderate shift | > 0.25 major shift.

In production this runs on a schedule over incoming features and predictions; a
major shift fires a retraining job / pages the on-call. This is the difference
between "I trained a model once" and "I run a system."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODERATE, MAJOR = 0.10, 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    # Quantile bin edges from the reference, so bins are balanced.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def label(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value > MAJOR:
        return "MAJOR"
    if value > MODERATE:
        return "moderate"
    return "stable"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        if f in reference and f in current:
            v = psi(reference[f].to_numpy(), current[f].to_numpy())
            rows.append({"feature": f, "psi": round(v, 4), "status": label(v)})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    same = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.2, 1.4, 5000)
    print("no drift  PSI:", round(psi(ref, same), 4), label(psi(ref, same)))
    print("big drift PSI:", round(psi(ref, shifted), 4), label(psi(ref, shifted)))
