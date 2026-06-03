"""
Latency / throughput benchmark for the serving predict path.

Measures the model inference call (the dominant cost) directly, reporting
p50/p95/p99 single-request latency and batch throughput. This is the number an
SRE asks for -- "what's your p99?" -- and "R^2 = 0.79" can't answer it.

For an end-to-end HTTP benchmark, point a tool like `hey`/`wrk`/locust at the
running FastAPI service; this script isolates model cost from network/framework.

Run: python -m benchmarks.latency
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.data.generate import generate_demand
from src.features.engineering import TARGET, build_features, feature_columns
from src.models.train import DemandModel

ART = Path("artifacts")


def _percentiles(samples_ms: list[float]) -> dict:
    a = np.array(samples_ms)
    return {
        "p50_ms": round(float(np.percentile(a, 50)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "p99_ms": round(float(np.percentile(a, 99)), 4),
        "max_ms": round(float(a.max()), 4),
        "mean_ms": round(float(a.mean()), 4),
    }


def main(n_single: int = 3000, batch_size: int = 500):
    print("Preparing model + sample features...")
    raw = generate_demand(n_stores=5, n_products=10, n_days=400)
    feat = build_features(raw)
    cols = feature_columns(feat)
    train = feat.iloc[: int(len(feat) * 0.9)]
    model = DemandModel("hgb").fit(train[cols], train[TARGET])

    X = feat[cols].to_numpy(dtype=float)
    rng = np.random.default_rng(0)

    # warmup
    for _ in range(50):
        model.model.predict(X[rng.integers(0, len(X), 1)])

    print(f"Single-request latency over {n_single} calls...")
    lat = []
    for _ in range(n_single):
        row = X[rng.integers(0, len(X), 1)]
        t = time.perf_counter()
        model.model.predict(row)
        lat.append((time.perf_counter() - t) * 1000)
    single = _percentiles(lat)

    print(f"Batch throughput (batch={batch_size})...")
    n_batches, total = 200, 0
    t0 = time.perf_counter()
    for _ in range(n_batches):
        idx = rng.integers(0, len(X), batch_size)
        model.model.predict(X[idx])
        total += batch_size
    elapsed = time.perf_counter() - t0
    throughput = round(total / elapsed)

    result = {
        "single_request": single,
        "batch_throughput_pred_per_sec": throughput,
        "batch_size": batch_size,
        "n_features": len(cols),
        "engine": "HistGradientBoostingRegressor",
    }
    print(json.dumps(result, indent=2))
    ART.mkdir(exist_ok=True)
    (ART / "latency_benchmark.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
