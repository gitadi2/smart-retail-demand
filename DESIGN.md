# Design: Retail Demand Forecasting

A short design doc — the kind a reviewer reads before the code. It states the
problem, the decisions, the alternatives rejected, and the failure mode this
project is built to avoid.

## 1. Problem

Forecast next-period unit demand per `(store, product, day)` so downstream
inventory logic can set reorder points. Inputs known at forecast time: calendar,
planned price, planned promotion, holiday calendar, and the product's own
demand history. Output: expected units.

Demand is a **count with irreducible noise** — even a perfect model of the mean
cannot predict the realised Poisson draw. So there is a hard ceiling on accuracy.
Any result near R² = 1.0 is evidence of a bug, not skill.

## 2. The failure mode this project is built around: leakage

The single most common, most resume-destroying mistake in time-series ML is
target leakage, usually in one of two forms:

1. **Same-row rolling features.** `groupby(key)['y'].rolling(7).mean()` includes
   `y[t]` in the window used to predict `y[t]`. The model effectively sees the
   answer. This alone produces the absurd "R² = 0.99 / MAPE = 2%" numbers.
2. **Random train/test split** on temporally ordered data, letting the model
   train on the future and test on the past.

### How we prevent it
- **Features (`src/features/engineering.py`).** Every rolling/expanding statistic
  is computed on the series **shifted by one** (`.shift(1)` then `.rolling(w)`),
  so the window ends at `t-1`. Lags are `groupby(...).shift(k)`. Calendar, price,
  and promo flags are known in advance and are safe.
- **Splitting (`src/eval/validation.py`).** Only chronological splits. The CV is
  rolling-origin walk-forward: train on the past, test the next 28 days, advance.
- **Automated audit (`src/audit/leakage.py`).** A *target-perturbation test*:
  perturb `y` at the last row of each series, rebuild features, and assert no
  feature value at that row changed. A leakage-free feature for row `t` depends
  only on rows `< t`, so perturbing `y[t]` cannot move it. This catches the
  same-row-rolling bug **regardless of how it's named or disguised**, and runs in
  CI as a hard gate before any model trains.

This is the difference between a project that *looks* good and one that survives
an interviewer who knows what to look for.

## 3. Validation & metrics

- **Walk-forward CV, 5 folds × 28-day horizon.** We quote the distribution
  (`WMAPE 25.7% ± 0.27`), not a single lucky split.
- **WMAPE** (`Σ|err| / Σ|actual|`) is the headline — robust to the zero-demand
  days that make plain MAPE explode. We also track MAE/RMSE/R²/bias.
- **Seasonal-naive baseline** (`y[t] = y[t-7]`) is computed on every fold. A
  model that can't beat it doesn't ship. Here: 48.3% → 25.7% WMAPE, a ~47%
  relative improvement. *That* is the number worth putting on a resume.

## 4. Model choice

Default engine is `HistGradientBoostingRegressor`:
- Handles missing values natively (lag/rolling NaNs at series start need no
  imputation hacks).
- Fast, strong on tabular data, zero external deps.
- Target is modelled as `log1p(units)` and inverted on predict (demand is
  non-negative and right-skewed; this stabilises variance).

XGBoost / LightGBM are wired behind optional imports (`src/models/train.py`):
installing them lights up extra engines with no code change.

**Why not deep learning (LSTM/GRU/attention)?** On tabular demand with strong
calendar/price structure and moderate series length, gradient boosting matches
or beats sequence models at a fraction of the training cost and complexity.
Deep nets earn their place with very long histories, rich exogenous sequences,
or shared cross-series representations — not demonstrated to help here, so they'd
be complexity for its own sake. (This is the trade-off question to expect; the
honest answer is "I didn't see lift to justify the cost," not "more models = better.")

## 5. Serving

Stateless FastAPI service (`src/serving/api.py`): model + feature order loaded
once at startup, `/predict` and `/predict/batch`, an LRU cache for repeated
feature vectors, and `/health` for probes. Keeping feature computation upstream
of the request keeps p99 predictable.

Measured predict-path latency (`benchmarks/latency.py`):
`p50 ≈ 1.2 ms, p95 ≈ 1.6 ms, p99 ≈ 2.0 ms`; batched throughput ≈ 79k pred/s
single-process. These are the numbers an SRE asks for; R² can't answer "what's
your p99."

## 6. Monitoring

PSI-based drift detection (`src/monitoring/drift.py`) compares a reference
(training) window to incoming data per feature; `PSI > 0.25` is a major shift
that should trigger retraining. Note: deterministic **calendar** features will
always show large PSI across a temporal split (different seasons) — that's
expected and should be excluded from alerting; the meaningful signals are the
behavioural features (rolling demand level, price).

## 7. What I'd do next (honest backlog)

- Swap the synthetic generator for a real dataset (M5 / Rossmann / Favorita) via
  `load_real_dataset` and benchmark against the public leaderboard.
- Conformal prediction intervals instead of point forecasts.
- Feature store + scheduled retraining wired to the drift alarm.
- Per-segment error analysis (which stores/categories are hardest, and why).

## 8. Reproducibility

Fixed seeds throughout; `python run_pipeline.py` regenerates every metric and
figure; the leakage audit + performance-regression test run in CI on every push.

## 9. Cold-start forecasting (vector search + embeddings)

**Problem.** A brand-new product has no sales history, so every lag/rolling
feature is null and the model can only guess its demand level. This "cold start"
is a real, persistent retail-forecasting pain.

**Approach.** Embed each product (category + price band via the metadata backend,
or an LLM/sentence-transformer embedding in production), index the embeddings, and
for a new product retrieve its *k* most similar existing products. The new
product's forecast borrows those neighbours' training-period weekday demand
profile as a `neighbour_prior` feature. A dedicated cold-start model uses only
launch-time-known features (price, calendar, promo) plus this prior — deliberately
not lags, since a new product has none.

**Why C++.** The similarity search is a `extern "C"` shared library
(`vector_index/index.cpp`) called from Python via ctypes — no pybind11. It uses
AVX2 dot products over L2-normalised vectors (cosine), contiguous memory, and a
binary save/load (mmap is the production path). Measured: ~18 µs search over 1k
vectors, ~0.6 ms over 50k (brute-force). For >~1M vectors, add IVF (k-means cells)
or HNSW — the brute-force core is the honest, benchmarked starting point, not a
half-built FAISS clone.

**Leakage-safety.** The prior for product P is built only from *other* products'
*training* demand, computed once from a frozen training snapshot and merged as a
static `[product_id, dow]` lookup. Perturbing P's own demand cannot change P's
prior — there's a unit test asserting exactly this (`tests/test_cold_start.py`).

**Measured value.** On a held-out set of products with their history deleted
(simulated launches), the neighbour prior cut cold-start WMAPE from **58.8% to
49.1% — a 16.5% improvement** on products the model had never seen. It doesn't
reach warm-start accuracy (~25%), which is the honest expectation: a prior
narrows cold-start error, it doesn't eliminate it.

Run: `python -m experiments.cold_start`
