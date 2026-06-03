# RETAIL DEMAND FORECASTING (Leakage-Safe MLOps) + DYNAMIC PROGRAMMING INVENTORY SOLVER

End-to-end retail demand forecasting system — built so the accuracy number is
**trustworthy**: leakage-safe features, rolling-origin walk-forward validation, an
automated leakage audit that gates CI, drift monitoring, and a latency-benchmarked
FastAPI service — plus DSA-optimized inventory allocation, all deployable on AWS.

> **Headline:** on realistic, noisy demand the model reaches **WMAPE 25.7% (R² 0.79)**,
> a **~47% improvement over a seasonal-naive baseline** — and a built-in audit proves
> no feature leaks the target. (A retail demand model claiming R² ≈ 0.99 is leaking;
> this project is built to prove it isn't.)

<p align="left">
<a href="https://www.python.org" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/></a>
<a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikitlearn/scikitlearn-original.svg" alt="scikit-learn" width="40" height="40"/></a>
<a href="https://xgboost.readthedocs.io/" target="_blank" rel="noreferrer"><img src="https://upload.wikimedia.org/wikipedia/commons/6/69/XGBoost_logo.png" alt="xgboost" width="40" height="40"/></a>
<a href="https://lightgbm.readthedocs.io/" target="_blank" rel="noreferrer"><img src="https://lightgbm.readthedocs.io/en/latest/_images/LightGBM_logo_black_text.svg" alt="lightgbm" width="70" height="40"/></a>
<a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg" alt="tensorflow" width="40" height="40"/></a>
<a href="https://keras.io/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/keras/keras-original.svg" alt="keras" width="40" height="40"/></a>
<a href="https://fastapi.tiangolo.com/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/fastapi/fastapi-original.svg" alt="fastapi" width="40" height="40"/></a>
<a href="https://www.postgresql.org" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/postgresql/postgresql-original.svg" alt="postgresql" width="40" height="40"/></a>
<a href="https://aws.amazon.com" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/amazonwebservices/amazonwebservices-original-wordmark.svg" alt="aws" width="40" height="40"/></a>
<a href="https://www.docker.com/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original.svg" alt="docker" width="40" height="40"/></a>
<a href="https://mlflow.org/" target="_blank" rel="noreferrer"> <img src="https://cdn.simpleicons.org/mlflow/0194E2" alt="mlflow" width="40" height="40"/> </a>
<a href="https://git-scm.com/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/git/git-original.svg" alt="git" width="40" height="40"/></a>
<a href="https://github.com/features/actions" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/githubactions/githubactions-original.svg" alt="github-actions" width="40" height="40"/></a>
<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/></a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="numpy" width="40" height="40"/></a>
<a href="https://matplotlib.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/matplotlib/matplotlib-original.svg" alt="matplotlib" width="40" height="40"/></a>
<a href="https://docs.pytest.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytest/pytest-original.svg" alt="pytest" width="40" height="40"/></a>
<a href="https://www.tableau.com/" target="_blank" rel="noreferrer"><img src="https://cdn.worldvectorlogo.com/logos/tableau-software.svg" alt="tableau" width="40" height="40"/></a>
<a href="#algorithms--data-structures" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/TheAlgorithms/website/main/public/logo.svg" alt="dsa" width="40" height="40"/></a>
</p>

---

## Why this is built differently

| Common pitfall | This project |
|---|---|
| `rolling(7).mean()` on the target (includes `y[t]` → leak) | rolling stats on the `.shift(1)` series — window ends at `t-1` |
| random train/test split on time-ordered data | chronological split + rolling-origin walk-forward CV |
| a single headline accuracy number | distribution across 5 folds, always vs a baseline |
| no leakage check | automated **target-perturbation audit** as a CI hard gate |
| "trained a model" | drift detection, performance-regression tests, latency SLOs |

See **[DESIGN.md](DESIGN.md)** for the reasoning behind every choice — including
the leakage story and why gradient boosting over deep nets here.

---

## Architecture

```
┌────────────┐   ┌────────────┐   ┌──────────────┐   ┌──────────────────────┐
│  Raw / M5  │──▶│  Cleaning  │──▶│  Validation  │──▶│  LEAKAGE AUDIT (gate) │
│   data     │   │ dedup/types│   │ data contract│   │ target-perturbation   │
└────────────┘   └────────────┘   └──────────────┘   └──────────┬───────────┘
                                                                 │ pass
                                          ┌──────────────────────▼───────────┐
                                          │  Leakage-safe Feature Engineering │
                                          │  lag + rolling on shift(1), cyclic│
                                          └──────────────────────┬───────────┘
                                                                 │
                  ┌──────────────────────────────────────────────▼─────────────┐
                  │  Walk-forward CV (rolling origin)  vs  seasonal-naive base  │
                  │  → train best model (HGB / XGBoost / LightGBM, log1p target)│
                  └───────────────┬───────────────────────────┬─────────────────┘
                                  │                            │
                ┌─────────────────▼──────┐        ┌────────────▼───────────┐
                │  Inventory Optimizer   │        │   Drift Monitor (PSI)  │
                │   DP + Binary Search   │        │   retrain trigger      │
                └─────────────────┬──────┘        └────────────────────────┘
                                  │
                  ┌───────────────▼───────────────┐
                  │  FastAPI /predict (LRU cache)  │   p99 ≈ 2 ms
                  └───────────────┬───────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │   AWS: EC2 + S3 + RDS    │
                     └──────────────────────────┘
```

---

## Model Performance

> Numbers are from the leakage-safe pipeline (`python run_pipeline.py`), reproducible
> with a fixed seed. Swap in a real dataset (M5 / Rossmann / Favorita) to benchmark
> against public leaderboards — see [Using a real dataset](#using-a-real-dataset).

**Walk-forward CV — 5 folds × 28-day horizon (WMAPE, lower is better):**

| Model | WMAPE | vs baseline |
|-------|-------|-------------|
| **Gradient Boosting (HGB / XGBoost / LightGBM)** | **25.7% ± 0.27** | **−47%** |
| Seasonal-naive (baseline) | 48.3% ± 0.51 | — |

**Final temporal holdout (last 56 days):** WMAPE 25.8% · R² 0.79 · MAE 2.72 · bias −1.02

Gradient boosting is the production engine (handles missing lag values natively,
fast, strong on tabular data). Deep sequence models (LSTM / BiGRU / CNN-LSTM /
Attention) were evaluated but did not beat gradient boosting on this calendar- and
price-driven tabular data, so they were not worth the training cost — a deliberate
trade-off, documented in [DESIGN.md](DESIGN.md).

<p align="center">
  <img src="reports/figures/cv_wmape.png" width="70%" alt="Walk-forward CV: model vs seasonal-naive baseline"/>
</p>

---

## Leakage safety (the part that matters)

`src/audit/leakage.py` runs a **target-perturbation test**: perturb `y` at the last
row of each series, rebuild features, and assert nothing at that row moved. A
leakage-free feature for row `t` uses only rows `< t`, so it can't move. This catches
the same-row-rolling-mean bug regardless of how it's named, and runs in CI before any
model trains:

```
$ python -m src.audit.leakage
=== Clean pipeline ===   target_perturbation: OK: no feature uses y[t]    -> PASS
=== Leaky pipeline ===   target_perturbation: LEAK via ['roll_mean_7_LEAK'] -> FAIL
```

---

## Inventory Optimization

Dynamic Programming and Binary Search optimize inventory allocation across stores:

| Metric | Value |
|--------|-------|
| Fill Rate | 100.0% |
| Inventory Used | 265 units |
| Inventory Remaining | 4,735 units |
| Safety Stock | 147 units |
| Reorder Point | 245 units |
| Service Level | 95% |
| Anomalies Detected | 0 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML (production)** | scikit-learn HistGradientBoosting, optional XGBoost / LightGBM · log1p target transform |
| **Deep learning (explored)** | TensorFlow/Keras — LSTM, BiGRU, CNN-LSTM, Attention |
| **Validation** | Rolling-origin walk-forward CV, chronological splits, WMAPE / MAE / RMSE / R² / bias |
| **Quality gates** | Target-perturbation leakage audit, data-contract validation, performance-regression tests |
| **Monitoring** | PSI drift detection |
| **DSA** | Dynamic Programming, Binary Search, Sliding Window, Min-Heap, LRU Cache, Hash Map |
| **API** | FastAPI, Uvicorn, Pydantic validation, LRU cache (p99 ≈ 2 ms) |
| **Database** | PostgreSQL on AWS RDS, SQLAlchemy ORM |
| **Cloud** | AWS EC2, S3, RDS |
| **Experiment Tracking** | MLflow |
| **DevOps** | Docker, GitHub Actions CI/CD (audit + tests), Git |
| **Visualization** | Tableau, Matplotlib, Seaborn, Chart.js |

---

## Project Structure

```
smart-retail-demand/
├── run_pipeline.py              # generate → validate → AUDIT(gate) → features → walk-forward CV → train → optimize → drift
├── requirements.txt
├── Dockerfile
├── README.md
├── DESIGN.md                    # NEW: decisions, trade-offs, the leakage story
├── .github/workflows/ci.yml     # CI: leakage audit gate + tests on every push
│
├── src/
│   ├── data/
│   │   ├── generate.py          # synthetic generator (honest noise) + real-dataset loader stub
│   │   ├── data_cleaning.py     # type casting, dedup, derived columns
│   │   └── validation.py        # NEW: schema + data-contract checks (hard gate)
│   ├── features/
│   │   └── engineering.py       # LEAKAGE-SAFE: lag + rolling on shift(1), cyclical encoding
│   ├── eval/
│   │   ├── metrics.py           # NEW: WMAPE, MAE, RMSE, R², bias
│   │   └── validation.py        # NEW: temporal split + rolling-origin walk-forward CV
│   ├── models/
│   │   ├── train.py             # gradient boosting (HGB/XGBoost/LightGBM), log1p target
│   │   └── baselines.py         # NEW: seasonal-naive baseline
│   ├── audit/
│   │   └── leakage.py           # NEW: target-perturbation leakage audit (CI hard gate)
│   ├── monitoring/
│   │   └── drift.py             # NEW: PSI drift detection
│   ├── inventory_optimizer.py   # DP allocation, binary search reorder, sliding window
│   ├── api/
│   │   ├── forecasting_api.py   # FastAPI: /predict, /batch, /inventory — serves the safe model
│   │   └── schemas.py           # Pydantic request/response models
│   └── utils/
│       ├── algorithms.py        # DP, binary search, sliding window, min-heap
│       └── data_structures.py   # LRU Cache, SortedDemandArray, DemandBucketMap
│
├── sql/
│   ├── 01_create_schema.sql     # PostgreSQL schema
│   ├── 02_create_tables.sql     # Table definitions
│   ├── 03_etl_pipeline.sql      # SQL-based ETL
│   ├── 04_feature_engineering.sql   # NOTE: offset rolling windows by 1 row (no same-row aggregates)
│   └── 05_analytics_views.sql   # Aggregated views for dashboards
│
├── tests/
│   ├── test_leakage.py          # NEW: leakage regression + performance-regression gate
│   ├── test_algorithms.py       # DP, binary search, sliding window tests
│   ├── test_api.py              # API schema validation tests
│   └── test_data_structures.py  # LRU cache, sorted array, bucket map tests
│
├── data/
│   ├── raw/                     # retail_sales.csv, products.csv, stores.csv (or M5)
│   └── processed/               # cleaned_sales.csv, model_metrics.csv
│
├── models/                      # Trained model + feature_order.json + metrics
├── reports/figures/             # cv_wmape.png, inventory & comparison charts
├── benchmarks/
│   └── latency.py               # NEW: p50/p95/p99 + batch throughput
├── dashboards/                  # Interactive HTML dashboard
└── screenshots/                 # Tableau + Swagger UI + AWS Cloud screenshots
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/gitadi2/smart-retail-demand.git
cd smart-retail-demand
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp config/.env.example .env
# Edit .env with your database credentials
```

### 3. Run Full Pipeline

```bash
python run_pipeline.py
```

Stages:
1. **Generate / load** demand data (synthetic with irreducible noise, or a real dataset)
2. **Validate** — data-contract checks (no negatives, no dupes, contiguous dates)
3. **Leakage audit** — target-perturbation test; pipeline halts if any feature leaks
4. **Feature engineering** — leakage-safe lags + shifted rolling stats + cyclical encoding
5. **Walk-forward CV** — rolling-origin, vs seasonal-naive baseline; train best model
6. **Optimize** — DP inventory allocation, binary search reorder points
7. **Drift report** — PSI between training and holdout windows

### 4. Verify the leakage audit & benchmarks

```bash
python -m src.audit.leakage     # pass/fail demo on clean vs leaky features
python -m benchmarks.latency    # p50/p95/p99 + throughput
```

### 5. Launch Forecasting API

```bash
uvicorn src.api.forecasting_api:app --port 8000
```

Open Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 6. Run Tests

```bash
pytest tests/ -v                # includes leakage + performance-regression gates
```

---

## Using a real dataset

The pipeline is dataset-agnostic. Implement `load_real_dataset()` in
`src/data/generate.py` to return the project schema (`date, store_id, product_id,
category, price, is_promotion, is_holiday, units_sold`) from **M5 / Rossmann /
Favorita**, and every stage — validation, audit, features, CV, serving — works
unchanged, with WMAPE now directly comparable to public leaderboards.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status, cache stats, version |
| `/predict` | POST | Single demand forecast |
| `/predict/batch` | POST | Batch forecast (up to 500 items) |
| `/inventory/allocate` | POST | DP-based inventory allocation across stores |
| `/cache/stats` | GET | Cache utilization metrics |
| `/cache/clear` | POST | Clear prediction cache |

**Serving latency (predict path):** p50 ≈ 1.2 ms · p95 ≈ 1.6 ms · p99 ≈ 2.0 ms · ~79k pred/s batched

**Example Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "S012",
    "product_id": "P0133",
    "category": "Snacks",
    "month": 10,
    "day_of_week": 4,
    "is_promotion": false,
    "is_holiday": false
  }'
```

**Example Response:**

```json
{
  "store_id": "S012",
  "product_id": "P0133",
  "predicted_demand": 15.3,
  "confidence_interval": {"lower": 12.1, "upper": 18.5},
  "model_used": "HistGradientBoosting",
  "cached": false
}
```

---

## Algorithms & Data Structures

| Component | Complexity | Purpose |
|-----------|-----------|---------|
| **DP Inventory Allocation** | O(n × W) | Optimal stock distribution across stores |
| **Binary Search Reorder** | O(log n) | Find optimal reorder point for service level |
| **Sliding Window** | O(n) single pass | Rolling demand anomaly detection |
| **Min-Heap Top-K** | O(n log k) | Identify top stockout risk products |
| **LRU Cache** | O(1) get/put | Cache repeated prediction requests |
| **SortedDemandArray** | O(log n) query | Fast percentile & threshold lookups |
| **DemandBucketMap** | O(1) lookup | Demand aggregation by segment |

---

## Screenshots

<p align="center">
  <img src="screenshots/swagger_api_smart_retail.jpg" width="80%" alt="Swagger API"/>
</p>

<p align="center">
  <img src="screenshots/smart_retail_demand_and_inventory_opt_dashboard.jpg" width="80%" alt="Tableau Dashboard"/>
</p>

<p align="center">
  <img src="screenshots/revenue_trend_1_tableau.jpg" width="45%" alt="Revenue Trend"/>
  <img src="screenshots/revenue_by_category_2_tableau.jpg" width="45%" alt="Revenue by Category"/>
</p>

<p align="center">
  <img src="screenshots/regional_performance_3_tableau.jpg" width="45%" alt="Regional Performance"/>
  <img src="screenshots/store_type_analysis_4_tableau.jpg" width="45%" alt="Store Type Analysis"/>
</p>

<p align="center">
  <img src="screenshots/promotion_impact_5_tableau.jpg" width="45%" alt="Promotion Impact"/>
  <img src="screenshots/weekly_heatmap_6_tableau.jpg" width="45%" alt="Weekly Heatmap"/>
</p>

<p align="center">
  <img src="screenshots/discount_analysis_7_tableau.jpg" width="45%" alt="Discount Analysis"/>
  <img src="screenshots/model_comparison_8_tableau.jpg" width="45%" alt="Model Comparison"/>
</p>

<p align="center">
  <img src="screenshots/aws_console_home.png" width="45%" alt="AWS Console Home"/>
  <img src="screenshots/ec2_instance_running.png" width="45%" alt="EC2 Instance Running"/>
</p>

<p align="center">
  <img src="screenshots/api_live_on_aws.png" width="45%" alt="API Live on AWS"/>
  <img src="screenshots/rds_instance_details.png" width="45%" alt="RDS Database"/>
</p>

---

## Interactive Dashboard

- **Tableau Public (online)**: [SMART RETAIL DEMAND DASHBOARD](https://public.tableau.com/views/SMARTRETAILDEMANDDASHBOARD/SMARTRETAILDEMANDINVENTORYOPTIMIZATIONDASHBOARD?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- **Local HTML**: [INTERACTIVE DASHBOARD](https://raw.githack.com/gitadi2/smart-retail-demand/master/dashboards/retail_demand_dashboard.html)

---

## Docker (Optional)

> **Prerequisite:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) first.

```bash
docker build -t smart-retail-demand .
docker run -p 8000:8000 smart-retail-demand
```

**Without Docker** — run the API directly:

```bash
uvicorn src.api.forecasting_api:app --host 0.0.0.0 --port 8000
```

---

## Author

<h3> ADITYA SATAPATHY </h3>
<a href="https://github.com/gitadi2" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" alt="gitadi2" height="30" width="40" /></a>
<a href="https://linkedin.com/in/adisatapathy" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="adisatapathy" height="30" width="40" /></a>
<a href="mailto:satgriezeleo1007@gmail.com" target="blank"><img align="center" src="https://cdn.simpleicons.org/gmail/EA4335" alt="gmail" height="30" width="40" /></a>
