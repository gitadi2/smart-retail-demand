"""
One-command pipeline:
  generate -> validate -> LEAKAGE AUDIT (hard gate) -> walk-forward CV vs baseline
  -> final temporal holdout -> drift report -> persist artifacts.

Run: python run_pipeline.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.leakage import run_audit
from src.data.generate import generate_demand
from src.data.validation import validate
from src.eval.metrics import all_metrics, wmape
from src.eval.validation import temporal_split, walk_forward
from src.features.engineering import TARGET, build_features, feature_columns
from src.models.baselines import seasonal_naive
from src.models.train import DemandModel, available_models
from src.monitoring.drift import drift_report

ART = Path("artifacts")
FIG = Path("reports/figures")
ART.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    t0 = time.time()
    print("[1/6] generate + validate")
    raw = generate_demand()
    validate(raw)
    print(f"      rows={len(raw):,}  series={raw.groupby(['store_id','product_id']).ngroups}")

    print("[2/6] LEAKAGE AUDIT (hard gate)")
    audit = run_audit(raw, build_features)
    print("     ", audit["target_perturbation"]["detail"])
    if not audit["passed"]:
        raise SystemExit("Pipeline halted: leakage audit failed.")

    print("[3/6] feature engineering")
    feat = build_features(raw)
    cols = feature_columns(feat)
    print(f"      {len(cols)} features")

    print("[4/6] walk-forward CV (5 folds x 28d) vs seasonal-naive")
    models = available_models()
    fold_scores = {m: [] for m in models}
    fold_scores["seasonal_naive"] = []
    for k, tr, te in walk_forward(feat, n_folds=5, horizon=28):
        Xtr, ytr = tr[cols], tr[TARGET]
        Xte, yte = te[cols], te[TARGET]
        fold_scores["seasonal_naive"].append(wmape(yte, seasonal_naive(te)))
        for m in models:
            model = DemandModel(m).fit(Xtr, ytr)
            fold_scores[m].append(wmape(yte, model.predict(Xte)))
    cv = {
        k: {"wmape_mean": round(float(np.mean(v)), 3),
            "wmape_std": round(float(np.std(v)), 3)}
        for k, v in fold_scores.items()
    }
    for k, v in sorted(cv.items(), key=lambda x: x[1]["wmape_mean"]):
        print(f"      {k:16s} WMAPE {v['wmape_mean']:6.2f}% +/- {v['wmape_std']:.2f}")

    print("[5/6] final temporal holdout (last 56 days)")
    train, test, cutoff = temporal_split(feat, test_days=56)
    best_kind = min(models, key=lambda m: cv[m]["wmape_mean"])
    best = DemandModel(best_kind).fit(train[cols], train[TARGET])
    pred = best.predict(test[cols])
    holdout = all_metrics(test[TARGET], pred)
    base_wmape = wmape(test[TARGET], seasonal_naive(test))
    print(f"      best={best_kind}  cutoff={cutoff.date()}")
    print(f"      holdout: {holdout}")
    print(f"      seasonal-naive WMAPE on same holdout: {base_wmape:.2f}%")

    print("[6/6] drift report (train vs holdout)")
    drift = drift_report(train, test, cols)
    print(drift.head(6).to_string(index=False))

    # ---- persist ----
    results = {
        "rows": int(len(raw)),
        "n_features": len(cols),
        "models_available": models,
        "best_model": best_kind,
        "walk_forward_cv": cv,
        "holdout_metrics": holdout,
        "baseline_holdout_wmape": round(base_wmape, 3),
        "improvement_vs_naive_pct": round(
            (base_wmape - holdout["wmape"]) / base_wmape * 100, 2
        ),
        "audit": audit,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (ART / "metrics.json").write_text(json.dumps(results, indent=2))
    drift.to_csv(ART / "drift_report.csv", index=False)
    import joblib
    joblib.dump(best, ART / "model.joblib")
    (ART / "feature_order.json").write_text(json.dumps(cols))
    _plot(cv, holdout, base_wmape)
    print(f"\nDone in {results['runtime_sec']}s. Artifacts -> {ART}/  Figures -> {FIG}/")
    return results


def _plot(cv, holdout, base_wmape):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(cv.items(), key=lambda x: x[1]["wmape_mean"])
    names = [k for k, _ in items]
    means = [v["wmape_mean"] for _, v in items]
    errs = [v["wmape_std"] for _, v in items]
    colors = ["#2e7d32" if n != "seasonal_naive" else "#9e9e9e" for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, means, xerr=errs, color=colors, alpha=0.85, capsize=4)
    ax.set_xlabel("WMAPE % (lower is better)  -- walk-forward CV, 5 folds")
    ax.set_title("Demand forecasting: model vs seasonal-naive baseline")
    ax.invert_yaxis()
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(m + e + 0.3, i, f"{m:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "cv_wmape.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
