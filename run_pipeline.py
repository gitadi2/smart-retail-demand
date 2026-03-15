"""
run_pipeline.py - Master Pipeline Runner
==========================================
Runs all stages of the retail demand forecasting pipeline sequentially.

Usage: python run_pipeline.py
"""

import subprocess
import sys
import time
import os


def run_stage(stage_num, name, command):
    """Run a pipeline stage and track timing."""
    print(f"\n{'='*60}")
    print(f"STAGE {stage_num}: {name}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [sys.executable] + command.split(),
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - start

    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"\n[{status}] Stage {stage_num} completed in {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"ERROR: Stage {stage_num} failed. Stopping pipeline.")
        sys.exit(1)

    return elapsed


def main():
    print("=" * 60)
    print("SMART RETAIL DEMAND & INVENTORY OPTIMIZATION SYSTEM")
    print("Full Pipeline Runner v1.0.0")
    print("=" * 60)

    total_start = time.time()
    timings = {}

    # Stage 0: Generate sample data
    timings["data_gen"] = run_stage(
        0, "DATA GENERATION",
        "scripts/generate_sample_data.py"
    )

    # Stage 1: Data cleaning
    timings["cleaning"] = run_stage(
        1, "DATA CLEANING",
        "src/data_cleaning.py"
    )

    # Stage 2: Feature engineering
    timings["features"] = run_stage(
        2, "FEATURE ENGINEERING",
        "src/feature_engineering.py"
    )

    # Stage 3: Model training (Keras + XGBoost + LightGBM)
    timings["training"] = run_stage(
        3, "MODEL TRAINING (Keras LSTM/GRU/CNN-LSTM/Attention + XGBoost + LightGBM)",
        "src/model_training.py"
    )

    # Stage 4: Inventory optimization (DSA algorithms)
    timings["optimization"] = run_stage(
        4, "INVENTORY OPTIMIZATION (DP + Binary Search + Sliding Window)",
        "src/inventory_optimizer.py"
    )

    # Stage 5: Run tests
    print(f"\n{'='*60}")
    print("STAGE 5: TESTS")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    timings["tests"] = time.time() - start
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"\n[{status}] Tests completed in {timings['tests']:.1f}s")

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Data Generation:       {timings['data_gen']:.1f}s")
    print(f"  Data Cleaning:         {timings['cleaning']:.1f}s")
    print(f"  Feature Engineering:   {timings['features']:.1f}s")
    print(f"  Model Training:        {timings['training']:.1f}s")
    print(f"  Inventory Optimization:{timings['optimization']:.1f}s")
    print(f"  Tests:                 {timings['tests']:.1f}s")
    print(f"  {'─'*40}")
    print(f"  TOTAL:                 {total_time:.1f}s")
    print(f"\nAll stages completed successfully!")
    print(f"Run API: uvicorn src.api.forecasting_api:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
