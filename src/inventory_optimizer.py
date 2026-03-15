"""
inventory_optimizer.py - DSA-Optimized Inventory Policy Simulator
==================================================================
Uses dynamic programming, sliding windows, binary search, and
priority queues to simulate and optimize inventory policies.

Usage: python src/inventory_optimizer.py
"""

import pandas as pd
import numpy as np
import os
import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.algorithms import (
    dynamic_programming_allocation,
    sliding_window_demand_rate,
    binary_search_reorder_point,
    benchmark_algorithms,
    prefix_sum_demand,
    range_demand_query,
)
from src.utils.data_structures import (
    SortedDemandArray,
    DemandBucketMap,
    TopKStockoutHeap,
    InventoryQueue,
)


class InventoryOptimizer:
    """Runs DSA-optimized inventory simulations and policy analysis."""

    def __init__(self):
        self.results = {}

    def load_data(self, path="data/processed/dashboard_data.csv"):
        """Load featured data for simulation."""
        if not os.path.exists(path):
            print("ERROR: dashboard_data.csv not found. Run feature_engineering.py first.")
            exit(1)
        self.df = pd.read_csv(path)
        print(f"Loaded {len(self.df):,} rows for inventory optimization")
        return self.df

    def run_dp_allocation(self, n_stores=10, total_inventory=5000):
        """Run dynamic programming inventory allocation across stores."""
        print("\n--- DP Inventory Allocation ---")

        # Get predicted demands per store (aggregate)
        store_demands = self.df.groupby("store_id")["units_sold"].mean().head(n_stores)
        demands = store_demands.values.tolist()
        capacities = [int(d * 2) for d in demands]  # Capacity = 2x avg demand

        result = dynamic_programming_allocation(demands, capacities, total_inventory)

        print(f"  Stores: {n_stores}, Total inventory: {total_inventory}")
        print(f"  Fill rate: {result['fill_rate']}%")
        print(f"  Inventory used: {result['inventory_used']}/{total_inventory}")

        self.results["dp_allocation"] = result
        return result

    def run_sliding_window_analysis(self, window_size=7):
        """Run sliding window demand anomaly detection."""
        print("\n--- Sliding Window Demand Analysis ---")

        # Aggregate daily demand
        daily = self.df.groupby("sale_date")["units_sold"].sum().sort_index()
        demands = daily.values.tolist()

        rates = sliding_window_demand_rate(demands, window_size)
        avg_rate = np.mean(rates)
        anomalies = sum(1 for r in rates if abs(r - avg_rate) > 2 * np.std(rates))

        print(f"  Window size: {window_size} days")
        print(f"  Avg rolling demand: {avg_rate:.2f}")
        print(f"  Demand anomalies detected: {anomalies}")

        self.results["sliding_window"] = {
            "window_size": window_size,
            "avg_rolling_demand": round(avg_rate, 2),
            "anomalies_detected": anomalies,
            "total_windows": len(rates),
        }
        return rates

    def run_reorder_optimization(self, service_level=0.95, lead_time=7):
        """Run binary search reorder point optimization."""
        print("\n--- Reorder Point Optimization ---")

        demands = sorted(self.df["units_sold"].values.tolist())
        result = binary_search_reorder_point(demands, service_level, lead_time)

        print(f"  Service level: {service_level*100}%")
        print(f"  Reorder point: {result['reorder_point']} units")
        print(f"  Safety stock: {result['safety_stock']} units")

        self.results["reorder_point"] = result
        return result

    def run_stockout_risk_analysis(self, k=20):
        """Identify top-K products at highest stockout risk using min-heap."""
        print(f"\n--- Top-{k} Stockout Risk (Min-Heap) ---")

        heap = TopKStockoutHeap(k=k)

        # Stockout risk = high demand variance + low recent supply
        product_stats = self.df.groupby("product_id").agg(
            avg_demand=("units_sold", "mean"),
            std_demand=("units_sold", "std"),
        ).fillna(0)

        for pid, row in product_stats.iterrows():
            risk = row["std_demand"] / max(row["avg_demand"], 1)  # CV as risk proxy
            heap.push(str(pid), risk)

        top_k = heap.get_top_k()
        print(f"  Highest risk product: {top_k[0][0]} (score={top_k[0][1]:.4f})")
        print(f"  Lowest in top-{k}: {top_k[-1][0]} (score={top_k[-1][1]:.4f})")

        self.results["top_k_stockout"] = [
            {"product_id": pid, "risk_score": round(score, 4)}
            for pid, score in top_k
        ]
        return top_k

    def run_category_demand_analysis(self):
        """Analyze demand by segment using hash map."""
        print("\n--- Segment Demand Analysis (Hash Map) ---")

        # Use volatility_band as demand segment (category lost during aggregation)
        segment_col = "volatility_band" if "volatility_band" in self.df.columns else "store_id"
        bucket_map = DemandBucketMap()
        bucket_map.build(
            self.df[segment_col].astype(str).tolist(),
            self.df["units_sold"].tolist(),
            self.df["revenue"].tolist() if "revenue" in self.df.columns else None,
        )

        stats = bucket_map.get_all_stats()
        for s in stats[:5]:
            print(f"  {s['segment']}: avg_demand={s['avg_demand']}, total_revenue=${s['total_revenue']:,.0f}")

        self.results["category_analysis"] = stats
        return stats

    def run_benchmarks(self):
        """Benchmark DSA algorithms vs naive approaches."""
        print("\n--- Algorithm Benchmarks ---")

        demands = self.df["units_sold"].values.tolist()[:50000]
        bench = benchmark_algorithms(demands)

        print(f"  Binary search: {bench['binary_search_time_ms']:.4f} ms")
        print(f"  Linear search: {bench['linear_search_time_ms']:.4f} ms")
        print(f"  Search speedup: {bench['search_speedup']}x")
        print(f"  Prefix sum query: {bench['prefix_sum_time_ms']:.4f} ms")
        print(f"  Naive sum query: {bench['naive_sum_time_ms']:.4f} ms")
        print(f"  Sum speedup: {bench['sum_speedup']}x")

        self.results["benchmarks"] = bench
        return bench

    def plot_results(self):
        """Generate inventory optimization charts."""
        os.makedirs("reports/figures", exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. DP Allocation
        if "dp_allocation" in self.results:
            alloc = self.results["dp_allocation"]["allocation"]
            axes[0, 0].bar(range(len(alloc)), alloc, color="#3498db")
            axes[0, 0].set_title("DP Inventory Allocation by Store")
            axes[0, 0].set_xlabel("Store Index")
            axes[0, 0].set_ylabel("Units Allocated")

        # 2. Algorithm benchmarks
        if "benchmarks" in self.results:
            bench = self.results["benchmarks"]
            labels = ["Binary\nSearch", "Linear\nSearch", "Prefix\nSum", "Naive\nSum"]
            times = [bench["binary_search_time_ms"], bench["linear_search_time_ms"],
                     bench["prefix_sum_time_ms"], bench["naive_sum_time_ms"]]
            colors = ["#2ecc71", "#e74c3c", "#2ecc71", "#e74c3c"]
            axes[0, 1].barh(labels, times, color=colors)
            axes[0, 1].set_xlabel("Time (ms)")
            axes[0, 1].set_title("Algorithm Benchmarks")

        # 3. Category demand
        if "category_analysis" in self.results:
            stats = self.results["category_analysis"][:10]
            cats = [s["segment"] for s in stats]
            demands = [s["avg_demand"] for s in stats]
            axes[1, 0].barh(cats, demands, color="#9b59b6")
            axes[1, 0].set_xlabel("Avg Daily Demand")
            axes[1, 0].set_title("Demand by Category")

        # 4. Top-K stockout risk
        if "top_k_stockout" in self.results:
            top_k = self.results["top_k_stockout"][:10]
            pids = [t["product_id"][:8] for t in top_k]
            risks = [t["risk_score"] for t in top_k]
            axes[1, 1].barh(pids, risks, color="#e67e22")
            axes[1, 1].set_xlabel("Stockout Risk Score")
            axes[1, 1].set_title("Top-10 Stockout Risk Products")

        plt.tight_layout()
        plt.savefig("reports/figures/inventory_optimization.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\nSaved: reports/figures/inventory_optimization.png")

    def save_report(self):
        """Save full optimization report as JSON."""
        os.makedirs("reports", exist_ok=True)
        with open("reports/inventory_report.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print("Saved: reports/inventory_report.json")


def run_full_pipeline():
    """Execute the full inventory optimization pipeline."""
    print("=" * 60 + "\nINVENTORY OPTIMIZATION PIPELINE\n" + "=" * 60)

    optimizer = InventoryOptimizer()
    optimizer.load_data()

    optimizer.run_dp_allocation()
    optimizer.run_sliding_window_analysis()
    optimizer.run_reorder_optimization()
    optimizer.run_stockout_risk_analysis()
    optimizer.run_category_demand_analysis()
    optimizer.run_benchmarks()

    optimizer.plot_results()
    optimizer.save_report()

    return optimizer


if __name__ == "__main__":
    run_full_pipeline()
