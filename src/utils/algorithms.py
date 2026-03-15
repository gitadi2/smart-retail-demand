"""
algorithms.py - DSA-optimized algorithms for retail demand
============================================================
Custom algorithm implementations demonstrating algorithmic
thinking applied to real business problems.

Includes:
    - dynamic_programming_allocation: DP for multi-store inventory allocation
    - sliding_window_demand_rate: O(n) rolling demand monitoring
    - binary_search_reorder_point: O(log n) optimal reorder threshold
    - two_pointer_paired_products: Bundle detection via two pointers
    - benchmark_algorithms: Performance comparison
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional


def dynamic_programming_allocation(
    demands: List[float],
    store_capacities: List[int],
    total_inventory: int
) -> Dict:
    """
    Allocate inventory across stores to maximize fulfilled demand
    using dynamic programming (bounded knapsack variant).

    Time Complexity:  O(n * W) where n=stores, W=total_inventory
    Space Complexity: O(n * W)

    Parameters
    ----------
    demands : List[float]
        Predicted demand for each store
    store_capacities : List[int]
        Max inventory each store can hold
    total_inventory : int
        Total available inventory to distribute

    Returns
    -------
    Dict with allocation per store, total fulfilled demand, fill rate
    """
    n = len(demands)
    W = total_inventory

    # Clamp to reasonable size for DP table
    W = min(W, 10000)
    capacities = [min(int(c), W) for c in store_capacities]

    # DP table: dp[i][w] = max demand fulfilled using stores 0..i-1 with w units
    dp = [[0.0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        max_alloc = capacities[i - 1]
        demand = demands[i - 1]
        for w in range(W + 1):
            # Option 1: don't allocate to store i
            dp[i][w] = dp[i - 1][w]
            # Option 2: allocate k units to store i
            for k in range(1, min(max_alloc, w) + 1):
                fulfilled = min(k, demand)
                dp[i][w] = max(dp[i][w], dp[i - 1][w - k] + fulfilled)

    # Backtrack to find allocation
    allocation = [0] * n
    w = W
    for i in range(n, 0, -1):
        max_alloc = capacities[i - 1]
        demand = demands[i - 1]
        for k in range(min(max_alloc, w), 0, -1):
            fulfilled = min(k, demand)
            if abs(dp[i][w] - (dp[i - 1][w - k] + fulfilled)) < 1e-9:
                allocation[i - 1] = k
                w -= k
                break

    total_fulfilled = sum(min(allocation[i], demands[i]) for i in range(n))
    total_demand = sum(demands)

    return {
        "allocation": allocation,
        "total_fulfilled": round(total_fulfilled, 2),
        "total_demand": round(total_demand, 2),
        "fill_rate": round(total_fulfilled / total_demand * 100, 2) if total_demand > 0 else 0,
        "inventory_used": sum(allocation),
        "inventory_remaining": total_inventory - sum(allocation),
    }


def sliding_window_demand_rate(
    daily_demands: List[float],
    window_size: int = 7
) -> List[float]:
    """
    Compute rolling average demand using sliding window.
    Single pass, O(n) time, O(1) extra space.

    Use Case: Detect sudden demand spikes or drops for anomaly alerts.

    Time Complexity:  O(n)
    Space Complexity: O(n) for output, O(1) working space
    """
    n = len(daily_demands)
    if n == 0:
        return []
    window_size = min(window_size, n)
    rates = []

    window_sum = sum(daily_demands[:window_size])
    rates.append(window_sum / window_size)

    for i in range(window_size, n):
        window_sum += daily_demands[i] - daily_demands[i - window_size]
        rates.append(window_sum / window_size)

    return [round(r, 4) for r in rates]


def binary_search_reorder_point(
    sorted_demands: List[float],
    service_level: float = 0.95,
    lead_time_days: int = 7
) -> Dict:
    """
    Find optimal reorder point using binary search on sorted demand history.

    The reorder point is the demand level at the target service level percentile,
    multiplied by lead time — ensures we order before stockout.

    Time Complexity:  O(log n)
    Space Complexity: O(1)

    Parameters
    ----------
    sorted_demands : List[float]
        Historical daily demands, sorted ascending
    service_level : float
        Target service level (0.0 to 1.0), typically 0.95
    lead_time_days : int
        Number of days for restocking

    Returns
    -------
    Dict with reorder_point, safety_stock, daily_demand_at_service_level
    """
    n = len(sorted_demands)
    if n == 0:
        return {"reorder_point": 0, "safety_stock": 0, "daily_demand_at_sl": 0}

    # Binary search for the demand at service_level percentile
    target_idx = int(service_level * (n - 1))
    target_idx = max(0, min(target_idx, n - 1))

    demand_at_sl = sorted_demands[target_idx]
    avg_demand = sum(sorted_demands) / n

    safety_stock = (demand_at_sl - avg_demand) * lead_time_days
    safety_stock = max(0, round(safety_stock))
    reorder_point = round(avg_demand * lead_time_days + safety_stock)

    return {
        "reorder_point": reorder_point,
        "safety_stock": safety_stock,
        "daily_demand_at_sl": round(demand_at_sl, 2),
        "avg_daily_demand": round(avg_demand, 2),
        "lead_time_days": lead_time_days,
        "service_level": service_level,
    }


def linear_search_reorder_point(
    demands: List[float],
    service_level: float = 0.95,
    lead_time_days: int = 7
) -> Dict:
    """
    Naive O(n log n) reorder point calculation (sort + index).
    Used as a benchmark baseline against binary search.
    """
    if not demands:
        return {"reorder_point": 0, "safety_stock": 0, "daily_demand_at_sl": 0}

    sorted_d = sorted(demands)  # O(n log n)
    target_idx = int(service_level * (len(sorted_d) - 1))
    demand_at_sl = sorted_d[target_idx]
    avg_demand = sum(sorted_d) / len(sorted_d)
    safety_stock = max(0, round((demand_at_sl - avg_demand) * lead_time_days))
    reorder_point = round(avg_demand * lead_time_days + safety_stock)

    return {
        "reorder_point": reorder_point,
        "safety_stock": safety_stock,
        "daily_demand_at_sl": round(demand_at_sl, 2),
    }


def two_pointer_paired_products(
    product_demands: List[Tuple[str, float]],
    target_combined_demand: float
) -> List[Tuple[str, str, float]]:
    """
    Find pairs of products whose combined demand equals a target.
    Useful for bundle promotions.

    Time Complexity:  O(n log n) for sort + O(n) for two-pointer scan
    Space Complexity: O(n) for output
    """
    sorted_products = sorted(product_demands, key=lambda x: x[1])
    pairs = []
    left, right = 0, len(sorted_products) - 1
    tolerance = target_combined_demand * 0.05  # 5% tolerance

    while left < right:
        combined = sorted_products[left][1] + sorted_products[right][1]
        if abs(combined - target_combined_demand) <= tolerance:
            pairs.append((
                sorted_products[left][0],
                sorted_products[right][0],
                round(combined, 2)
            ))
            left += 1
            right -= 1
        elif combined < target_combined_demand:
            left += 1
        else:
            right -= 1

    return pairs


def prefix_sum_demand(daily_demands: List[float]) -> List[float]:
    """
    Build prefix sum array for O(1) range demand queries.

    Time Complexity:  O(n) build, O(1) per query
    Space Complexity: O(n)
    """
    n = len(daily_demands)
    prefix = [0.0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + daily_demands[i]
    return prefix


def range_demand_query(prefix: List[float], left: int, right: int) -> float:
    """Query total demand in range [left, right] in O(1)."""
    right = min(right, len(prefix) - 2)
    left = max(left, 0)
    return round(prefix[right + 1] - prefix[left], 2)


def benchmark_algorithms(demands: List[float], n_iterations: int = 100) -> Dict:
    """Benchmark binary search vs linear search for reorder optimization."""
    sorted_demands = sorted(demands)

    # Benchmark binary search (on pre-sorted data)
    start = time.time()
    for _ in range(n_iterations):
        binary_search_reorder_point(sorted_demands)
    bs_time = (time.time() - start) / n_iterations * 1000

    # Benchmark linear search (includes sorting each time)
    start = time.time()
    for _ in range(n_iterations):
        linear_search_reorder_point(demands)  # unsorted, must sort each time
    ls_time = (time.time() - start) / n_iterations * 1000

    # Benchmark prefix sum vs naive range sum
    prefix = prefix_sum_demand(demands)
    start = time.time()
    for _ in range(n_iterations):
        range_demand_query(prefix, 10, len(demands) - 10)
    ps_time = (time.time() - start) / n_iterations * 1000

    start = time.time()
    for _ in range(n_iterations):
        sum(demands[10:len(demands) - 10])
    naive_time = (time.time() - start) / n_iterations * 1000

    return {
        "binary_search_time_ms": round(bs_time, 4),
        "linear_search_time_ms": round(ls_time, 4),
        "search_speedup": round(ls_time / bs_time, 2) if bs_time > 0 else 0,
        "prefix_sum_time_ms": round(ps_time, 4),
        "naive_sum_time_ms": round(naive_time, 4),
        "sum_speedup": round(naive_time / ps_time, 2) if ps_time > 0 else 0,
    }
