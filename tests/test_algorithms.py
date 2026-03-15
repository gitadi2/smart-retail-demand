"""
test_algorithms.py - Tests for DSA algorithm implementations
==============================================================
Run: pytest tests/test_algorithms.py -v
"""

import pytest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.algorithms import (
    dynamic_programming_allocation,
    sliding_window_demand_rate,
    binary_search_reorder_point,
    linear_search_reorder_point,
    two_pointer_paired_products,
    prefix_sum_demand,
    range_demand_query,
    benchmark_algorithms,
)


class TestDPAllocation:
    """Test dynamic programming inventory allocation."""

    def test_basic_allocation(self):
        result = dynamic_programming_allocation(
            demands=[100, 50, 30],
            store_capacities=[150, 80, 50],
            total_inventory=200
        )
        assert result["fill_rate"] > 0
        assert result["inventory_used"] <= 200
        assert sum(result["allocation"]) <= 200

    def test_scarce_inventory(self):
        result = dynamic_programming_allocation(
            demands=[100, 100, 100],
            store_capacities=[100, 100, 100],
            total_inventory=50
        )
        assert result["inventory_used"] == 50
        assert result["fill_rate"] < 100

    def test_abundant_inventory(self):
        result = dynamic_programming_allocation(
            demands=[10, 20, 30],
            store_capacities=[50, 50, 50],
            total_inventory=500
        )
        assert result["fill_rate"] == 100.0

    def test_single_store(self):
        result = dynamic_programming_allocation(
            demands=[50],
            store_capacities=[100],
            total_inventory=75
        )
        assert result["allocation"][0] >= 50


class TestSlidingWindow:
    """Test sliding window demand rate."""

    def test_basic_window(self):
        demands = [10, 20, 30, 40, 50, 60, 70]
        rates = sliding_window_demand_rate(demands, window_size=3)
        assert len(rates) == 5  # n - window + 1
        assert rates[0] == 20.0  # avg of [10, 20, 30]

    def test_constant_demand(self):
        demands = [10, 10, 10, 10, 10]
        rates = sliding_window_demand_rate(demands, window_size=3)
        assert all(r == 10.0 for r in rates)

    def test_single_element_window(self):
        demands = [5, 10, 15]
        rates = sliding_window_demand_rate(demands, window_size=1)
        assert rates == [5.0, 10.0, 15.0]

    def test_window_equals_data(self):
        demands = [10, 20, 30]
        rates = sliding_window_demand_rate(demands, window_size=3)
        assert len(rates) == 1
        assert rates[0] == 20.0

    def test_empty_input(self):
        assert sliding_window_demand_rate([], window_size=5) == []


class TestBinarySearchReorder:
    """Test binary search reorder point optimization."""

    def test_basic_reorder(self):
        demands = sorted([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        result = binary_search_reorder_point(demands, service_level=0.95, lead_time_days=7)
        assert result["reorder_point"] > 0
        assert result["safety_stock"] >= 0
        assert result["lead_time_days"] == 7

    def test_matches_linear(self):
        np.random.seed(42)
        demands = list(np.random.poisson(20, 1000).astype(float))
        sorted_demands = sorted(demands)
        result_bs = binary_search_reorder_point(sorted_demands)
        result_ls = linear_search_reorder_point(demands)
        assert result_bs["reorder_point"] == result_ls["reorder_point"]

    def test_high_service_level(self):
        demands = sorted([10.0] * 100)
        result = binary_search_reorder_point(demands, service_level=0.99)
        assert result["safety_stock"] == 0  # No variance = no safety stock needed

    def test_empty_demands(self):
        result = binary_search_reorder_point([], service_level=0.95)
        assert result["reorder_point"] == 0

    def test_binary_faster_than_linear(self):
        np.random.seed(42)
        demands = list(np.random.poisson(20, 10000).astype(float))
        bench = benchmark_algorithms(demands, n_iterations=50)
        assert bench["binary_search_time_ms"] <= bench["linear_search_time_ms"]


class TestTwoPointerPaired:
    """Test two-pointer product pairing."""

    def test_basic_pairing(self):
        products = [("A", 10.0), ("B", 20.0), ("C", 30.0), ("D", 40.0)]
        pairs = two_pointer_paired_products(products, target_combined_demand=50.0)
        assert len(pairs) > 0
        for p1, p2, combined in pairs:
            assert abs(combined - 50.0) <= 2.5  # 5% tolerance

    def test_no_valid_pairs(self):
        products = [("A", 1.0), ("B", 2.0)]
        pairs = two_pointer_paired_products(products, target_combined_demand=100.0)
        assert len(pairs) == 0


class TestPrefixSum:
    """Test prefix sum for O(1) range queries."""

    def test_basic_prefix_sum(self):
        demands = [10, 20, 30, 40, 50]
        prefix = prefix_sum_demand(demands)
        assert len(prefix) == 6
        assert prefix[0] == 0
        assert prefix[5] == 150

    def test_range_query(self):
        demands = [10, 20, 30, 40, 50]
        prefix = prefix_sum_demand(demands)
        assert range_demand_query(prefix, 1, 3) == 90  # 20 + 30 + 40

    def test_single_element_query(self):
        demands = [10, 20, 30]
        prefix = prefix_sum_demand(demands)
        assert range_demand_query(prefix, 1, 1) == 20

    def test_full_range_query(self):
        demands = [10, 20, 30]
        prefix = prefix_sum_demand(demands)
        assert range_demand_query(prefix, 0, 2) == 60
