"""
test_data_structures.py - Tests for custom data structures
============================================================
Run: pytest tests/test_data_structures.py -v
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_structures import (
    LRUCache,
    SortedDemandArray,
    DemandBucketMap,
    TopKStockoutHeap,
    InventoryQueue,
)


class TestLRUCache:
    """Test LRU Cache implementation."""

    def test_basic_get_put(self):
        cache = LRUCache(capacity=3)
        cache.put("A", 100)
        assert cache.get("A") == 100

    def test_cache_miss(self):
        cache = LRUCache(capacity=3)
        assert cache.get("missing") is None

    def test_eviction(self):
        cache = LRUCache(capacity=2)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.put("C", 3)  # Evicts A
        assert cache.get("A") is None
        assert cache.get("B") == 2
        assert cache.get("C") == 3

    def test_lru_order(self):
        cache = LRUCache(capacity=2)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.get("A")  # A is now most recent
        cache.put("C", 3)  # Evicts B (least recently used)
        assert cache.get("B") is None
        assert cache.get("A") == 1

    def test_overwrite(self):
        cache = LRUCache(capacity=2)
        cache.put("A", 1)
        cache.put("A", 99)
        assert cache.get("A") == 99
        assert cache.size == 1

    def test_hit_rate(self):
        cache = LRUCache(capacity=5)
        cache.put("A", 1)
        cache.get("A")  # Hit
        cache.get("A")  # Hit
        cache.get("B")  # Miss
        assert cache.hit_rate == 66.7 or abs(cache.hit_rate - 66.67) < 0.1

    def test_clear(self):
        cache = LRUCache(capacity=5)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.get("A") is None

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=0)

    def test_capacity_one(self):
        cache = LRUCache(capacity=1)
        cache.put("A", 1)
        cache.put("B", 2)
        assert cache.get("A") is None
        assert cache.get("B") == 2

    def test_o1_performance(self):
        cache = LRUCache(capacity=10000)
        for i in range(10000):
            cache.put(f"key_{i}", i)
        start = time.time()
        for i in range(10000):
            cache.get(f"key_{i}")
        elapsed = time.time() - start
        assert elapsed < 1.0  # 10K ops should be well under 1s


class TestSortedDemandArray:
    """Test sorted demand array with binary search."""

    def test_count_below(self):
        arr = SortedDemandArray([10, 20, 30, 40, 50])
        assert arr.count_below(25) == 2

    def test_count_above(self):
        arr = SortedDemandArray([10, 20, 30, 40, 50])
        assert arr.count_above(30) == 2

    def test_fraction_below(self):
        arr = SortedDemandArray([10, 20, 30, 40, 50])
        assert abs(arr.fraction_below(30) - 0.6) < 0.01

    def test_find_threshold_for_rate(self):
        arr = SortedDemandArray(list(range(100)))
        threshold = arr.find_threshold_for_rate(0.5)
        assert 45 <= threshold <= 55

    def test_percentile(self):
        arr = SortedDemandArray(list(range(101)))
        assert arr.percentile(50) == 50
        assert arr.percentile(0) == 0
        assert arr.percentile(100) == 100

    def test_empty_array(self):
        arr = SortedDemandArray([])
        assert len(arr) == 0

    def test_ologn_performance(self):
        arr = SortedDemandArray(list(range(1000000)))
        start = time.time()
        for _ in range(10000):
            arr.count_below(500000)
        elapsed = time.time() - start
        assert elapsed < 1.0  # Binary search on 1M elements


class TestDemandBucketMap:
    """Test hash map for demand segments."""

    def test_build_and_lookup(self):
        bucket_map = DemandBucketMap()
        bucket_map.build(
            ["Groceries", "Dairy", "Groceries"],
            [100.0, 50.0, 80.0],
            [500.0, 200.0, 400.0],
        )
        stats = bucket_map.get_stats("Groceries")
        assert stats["count"] == 2
        assert stats["total_demand"] == 180.0

    def test_avg_demand(self):
        bucket_map = DemandBucketMap()
        bucket_map.build(["A", "A", "A"], [10.0, 20.0, 30.0])
        assert bucket_map.get_stats("A")["avg_demand"] == 20.0

    def test_missing_segment(self):
        bucket_map = DemandBucketMap()
        bucket_map.build(["A"], [10.0])
        result = bucket_map.get_stats("MISSING")
        assert "error" in result

    def test_segments_list(self):
        bucket_map = DemandBucketMap()
        bucket_map.build(["C", "A", "B"], [1.0, 2.0, 3.0])
        assert bucket_map.segments == ["A", "B", "C"]


class TestTopKStockoutHeap:
    """Test min-heap for top-K stockout risk."""

    def test_basic_top_k(self):
        heap = TopKStockoutHeap(k=3)
        for i, score in enumerate([0.1, 0.9, 0.5, 0.3, 0.8]):
            heap.push(f"P{i}", score)
        top = heap.get_top_k()
        assert len(top) == 3
        assert top[0][1] == 0.9  # Highest risk first

    def test_size_limit(self):
        heap = TopKStockoutHeap(k=2)
        for i in range(100):
            heap.push(f"P{i}", float(i))
        assert heap.size == 2

    def test_min_score(self):
        heap = TopKStockoutHeap(k=3)
        heap.push("A", 5.0)
        heap.push("B", 3.0)
        heap.push("C", 7.0)
        assert heap.min_score == 3.0

    def test_empty_heap(self):
        heap = TopKStockoutHeap(k=5)
        assert heap.size == 0
        assert heap.get_top_k() == []


class TestInventoryQueue:
    """Test priority queue for reorder scheduling."""

    def test_enqueue_dequeue(self):
        q = InventoryQueue()
        q.enqueue("P001", urgency=0.9, quantity=100)
        q.enqueue("P002", urgency=0.5, quantity=50)
        result = q.dequeue()
        assert result["product_id"] == "P001"
        assert result["urgency"] == 0.9

    def test_priority_order(self):
        q = InventoryQueue()
        q.enqueue("Low", urgency=0.2, quantity=10)
        q.enqueue("High", urgency=0.95, quantity=200)
        q.enqueue("Med", urgency=0.5, quantity=50)
        assert q.dequeue()["product_id"] == "High"
        assert q.dequeue()["product_id"] == "Med"
        assert q.dequeue()["product_id"] == "Low"

    def test_peek(self):
        q = InventoryQueue()
        q.enqueue("P001", urgency=0.8, quantity=100)
        peeked = q.peek()
        assert peeked["product_id"] == "P001"
        assert q.size == 1  # Peek doesn't remove

    def test_empty_dequeue(self):
        q = InventoryQueue()
        assert q.dequeue() is None
        assert q.is_empty

    def test_size(self):
        q = InventoryQueue()
        q.enqueue("A", 0.5, 10)
        q.enqueue("B", 0.6, 20)
        assert q.size == 2
        q.dequeue()
        assert q.size == 1
