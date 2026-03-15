"""
performance_benchmark.py - Benchmark DSA vs naive approaches
==============================================================
Run: python benchmarks/performance_benchmark.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.algorithms import benchmark_algorithms
from src.utils.data_structures import LRUCache, SortedDemandArray, TopKStockoutHeap


def benchmark_cache():
    """Benchmark LRU cache vs recomputation."""
    print("\n--- LRU Cache Benchmark ---")
    cache = LRUCache(capacity=10000)
    n = 50000

    # Populate cache
    for i in range(min(n, 10000)):
        cache.put(f"key_{i}", i * 1.5)

    # Benchmark cache hits
    start = time.time()
    for i in range(n):
        cache.get(f"key_{i % 10000}")
    cache_time = (time.time() - start) * 1000

    # Benchmark recomputation (simulate model inference)
    start = time.time()
    for i in range(n):
        _ = np.random.random() * 100  # Simulate prediction
    recompute_time = (time.time() - start) * 1000

    print(f"  Cache lookup ({n} ops):   {cache_time:.2f} ms")
    print(f"  Recomputation ({n} ops):  {recompute_time:.2f} ms")
    print(f"  Speedup:                  {recompute_time/cache_time:.1f}x")


def benchmark_sorted_array():
    """Benchmark binary search vs linear scan."""
    print("\n--- Sorted Array Benchmark ---")
    data = list(np.random.random(1000000))
    arr = SortedDemandArray(data)

    # Binary search
    start = time.time()
    for _ in range(10000):
        arr.count_below(0.5)
    bs_time = (time.time() - start) * 1000

    # Linear scan
    start = time.time()
    for _ in range(10000):
        sum(1 for x in data if x <= 0.5)
    ls_time = (time.time() - start) * 1000

    print(f"  Binary search (10K queries on 1M): {bs_time:.2f} ms")
    print(f"  Linear scan (10K queries on 1M):   {ls_time:.2f} ms")
    print(f"  Speedup:                           {ls_time/bs_time:.1f}x")


def benchmark_heap():
    """Benchmark min-heap vs full sort for top-K."""
    print("\n--- Top-K Heap Benchmark ---")
    n = 100000
    k = 100
    data = [(f"P{i}", np.random.random()) for i in range(n)]

    # Heap approach
    start = time.time()
    heap = TopKStockoutHeap(k=k)
    for pid, score in data:
        heap.push(pid, score)
    top_k_heap = heap.get_top_k()
    heap_time = (time.time() - start) * 1000

    # Sort approach
    start = time.time()
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)[:k]
    sort_time = (time.time() - start) * 1000

    print(f"  Heap top-{k} from {n}: {heap_time:.2f} ms")
    print(f"  Full sort top-{k}:     {sort_time:.2f} ms")
    print(f"  Speedup:               {sort_time/heap_time:.1f}x")


def main():
    print("=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)

    np.random.seed(42)
    benchmark_cache()
    benchmark_sorted_array()
    benchmark_heap()

    print("\n--- Algorithm Benchmarks ---")
    demands = list(np.random.poisson(20, 50000).astype(float))
    results = benchmark_algorithms(demands)
    for key, val in results.items():
        print(f"  {key}: {val}")

    print(f"\n{'='*60}")
    print("All benchmarks complete!")


if __name__ == "__main__":
    main()
