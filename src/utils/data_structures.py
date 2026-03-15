"""
data_structures.py - Custom Data Structures for Retail Demand
==============================================================
Production-grade data structures for demand forecasting,
inventory management, and real-time analytics.

Includes:
    - LRUCache: O(1) prediction caching for repeated forecasts
    - SortedDemandArray: O(log n) demand threshold queries
    - DemandBucketMap: O(1) segment-level demand lookups
    - TopKStockoutHeap: Min-heap for top-K stockout risk items
    - InventoryQueue: Priority queue for reorder scheduling

Time/Space complexities documented for every operation.
"""

from collections import OrderedDict
from typing import Any, Optional, List, Tuple
import heapq


class LRUCache:
    """
    Least Recently Used Cache for forecast predictions.

    Use Case: Cache repeated store-product forecast requests
    to avoid re-running Keras inference.

    Time Complexity:  O(1) get, O(1) put
    Space Complexity: O(capacity)
    """

    def __init__(self, capacity: int = 5000):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached forecast. O(1)"""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Cache a forecast result. O(1)"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return f"LRUCache(capacity={self.capacity}, size={self.size}, hit_rate={self.hit_rate:.1f}%)"


class SortedDemandArray:
    """
    Sorted array of demand values enabling O(log n) threshold queries.

    Use Case: Quickly find how many products have demand below/above
    a threshold for inventory restocking decisions.

    Time Complexity:
        - build:           O(n log n) one-time sort
        - count_below:     O(log n) via bisect
        - count_above:     O(log n) via bisect
        - find_threshold:  O(log n) binary search
    Space Complexity: O(n)
    """

    def __init__(self, values: List[float]):
        self.values = sorted(values)
        self.n = len(self.values)

    def _bisect_left(self, target: float) -> int:
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.values[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _bisect_right(self, target: float) -> int:
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.values[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def count_below(self, threshold: float) -> int:
        """Count values <= threshold. O(log n)"""
        return self._bisect_right(threshold)

    def count_above(self, threshold: float) -> int:
        """Count values > threshold. O(log n)"""
        return self.n - self._bisect_right(threshold)

    def fraction_below(self, threshold: float) -> float:
        """Fraction of items below threshold. O(log n)"""
        return self.count_below(threshold) / self.n if self.n > 0 else 0.0

    def find_threshold_for_rate(self, target_rate: float) -> float:
        """Binary search for threshold at target percentile. O(log n)"""
        if target_rate <= 0:
            return self.values[0] if self.n > 0 else 0.0
        if target_rate >= 1:
            return self.values[-1] if self.n > 0 else 0.0
        idx = max(0, min(int(target_rate * self.n), self.n - 1))
        return self.values[idx]

    def percentile(self, p: float) -> float:
        """Get the p-th percentile value. O(1) after sort."""
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be 0-100")
        idx = int(p / 100 * (self.n - 1))
        return self.values[idx]

    def __len__(self) -> int:
        return self.n


class DemandBucketMap:
    """
    Hash map for O(1) demand segment lookups and aggregation.

    Use Case: Dashboard needs instant demand stats by category,
    store, or region without scanning the full dataset.

    Time Complexity:
        - build:      O(n) single pass
        - get_stats:  O(1) lookup
    Space Complexity: O(k) where k = number of unique segments
    """

    def __init__(self):
        self.buckets = {}

    def build(self, segments: List[str], demands: List[float],
              revenues: Optional[List[float]] = None) -> None:
        """Build the hash map in a single O(n) pass."""
        self.buckets = {}
        for i in range(len(segments)):
            seg = segments[i]
            if seg not in self.buckets:
                self.buckets[seg] = {"count": 0, "total_demand": 0.0, "total_revenue": 0.0}
            self.buckets[seg]["count"] += 1
            self.buckets[seg]["total_demand"] += demands[i]
            if revenues is not None:
                self.buckets[seg]["total_revenue"] += revenues[i]

    def get_stats(self, segment: str) -> dict:
        """O(1) lookup for segment statistics."""
        if segment not in self.buckets:
            return {"error": f"Segment '{segment}' not found"}
        b = self.buckets[segment]
        return {
            "segment": segment,
            "count": b["count"],
            "total_demand": round(b["total_demand"], 2),
            "avg_demand": round(b["total_demand"] / b["count"], 2) if b["count"] > 0 else 0,
            "total_revenue": round(b["total_revenue"], 2),
        }

    def get_all_stats(self) -> List[dict]:
        return [self.get_stats(seg) for seg in sorted(self.buckets.keys())]

    @property
    def segments(self) -> List[str]:
        return sorted(self.buckets.keys())


class TopKStockoutHeap:
    """
    Min-heap to efficiently track top-K products at stockout risk.

    Use Case: Real-time dashboard showing top 100 products most
    likely to run out of stock, without sorting all products.

    Time Complexity:
        - push:      O(log k)
        - get_top_k: O(k log k)
    Space Complexity: O(k)
    """

    def __init__(self, k: int = 100):
        self.k = k
        self.heap = []

    def push(self, item_id: str, stockout_risk: float) -> None:
        """Add item to heap. O(log k)."""
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (stockout_risk, item_id))
        elif stockout_risk > self.heap[0][0]:
            heapq.heapreplace(self.heap, (stockout_risk, item_id))

    def get_top_k(self) -> List[Tuple[str, float]]:
        """Return top-K riskiest items, sorted descending. O(k log k)."""
        return [(id_, score) for score, id_ in sorted(self.heap, reverse=True)]

    @property
    def min_score(self) -> float:
        return self.heap[0][0] if self.heap else 0.0

    @property
    def size(self) -> int:
        return len(self.heap)


class InventoryQueue:
    """
    Priority queue for reorder scheduling by urgency.

    Use Case: Warehouse manager needs to reorder products in
    priority order — highest urgency first.

    Time Complexity:
        - enqueue:  O(log n)
        - dequeue:  O(log n)
        - peek:     O(1)
    Space Complexity: O(n)
    """

    def __init__(self):
        self.queue = []  # max-heap via negation
        self.counter = 0

    def enqueue(self, product_id: str, urgency: float, quantity: int) -> None:
        """Add a reorder request. O(log n). Higher urgency = dequeued first."""
        heapq.heappush(self.queue, (-urgency, self.counter, product_id, quantity))
        self.counter += 1

    def dequeue(self) -> Optional[dict]:
        """Get highest-urgency reorder request. O(log n)."""
        if not self.queue:
            return None
        neg_urgency, _, product_id, quantity = heapq.heappop(self.queue)
        return {"product_id": product_id, "urgency": -neg_urgency, "quantity": quantity}

    def peek(self) -> Optional[dict]:
        """See highest-urgency request without removing. O(1)."""
        if not self.queue:
            return None
        neg_urgency, _, product_id, quantity = self.queue[0]
        return {"product_id": product_id, "urgency": -neg_urgency, "quantity": quantity}

    @property
    def size(self) -> int:
        return len(self.queue)

    @property
    def is_empty(self) -> bool:
        return len(self.queue) == 0
