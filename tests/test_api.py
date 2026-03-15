"""
test_api.py - Tests for FastAPI schemas and endpoints
======================================================
Run: pytest tests/test_api.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import ValidationError
from src.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    InventoryAllocationRequest,
    ProductCategory,
)


class TestForecastRequestSchema:
    """Test Pydantic validation on ForecastRequest."""

    def test_valid_request(self):
        req = ForecastRequest(
            store_id="S001", product_id="P0042",
            category=ProductCategory.GROCERIES,
            current_price=12.99, competitor_price=13.50,
            is_promotion=0, lag_7d=45, rolling_7d_mean=42.5,
        )
        assert req.store_id == "S001"
        assert req.current_price == 12.99

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            ForecastRequest(
                store_id="S001", product_id="P001",
                category=ProductCategory.DAIRY,
                current_price=-5.0,
            )

    def test_extreme_price_rejected(self):
        with pytest.raises(ValidationError):
            ForecastRequest(
                store_id="S001", product_id="P001",
                category=ProductCategory.ELECTRONICS,
                current_price=6000.0,  # Exceeds $5000 validator
            )

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            ForecastRequest(
                store_id="S001", product_id="P001",
                category="InvalidCategory",
                current_price=10.0,
            )

    def test_discount_boundary(self):
        req = ForecastRequest(
            store_id="S001", product_id="P001",
            category=ProductCategory.SNACKS,
            current_price=5.0, discount_pct=100.0,
        )
        assert req.discount_pct == 100.0

    def test_discount_over_100_rejected(self):
        with pytest.raises(ValidationError):
            ForecastRequest(
                store_id="S001", product_id="P001",
                category=ProductCategory.SNACKS,
                current_price=5.0, discount_pct=150.0,
            )

    def test_defaults_applied(self):
        req = ForecastRequest(
            store_id="S001", product_id="P001",
            category=ProductCategory.BEVERAGES,
            current_price=3.99,
        )
        assert req.is_promotion == 0
        assert req.is_holiday == 0
        assert req.discount_pct == 0
        assert req.lag_7d == 0

    def test_empty_store_id_rejected(self):
        with pytest.raises(ValidationError):
            ForecastRequest(
                store_id="", product_id="P001",
                category=ProductCategory.GROCERIES,
                current_price=10.0,
            )


class TestInventoryAllocationRequest:
    """Test inventory allocation request schema."""

    def test_valid_allocation(self):
        req = InventoryAllocationRequest(
            store_demands=[100.0, 50.0, 30.0],
            store_capacities=[150, 80, 50],
            total_inventory=200,
        )
        assert len(req.store_demands) == 3

    def test_zero_inventory_rejected(self):
        with pytest.raises(ValidationError):
            InventoryAllocationRequest(
                store_demands=[100.0],
                store_capacities=[150],
                total_inventory=0,
            )

    def test_empty_demands_rejected(self):
        with pytest.raises(ValidationError):
            InventoryAllocationRequest(
                store_demands=[],
                store_capacities=[],
                total_inventory=100,
            )
