"""
schemas.py - Pydantic Models for API Request/Response
======================================================
Strong typing with validation for the demand forecasting API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class ProductCategory(str, Enum):
    GROCERIES = "Groceries"
    DAIRY = "Dairy"
    BEVERAGES = "Beverages"
    SNACKS = "Snacks"
    FROZEN = "Frozen"
    PERSONAL_CARE = "Personal Care"
    HOUSEHOLD = "Household"
    ELECTRONICS = "Electronics"
    CLOTHING = "Clothing"
    TOYS = "Toys"


class StoreType(str, Enum):
    SUPERMARKET = "Supermarket"
    EXPRESS = "Express"
    WAREHOUSE = "Warehouse"
    ONLINE = "Online"


class ForecastRequest(BaseModel):
    """Input schema for a demand forecast request."""

    store_id: str = Field(..., min_length=1, max_length=10, description="Store identifier")
    product_id: str = Field(..., min_length=1, max_length=10, description="Product identifier")
    category: ProductCategory = Field(..., description="Product category")
    current_price: float = Field(..., gt=0, le=10000, description="Current selling price")
    competitor_price: float = Field(default=0, ge=0, le=10000, description="Competitor price")
    is_promotion: int = Field(default=0, ge=0, le=1, description="Active promotion flag")
    discount_pct: float = Field(default=0, ge=0, le=100, description="Discount percentage")
    is_holiday: int = Field(default=0, ge=0, le=1, description="Holiday flag")
    lag_7d: float = Field(default=0, ge=0, description="Demand 7 days ago")
    lag_30d: float = Field(default=0, ge=0, description="Demand 30 days ago")
    rolling_7d_mean: float = Field(default=0, ge=0, description="7-day rolling avg demand")
    rolling_30d_mean: float = Field(default=0, ge=0, description="30-day rolling avg demand")

    @field_validator("current_price")
    @classmethod
    def price_must_be_reasonable(cls, v):
        if v > 5000:
            raise ValueError("Price exceeds $5,000 — please verify")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "store_id": "S001", "product_id": "P0042",
                "category": "Groceries", "current_price": 12.99,
                "competitor_price": 13.50, "is_promotion": 0,
                "lag_7d": 45, "rolling_7d_mean": 42.5,
            }
        }


class ForecastResponse(BaseModel):
    """Output schema for a demand forecast."""

    request_id: str
    store_id: str
    product_id: str
    predicted_demand: float = Field(..., ge=0)
    demand_band: str  # Low / Medium / High / Surge
    reorder_recommendation: str  # REORDER_NOW / MONITOR / SUFFICIENT
    confidence: float = Field(..., ge=0, le=1)
    risk_factors: List[str]
    cached: bool = False


class BatchForecastRequest(BaseModel):
    """Batch forecast input (up to 500 items)."""
    items: List[ForecastRequest] = Field(..., max_length=500)


class InventoryAllocationRequest(BaseModel):
    """Input for DP inventory allocation."""

    store_demands: List[float] = Field(..., min_length=1, max_length=100)
    store_capacities: List[int] = Field(..., min_length=1, max_length=100)
    total_inventory: int = Field(..., gt=0, le=1000000)


class InventoryAllocationResponse(BaseModel):
    """Output for DP allocation."""

    allocation: List[int]
    total_fulfilled: float
    total_demand: float
    fill_rate: float
    inventory_used: int
    inventory_remaining: int


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "healthy"
    model_loaded: bool
    cache_size: int
    cache_hit_rate: float
    version: str = "1.0.0"
