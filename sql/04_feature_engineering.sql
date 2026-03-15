-- 04_feature_engineering.sql
-- SQL-based feature engineering using window functions, CTEs, and aggregations

-- ── Lag and Rolling Features ─────────────────────────────
CREATE TABLE IF NOT EXISTS analytics.demand_features AS
WITH lagged AS (
    SELECT
        sale_date, store_id, product_id, category, region, store_type,
        units_sold, revenue, unit_price, competitor_price,
        is_promotion, is_holiday, is_perishable,

        -- Lag features
        LAG(units_sold, 1)  OVER w AS lag_1d,
        LAG(units_sold, 7)  OVER w AS lag_7d,
        LAG(units_sold, 14) OVER w AS lag_14d,
        LAG(units_sold, 30) OVER w AS lag_30d,

        -- Rolling averages (7-day and 30-day)
        AVG(units_sold) OVER (PARTITION BY store_id, product_id
            ORDER BY sale_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7d_mean,
        AVG(units_sold) OVER (PARTITION BY store_id, product_id
            ORDER BY sale_date ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS rolling_30d_mean,

        -- Rolling max/min
        MAX(units_sold) OVER (PARTITION BY store_id, product_id
            ORDER BY sale_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7d_max,
        MIN(units_sold) OVER (PARTITION BY store_id, product_id
            ORDER BY sale_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7d_min,

        -- Year-over-year change
        units_sold - LAG(units_sold, 365) OVER w AS yoy_change,

        -- Demand rank within store
        RANK() OVER (PARTITION BY store_id, sale_date ORDER BY units_sold DESC) AS demand_rank,

        -- Price ratio (ours vs competitor)
        CASE WHEN competitor_price > 0
             THEN ROUND(unit_price / competitor_price, 4)
             ELSE 1.0 END AS price_ratio

    FROM cleaned_data.sales_enriched
    WINDOW w AS (PARTITION BY store_id, product_id ORDER BY sale_date)
),

-- ── Demand Bands and Risk Segments ───────────────────────
segmented AS (
    SELECT
        *,
        -- Demand band
        CASE
            WHEN units_sold < 5 THEN 'Low'
            WHEN units_sold < 20 THEN 'Medium'
            WHEN units_sold < 50 THEN 'High'
            ELSE 'Surge'
        END AS demand_band,

        -- Demand volatility (coefficient of variation proxy)
        CASE
            WHEN rolling_7d_mean > 0
            THEN ROUND((rolling_7d_max - rolling_7d_min)::NUMERIC / rolling_7d_mean, 4)
            ELSE 0
        END AS demand_volatility,

        -- Stockout risk score
        CASE
            WHEN lag_1d = 0 AND lag_7d < rolling_7d_mean * 0.5 THEN 'High'
            WHEN lag_1d < rolling_7d_mean * 0.3 THEN 'Medium'
            ELSE 'Low'
        END AS stockout_risk

    FROM lagged
    WHERE sale_date >= (SELECT MIN(sale_date) + INTERVAL '30 days' FROM lagged)
)

SELECT * FROM segmented;

-- ── Indexes for dashboard queries ──────────────────────
CREATE INDEX IF NOT EXISTS idx_features_date ON analytics.demand_features(sale_date);
CREATE INDEX IF NOT EXISTS idx_features_store ON analytics.demand_features(store_id);
CREATE INDEX IF NOT EXISTS idx_features_product ON analytics.demand_features(product_id);
CREATE INDEX IF NOT EXISTS idx_features_demand_band ON analytics.demand_features(demand_band);
CREATE INDEX IF NOT EXISTS idx_features_stockout ON analytics.demand_features(stockout_risk);
