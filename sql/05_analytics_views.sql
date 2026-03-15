-- 05_analytics_views.sql
-- Materialized views for Tableau dashboards and API queries

-- ── 1. Daily demand summary by store ──────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_daily_store_demand AS
SELECT
    sale_date,
    store_id,
    region,
    store_type,
    COUNT(*) AS num_transactions,
    SUM(units_sold) AS total_units,
    SUM(revenue) AS total_revenue,
    AVG(units_sold) AS avg_units,
    AVG(price_ratio) AS avg_price_ratio
FROM analytics.demand_features
GROUP BY sale_date, store_id, region, store_type
ORDER BY sale_date, store_id;

-- ── 2. Category demand summary ────────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_category_demand AS
SELECT
    category,
    COUNT(*) AS total_records,
    SUM(units_sold) AS total_units,
    SUM(revenue) AS total_revenue,
    AVG(units_sold) AS avg_daily_demand,
    AVG(demand_volatility) AS avg_volatility,
    SUM(CASE WHEN stockout_risk = 'High' THEN 1 ELSE 0 END) AS high_stockout_count,
    ROUND(SUM(CASE WHEN is_promotion = 1 THEN units_sold ELSE 0 END)::NUMERIC /
          NULLIF(SUM(CASE WHEN is_promotion = 0 THEN units_sold ELSE 0 END), 0), 4)
          AS promo_demand_lift
FROM analytics.demand_features
GROUP BY category
ORDER BY total_revenue DESC;

-- ── 3. Regional demand heatmap ────────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_regional_demand AS
SELECT
    region,
    EXTRACT(MONTH FROM sale_date) AS month,
    EXTRACT(YEAR FROM sale_date) AS year,
    SUM(units_sold) AS total_units,
    SUM(revenue) AS total_revenue,
    AVG(units_sold) AS avg_demand,
    COUNT(DISTINCT store_id) AS active_stores,
    COUNT(DISTINCT product_id) AS active_products
FROM analytics.demand_features
GROUP BY region, EXTRACT(MONTH FROM sale_date), EXTRACT(YEAR FROM sale_date)
ORDER BY year, month, region;

-- ── 4. Stockout risk summary ──────────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_stockout_summary AS
SELECT
    product_id,
    category,
    COUNT(*) AS total_days,
    SUM(CASE WHEN stockout_risk = 'High' THEN 1 ELSE 0 END) AS high_risk_days,
    SUM(CASE WHEN stockout_risk = 'Medium' THEN 1 ELSE 0 END) AS medium_risk_days,
    ROUND(SUM(CASE WHEN stockout_risk = 'High' THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 100, 2)
        AS stockout_frequency_pct,
    AVG(units_sold) AS avg_demand,
    AVG(demand_volatility) AS avg_volatility
FROM analytics.demand_features
GROUP BY product_id, category
HAVING COUNT(*) > 30
ORDER BY stockout_frequency_pct DESC;

-- ── 5. Promotion impact analysis ──────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_promo_impact AS
SELECT
    category,
    is_promotion,
    COUNT(*) AS record_count,
    AVG(units_sold) AS avg_demand,
    AVG(revenue) AS avg_revenue,
    AVG(price_ratio) AS avg_price_ratio
FROM analytics.demand_features
GROUP BY category, is_promotion
ORDER BY category, is_promotion;

-- ── Refresh views (run periodically) ──────────────────────
-- REFRESH MATERIALIZED VIEW analytics.mv_daily_store_demand;
-- REFRESH MATERIALIZED VIEW analytics.mv_category_demand;
-- REFRESH MATERIALIZED VIEW analytics.mv_regional_demand;
-- REFRESH MATERIALIZED VIEW analytics.mv_stockout_summary;
-- REFRESH MATERIALIZED VIEW analytics.mv_promo_impact;
