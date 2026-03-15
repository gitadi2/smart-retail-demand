-- 03_etl_pipeline.sql
-- ETL: Clean raw data → cleaned_data schema

-- ── Step 1: Deduplicate ──────────────────────────────────
CREATE TABLE IF NOT EXISTS cleaned_data.sales_deduped AS
SELECT DISTINCT ON (sale_date, store_id, product_id)
    sale_date, store_id, product_id, category,
    units_sold, unit_price, revenue,
    discount_pct, is_promotion, is_holiday,
    competitor_price, day_of_week, month, year
FROM raw_data.sales_transactions
ORDER BY sale_date, store_id, product_id, loaded_at DESC;

-- ── Step 2: Remove invalid records ────────────────────────
DELETE FROM cleaned_data.sales_deduped
WHERE units_sold < 0
   OR unit_price <= 0
   OR revenue < 0
   OR sale_date IS NULL;

-- ── Step 3: Cap outliers at 99th percentile ────────────────
WITH percentiles AS (
    SELECT
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY units_sold) AS units_p99,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY unit_price) AS price_p99,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY revenue) AS rev_p99
    FROM cleaned_data.sales_deduped
)
UPDATE cleaned_data.sales_deduped s
SET
    units_sold = LEAST(s.units_sold, p.units_p99::INTEGER),
    unit_price = LEAST(s.unit_price, p.price_p99),
    revenue = LEAST(s.revenue, p.rev_p99)
FROM percentiles p;

-- ── Step 4: Fill missing competitor prices with median ─────
UPDATE cleaned_data.sales_deduped
SET competitor_price = (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY competitor_price)
    FROM cleaned_data.sales_deduped
    WHERE competitor_price IS NOT NULL
)
WHERE competitor_price IS NULL;

-- ── Step 5: Create cleaned sales with store/product joins ──
CREATE TABLE IF NOT EXISTS cleaned_data.sales_enriched AS
SELECT
    s.*,
    st.region,
    st.store_type,
    st.store_size_sqft,
    p.sub_category,
    p.base_price,
    p.cost_price,
    p.is_perishable,
    -- Derived fields
    s.revenue - (s.units_sold * p.cost_price) AS gross_profit,
    CASE WHEN p.base_price > 0
         THEN ROUND((p.base_price - s.unit_price) / p.base_price * 100, 2)
         ELSE 0 END AS effective_discount
FROM cleaned_data.sales_deduped s
LEFT JOIN raw_data.stores st ON s.store_id = st.store_id
LEFT JOIN raw_data.products p ON s.product_id = p.product_id;

CREATE INDEX IF NOT EXISTS idx_enriched_date ON cleaned_data.sales_enriched(sale_date);
CREATE INDEX IF NOT EXISTS idx_enriched_store ON cleaned_data.sales_enriched(store_id);
CREATE INDEX IF NOT EXISTS idx_enriched_product ON cleaned_data.sales_enriched(product_id);
