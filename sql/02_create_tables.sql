-- 02_create_tables.sql
-- Create tables for retail demand forecasting pipeline

-- ── Raw Data Tables ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS raw_data.sales_transactions (
    id              SERIAL PRIMARY KEY,
    sale_date       DATE NOT NULL,
    store_id        VARCHAR(10) NOT NULL,
    product_id      VARCHAR(10) NOT NULL,
    category        VARCHAR(50),
    units_sold      INTEGER,
    unit_price      NUMERIC(10,2),
    revenue         NUMERIC(12,2),
    discount_pct    NUMERIC(5,2) DEFAULT 0,
    is_promotion    SMALLINT DEFAULT 0,
    is_holiday      SMALLINT DEFAULT 0,
    competitor_price NUMERIC(10,2),
    day_of_week     SMALLINT,
    month           SMALLINT,
    year            SMALLINT,
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_data.stores (
    store_id        VARCHAR(10) PRIMARY KEY,
    region          VARCHAR(20),
    store_type      VARCHAR(20),
    store_size_sqft INTEGER,
    opening_year    SMALLINT
);

CREATE TABLE IF NOT EXISTS raw_data.products (
    product_id      VARCHAR(10) PRIMARY KEY,
    category        VARCHAR(50),
    sub_category    VARCHAR(50),
    base_price      NUMERIC(10,2),
    cost_price      NUMERIC(10,2),
    is_perishable   SMALLINT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS raw_data.competitor_prices (
    id              SERIAL PRIMARY KEY,
    scrape_date     DATE NOT NULL,
    product_id      VARCHAR(10),
    competitor_name VARCHAR(50),
    competitor_price NUMERIC(10,2),
    source_url      TEXT,
    scraped_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Indexes ──────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_sales_date ON raw_data.sales_transactions(sale_date);
CREATE INDEX IF NOT EXISTS idx_sales_store ON raw_data.sales_transactions(store_id);
CREATE INDEX IF NOT EXISTS idx_sales_product ON raw_data.sales_transactions(product_id);
CREATE INDEX IF NOT EXISTS idx_sales_category ON raw_data.sales_transactions(category);
CREATE INDEX IF NOT EXISTS idx_competitor_date ON raw_data.competitor_prices(scrape_date);
