-- 01_create_schema.sql
-- Create database schemas for retail demand forecasting
-- Schemas: raw_data, cleaned_data, analytics

CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS cleaned_data;
CREATE SCHEMA IF NOT EXISTS analytics;

COMMENT ON SCHEMA raw_data IS 'Raw ingested data from CSV/Selenium scraping';
COMMENT ON SCHEMA cleaned_data IS 'Cleaned and validated data ready for feature engineering';
COMMENT ON SCHEMA analytics IS 'Feature-engineered data and forecast results';
