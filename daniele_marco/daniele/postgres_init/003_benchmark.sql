-- ContractIQ Benchmark Schema and Opt-in Setup

-- Add opt-in columns to usage_stats which tracks per-client settings
ALTER TABLE usage_stats ADD COLUMN IF NOT EXISTS benchmark_opt_in BOOLEAN DEFAULT FALSE;
ALTER TABLE usage_stats ADD COLUMN IF NOT EXISTS benchmark_opt_in_date TIMESTAMP;

-- Create a dedicated schema for benchmark to guarantee data isolation as promised
CREATE SCHEMA IF NOT EXISTS benchmark_schema;

-- Create the table for anonymized clause values
CREATE TABLE IF NOT EXISTS benchmark_schema.benchmark_store (
    id SERIAL PRIMARY KEY,
    clause_type VARCHAR(150) NOT NULL,
    sector VARCHAR(100) NOT NULL,
    country VARCHAR(10) NOT NULL,
    deal_size_bucket VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for fast querying during aggregations and statistics lookups
CREATE INDEX IF NOT EXISTS idx_benchmark_store_clause_sector ON benchmark_schema.benchmark_store(clause_type, sector);
CREATE INDEX IF NOT EXISTS idx_benchmark_store_created_at ON benchmark_schema.benchmark_store(created_at);
