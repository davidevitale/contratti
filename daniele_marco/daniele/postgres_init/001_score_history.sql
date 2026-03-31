-- postgres_init/001_score_history.sql
CREATE TABLE IF NOT EXISTS contract_score_history (
    id SERIAL PRIMARY KEY,
    client_id TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    overall_score INTEGER NOT NULL CHECK (overall_score BETWEEN 0 AND 100),
    revenue_quality INTEGER CHECK (revenue_quality BETWEEN 0 AND 100),
    risk_exposure INTEGER CHECK (risk_exposure BETWEEN 0 AND 100),
    strategic_value INTEGER CHECK (strategic_value BETWEEN 0 AND 100),
    concentration_risk INTEGER CHECK (concentration_risk BETWEEN 0 AND 100),
    score_data JSONB DEFAULT '{}',
    scored_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_score_history_client ON contract_score_history(client_id);
CREATE INDEX IF NOT EXISTS idx_score_history_contract ON contract_score_history(client_id, contract_id);
CREATE INDEX IF NOT EXISTS idx_score_history_time ON contract_score_history(scored_at DESC);
