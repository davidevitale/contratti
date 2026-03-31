-- ContractIQ Database Schema
-- This file runs automatically on PostgreSQL first startup

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100) DEFAULT 'general',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    optimization_version INTEGER DEFAULT 0
);

CREATE INDEX idx_projects_client_id ON projects(client_id);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id),
    client_id VARCHAR(100) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    raw_text TEXT,
    extracted_json TEXT,
    contract_score INTEGER,
    model_used VARCHAR(50) DEFAULT 'llama3.2:3b',
    cost_usd FLOAT DEFAULT 0.0,
    pages INTEGER DEFAULT 0,
    rag_ingested BOOLEAN DEFAULT FALSE,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_project_id ON documents(project_id);
CREATE INDEX idx_documents_client_id ON documents(client_id);

-- Messages table (chat history)
CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id),
    client_id VARCHAR(100) NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    sources TEXT DEFAULT '[]',
    user_rating INTEGER,
    user_comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_messages_project_id ON messages(project_id);
CREATE INDEX idx_messages_client_id ON messages(client_id);
CREATE INDEX idx_messages_session_id ON messages(session_id);

-- Usage stats table
CREATE TABLE IF NOT EXISTS usage_stats (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(100) UNIQUE NOT NULL,
    total_cost_usd FLOAT DEFAULT 0.0,
    total_time_saved_minutes INTEGER DEFAULT 0,
    requests_count INTEGER DEFAULT 0,
    feedback_count INTEGER DEFAULT 0,
    optimization_version INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_usage_stats_client_id ON usage_stats(client_id);

-- Insert default usage stats for common clients
INSERT INTO usage_stats (client_id) VALUES ('default_client') ON CONFLICT (client_id) DO NOTHING;
