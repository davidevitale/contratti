-- postgres_init/002_user_feedback.sql
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    client_id TEXT NOT NULL,
    session_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment TEXT DEFAULT '',
    context_tags JSONB DEFAULT '[]',
    comment_sentiment TEXT DEFAULT 'neutral' CHECK (comment_sentiment IN ('positive','negative','neutral')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_feedback_client ON user_feedback(client_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON user_feedback(client_id, rating);
CREATE INDEX IF NOT EXISTS idx_feedback_time ON user_feedback(created_at DESC);
