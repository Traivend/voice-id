CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS speakers (
    id BIGSERIAL PRIMARY KEY,
    speaker_key TEXT UNIQUE NOT NULL,
    display_name TEXT,
    embedding vector(192) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS speakers_embedding_hnsw
    ON speakers USING hnsw (embedding vector_cosine_ops);
