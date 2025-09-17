CREATE SCHEMA IF NOT EXISTS hone;
ALTER DATABASE hone SET search_path TO hone, public;
SET search_path TO hone, public;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS miners (
    uid INTEGER PRIMARY KEY,
    hotkey VARCHAR(64) NOT NULL UNIQUE,
    ip VARCHAR(45),
    port INTEGER,
    stake DECIMAL(20, 9),
    last_update_block INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    block INTEGER NOT NULL,
    uid INTEGER NOT NULL REFERENCES miners(uid),
    success BOOLEAN NOT NULL,
    response JSONB,
    error TEXT,
    response_time FLOAT,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_results_block ON query_results(block);
CREATE INDEX IF NOT EXISTS idx_query_results_uid ON query_results(uid);
CREATE INDEX IF NOT EXISTS idx_query_results_timestamp ON query_results(timestamp);

CREATE TABLE IF NOT EXISTS scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    uid INTEGER NOT NULL REFERENCES miners(uid),
    score FLOAT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scores_uid ON scores(uid);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores(timestamp);