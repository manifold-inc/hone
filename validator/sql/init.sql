-- Database schema for Hone subnet
CREATE TABLE IF NOT EXISTS miners (
    uid INTEGER PRIMARY KEY,
    hotkey VARCHAR(255),
    ip VARCHAR(45),
    port INTEGER,
    stake REAL,
    last_update_block BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_results (
    id SERIAL PRIMARY KEY,
    block BIGINT NOT NULL,
    uid INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    
    -- New metrics fields
    exact_match BOOLEAN DEFAULT FALSE,
    partial_correctness REAL DEFAULT 0.0,  -- 0-1 score
    grid_similarity REAL DEFAULT 0.0,      -- 0-1 score  
    efficiency_score REAL DEFAULT 0.0,     -- 0-1 score based on response time
    problem_difficulty VARCHAR(20),        -- easy/medium/hard
    problem_id VARCHAR(255),               -- unique identifier for the problem
    
    response JSONB,
    error TEXT,
    response_time REAL,
    timestamp TIMESTAMP NOT NULL,
    
    FOREIGN KEY (uid) REFERENCES miners(uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    uid INTEGER NOT NULL,
    score REAL NOT NULL,
    exact_match_rate REAL DEFAULT 0.0,
    partial_correctness_avg REAL DEFAULT 0.0,
    efficiency_avg REAL DEFAULT 0.0,
    timestamp TIMESTAMP NOT NULL,
    
    FOREIGN KEY (uid) REFERENCES miners(uid) ON DELETE CASCADE
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_query_results_block ON query_results(block);
CREATE INDEX IF NOT EXISTS idx_query_results_uid ON query_results(uid);
CREATE INDEX IF NOT EXISTS idx_query_results_timestamp ON query_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_scores_uid ON scores(uid);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores(timestamp);