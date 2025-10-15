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
    
    -- Metrics fields
    exact_match BOOLEAN DEFAULT FALSE,
    partial_correctness REAL DEFAULT 0.0,
    grid_similarity REAL DEFAULT 0.0,
    efficiency_score REAL DEFAULT 0.0,
    
    -- Task metadata for overfitting analysis
    problem_id VARCHAR(255),
    base_task_num INTEGER,                 -- Original ARC-1 task number
    chain_length INTEGER,                  -- Number of transformations
    transformation_chain JSONB,            -- Full transformation chain
    num_train_examples INTEGER,            -- Training examples provided
    
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
CREATE INDEX IF NOT EXISTS idx_query_results_base_task ON query_results(base_task_num);
CREATE INDEX IF NOT EXISTS idx_query_results_chain_length ON query_results(chain_length);

CREATE INDEX IF NOT EXISTS idx_scores_uid ON scores(uid);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores(timestamp);