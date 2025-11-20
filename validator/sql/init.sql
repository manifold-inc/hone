CREATE TABLE IF NOT EXISTS miners (
    uid INTEGER PRIMARY KEY,
    hotkey VARCHAR(255) UNIQUE NOT NULL,
    ip VARCHAR(45),
    port INTEGER,
    stake REAL,
    last_update_block BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- submission history to avoid re-evaluating identical solutions
CREATE TABLE IF NOT EXISTS submission_history (
    id SERIAL PRIMARY KEY,
    hotkey VARCHAR(255) NOT NULL,
    repo_url VARCHAR(500) NOT NULL,
    repo_branch VARCHAR(255) NOT NULL,
    repo_commit VARCHAR(100),
    repo_path VARCHAR(500),
    weight_class VARCHAR(50),
    use_vllm BOOLEAN DEFAULT FALSE,
    vllm_config JSONB,
    
    -- cached metrics from last evaluation
    exact_match_rate REAL DEFAULT 0.0,
    partial_correctness_avg REAL DEFAULT 0.0,
    grid_similarity_avg REAL DEFAULT 0.0,
    efficiency_avg REAL DEFAULT 0.0,
    overall_score REAL DEFAULT 0.0,
    
    first_submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluation_count INTEGER DEFAULT 1,
    
    FOREIGN KEY (hotkey) REFERENCES miners(hotkey) ON DELETE CASCADE,
    
    -- unique constraint on solution configuration
    UNIQUE (hotkey, repo_url, repo_branch, COALESCE(repo_commit, ''), repo_path, weight_class)
);

-- daily submission tracking
CREATE TABLE IF NOT EXISTS daily_submissions (
    id SERIAL PRIMARY KEY,
    hotkey VARCHAR(255) NOT NULL,
    submission_date DATE NOT NULL DEFAULT CURRENT_DATE,
    submission_count INTEGER DEFAULT 1,
    last_submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (hotkey) REFERENCES miners(hotkey) ON DELETE CASCADE,
    UNIQUE (hotkey, submission_date)
);

CREATE TABLE IF NOT EXISTS query_results (
    id SERIAL PRIMARY KEY,
    block BIGINT NOT NULL,
    uid INTEGER NOT NULL,
    hotkey VARCHAR(255) NOT NULL,
    success BOOLEAN NOT NULL,
    
    -- solution identification (for history matching)
    repo_url VARCHAR(500),
    repo_branch VARCHAR(255),
    repo_commit VARCHAR(100),
    repo_path VARCHAR(500),
    weight_class VARCHAR(50),
    
    -- whether this result was from cache or fresh evaluation
    from_cache BOOLEAN DEFAULT FALSE,
    
    -- metrics fields
    exact_match BOOLEAN DEFAULT FALSE,
    partial_correctness REAL DEFAULT 0.0,
    grid_similarity REAL DEFAULT 0.0,
    efficiency_score REAL DEFAULT 0.0,
    
    -- task metadata for overfitting analysis
    problem_id VARCHAR(255),
    base_task_num INTEGER,
    chain_length INTEGER,
    transformation_chain JSONB,
    num_train_examples INTEGER,
    
    response JSONB,
    error TEXT,
    response_time REAL,
    timestamp TIMESTAMP NOT NULL,
    
    FOREIGN KEY (uid) REFERENCES miners(uid) ON DELETE CASCADE,
    FOREIGN KEY (hotkey) REFERENCES miners(hotkey) ON DELETE CASCADE
);

-- top miners leaderboard (current top 5)
CREATE TABLE IF NOT EXISTS leaderboard (
    id SERIAL PRIMARY KEY,
    hotkey VARCHAR(255) NOT NULL UNIQUE,
    uid INTEGER,
    overall_score REAL NOT NULL,
    exact_match_rate REAL DEFAULT 0.0,
    partial_correctness_avg REAL DEFAULT 0.0,
    grid_similarity_avg REAL DEFAULT 0.0,
    efficiency_avg REAL DEFAULT 0.0,
    
    -- submission info
    repo_url VARCHAR(500),
    repo_branch VARCHAR(255),
    repo_commit VARCHAR(100),
    repo_path VARCHAR(500),
    
    first_achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluation_count INTEGER DEFAULT 0,
    
    FOREIGN KEY (hotkey) REFERENCES miners(hotkey) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    uid INTEGER NOT NULL,
    hotkey VARCHAR(255) NOT NULL,
    score REAL NOT NULL,
    exact_match_rate REAL DEFAULT 0.0,
    partial_correctness_avg REAL DEFAULT 0.0,
    efficiency_avg REAL DEFAULT 0.0,
    timestamp TIMESTAMP NOT NULL,
    
    FOREIGN KEY (uid) REFERENCES miners(uid) ON DELETE CASCADE,
    FOREIGN KEY (hotkey) REFERENCES miners(hotkey) ON DELETE CASCADE
);

-- indices for performance
CREATE INDEX IF NOT EXISTS idx_query_results_block ON query_results(block);
CREATE INDEX IF NOT EXISTS idx_query_results_uid ON query_results(uid);
CREATE INDEX IF NOT EXISTS idx_query_results_hotkey ON query_results(hotkey);
CREATE INDEX IF NOT EXISTS idx_query_results_timestamp ON query_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_results_base_task ON query_results(base_task_num);
CREATE INDEX IF NOT EXISTS idx_query_results_chain_length ON query_results(chain_length);

CREATE INDEX IF NOT EXISTS idx_scores_uid ON scores(uid);
CREATE INDEX IF NOT EXISTS idx_scores_hotkey ON scores(hotkey);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores(timestamp);

CREATE INDEX IF NOT EXISTS idx_submission_history_hotkey ON submission_history(hotkey);
CREATE INDEX IF NOT EXISTS idx_submission_history_lookup ON submission_history(hotkey, repo_url, repo_branch);

CREATE INDEX IF NOT EXISTS idx_daily_submissions_hotkey_date ON daily_submissions(hotkey, submission_date);

CREATE INDEX IF NOT EXISTS idx_leaderboard_score ON leaderboard(overall_score DESC);