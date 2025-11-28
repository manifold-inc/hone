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

CREATE TABLE IF NOT EXISTS submission_history (
    id SERIAL PRIMARY KEY,
    hotkey VARCHAR(255) NOT NULL,
    repo_url VARCHAR(500) NOT NULL,
    repo_branch VARCHAR(255) NOT NULL,
    repo_commit VARCHAR(100) DEFAULT '',
    repo_path VARCHAR(500) DEFAULT '',
    weight_class VARCHAR(50) NOT NULL,
    use_vllm BOOLEAN DEFAULT FALSE,
    vllm_config JSONB,
    
    exact_match_rate REAL DEFAULT 0.0,
    
    first_submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluation_count INTEGER DEFAULT 1,
    
    UNIQUE (hotkey, repo_url, repo_branch, repo_commit, repo_path, weight_class)
);

CREATE TABLE IF NOT EXISTS daily_submissions (
    id SERIAL PRIMARY KEY,
    hotkey VARCHAR(255) NOT NULL,
    submission_date DATE NOT NULL DEFAULT CURRENT_DATE,
    submission_count INTEGER DEFAULT 1,
    last_submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (hotkey, submission_date)
);

CREATE TABLE IF NOT EXISTS query_results (
    id SERIAL PRIMARY KEY,
    block BIGINT NOT NULL,
    uid INTEGER NOT NULL,
    hotkey VARCHAR(255) NOT NULL,
    success BOOLEAN NOT NULL,
    
    repo_url VARCHAR(500),
    repo_branch VARCHAR(255),
    repo_commit VARCHAR(100),
    repo_path VARCHAR(500),
    weight_class VARCHAR(50),
    
    from_cache BOOLEAN DEFAULT FALSE,
    
    exact_match_rate REAL DEFAULT 0.0,
    
    response JSONB,
    error TEXT,
    response_time REAL,
    timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS leaderboard (
    id SERIAL PRIMARY KEY,
    hotkey VARCHAR(255) NOT NULL UNIQUE,
    uid INTEGER,
    exact_match_rate REAL DEFAULT 0.0,
    
    repo_url VARCHAR(500),
    repo_branch VARCHAR(255),
    repo_commit VARCHAR(100),
    repo_path VARCHAR(500),
    
    first_achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluation_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    uid INTEGER NOT NULL,
    hotkey VARCHAR(255) NOT NULL,
    exact_match_rate REAL DEFAULT 0.0,
    timestamp TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_query_results_block ON query_results(block);
CREATE INDEX IF NOT EXISTS idx_query_results_uid ON query_results(uid);
CREATE INDEX IF NOT EXISTS idx_query_results_hotkey ON query_results(hotkey);
CREATE INDEX IF NOT EXISTS idx_query_results_timestamp ON query_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_results_exact_match_rate ON query_results(exact_match_rate);

CREATE INDEX IF NOT EXISTS idx_scores_uid ON scores(uid);
CREATE INDEX IF NOT EXISTS idx_scores_hotkey ON scores(hotkey);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores(timestamp);

CREATE INDEX IF NOT EXISTS idx_submission_history_hotkey ON submission_history(hotkey);
CREATE INDEX IF NOT EXISTS idx_submission_history_lookup ON submission_history(hotkey, repo_url, repo_branch);

CREATE INDEX IF NOT EXISTS idx_daily_submissions_hotkey_date ON daily_submissions(hotkey, submission_date);

CREATE INDEX IF NOT EXISTS idx_leaderboard_exact_match_rate ON leaderboard(exact_match_rate DESC);