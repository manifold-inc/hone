CREATE TABLE IF NOT EXISTS validator_heartbeat (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    version TEXT NOT NULL,
    cycle_count INTEGER,
    netuid INTEGER,
    wallet_hotkey TEXT
);

CREATE INDEX IF NOT EXISTS idx_validator_heartbeat_ts
    ON validator_heartbeat (ts DESC);


CREATE TABLE IF NOT EXISTS miner_metrics (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    block BIGINT NOT NULL,
    uid INTEGER NOT NULL,
    problem_id TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    response_time DOUBLE PRECISION NOT NULL,

    exact_match BOOLEAN NOT NULL,
    partial_correctness DOUBLE PRECISION NOT NULL,
    grid_similarity DOUBLE PRECISION NOT NULL,
    efficiency_score DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_miner_metrics_block_uid
    ON miner_metrics (block, uid);


CREATE TABLE IF NOT EXISTS batch_summary (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    block BIGINT NOT NULL,

    num_miners INTEGER NOT NULL,
    num_problems INTEGER NOT NULL,
    total_queries INTEGER NOT NULL,
    successful INTEGER NOT NULL,
    exact_matches INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_batch_summary_block
    ON batch_summary (block);
