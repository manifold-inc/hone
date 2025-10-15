MAINNET_ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"

NETUID_MAINNET = 5

MINER_PORT = 8091
VALIDATOR_PORT = 8092

HEALTH_ENDPOINT = "/health"
QUERY_ENDPOINT = "/query"
CHECK_TASK_ENDPOINT = "/check-task"

BLOCK_TIME = 12

# ARC Problem parameters
MAX_GRID_SIZE = 30
MIN_GRID_SIZE = 1

# Scoring weights
SCORING_WEIGHTS = {
    "exact_match": 0.5,
    "partial_correctness": 0.3,
    "grid_similarity": 0.15,
    "efficiency": 0.05
}

# Minimum requirements
MIN_RESPONSES_FOR_SCORING = 1
MIN_NON_BLACK_CELLS = 6
MIN_DISTINCT_COLORS = 2


MAX_POLL_ATTEMPTS = 50
POLL_INTERVAL = 10