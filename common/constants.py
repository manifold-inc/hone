MAINNET_ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"

NETUID_MAINNET = 5

DEFAULT_TIMEOUT = 30 
HEALTH_CHECK_TIMEOUT = 5

DEFAULT_HEALTH_BODY = {"status": "healthy"}

MINER_PORT = 8091
VALIDATOR_PORT = 8092

HEALTH_ENDPOINT = "/health"
QUERY_ENDPOINT = "/query"

BLOCK_TIME = 12

# ARC Problem parameters
MAX_GRID_SIZE = 30
MIN_GRID_SIZE = 1
MAX_RESPONSE_TIME = 30.0  # seconds
SUPPORTED_DIFFICULTIES = ["easy", "medium", "hard"]

# Scoring weights
SCORING_WEIGHTS = {
    "exact_match": 0.4,
    "partial_correctness": 0.3,
    "grid_similarity": 0.2,
    "efficiency": 0.1
}

# Minimum requirements
MIN_RESPONSES_FOR_SCORING = 1
MIN_NON_BLACK_CELLS = 6
MIN_DISTINCT_COLORS = 2