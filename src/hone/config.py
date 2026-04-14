"""Environment-based bucket configuration for R2/S3 storage."""

import json
import os

import botocore.config
from dotenv import load_dotenv

from .logging import logger

load_dotenv()


def format_bucket_secrets(base_secret_name: str) -> dict[str, str]:
    return {
        "account_id": os.getenv(f"{base_secret_name}_ACCOUNT_ID"),
        "name": os.getenv(f"{base_secret_name}_BUCKET_NAME"),
        "credentials": {
            "read": {
                "access_key_id": os.getenv(f"{base_secret_name}_READ_ACCESS_KEY_ID"),
                "secret_access_key": os.getenv(
                    f"{base_secret_name}_READ_SECRET_ACCESS_KEY"
                ),
            },
            "write": {
                "access_key_id": os.getenv(f"{base_secret_name}_WRITE_ACCESS_KEY_ID"),
                "secret_access_key": os.getenv(
                    f"{base_secret_name}_WRITE_SECRET_ACCESS_KEY"
                ),
            },
        },
    }


def load_bucket_secrets():
    secrets = {
        "gradients": format_bucket_secrets("R2_GRADIENTS"),
        "aggregator": format_bucket_secrets("R2_AGGREGATOR"),
        "dataset": format_bucket_secrets("R2_DATASET"),
    }

    bucket_list = os.environ.get("R2_DATASET_BUCKET_LIST")
    if bucket_list:
        try:
            dataset_configs = json.loads(bucket_list.strip())
            if isinstance(dataset_configs, list) and len(dataset_configs) > 0:
                secrets["dataset"] = {"multiple": dataset_configs}
        except Exception as e:
            logger.warning(f"Error parsing R2_DATASET_BUCKET_LIST: {e}")

    required_vars = [
        "R2_GRADIENTS_ACCOUNT_ID",
        "R2_GRADIENTS_BUCKET_NAME",
        "R2_GRADIENTS_READ_ACCESS_KEY_ID",
        "R2_GRADIENTS_READ_SECRET_ACCESS_KEY",
        "R2_GRADIENTS_WRITE_ACCESS_KEY_ID",
        "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
        "R2_AGGREGATOR_ACCOUNT_ID",
        "R2_AGGREGATOR_BUCKET_NAME",
        "R2_AGGREGATOR_READ_ACCESS_KEY_ID",
        "R2_AGGREGATOR_READ_SECRET_ACCESS_KEY",
        "R2_AGGREGATOR_WRITE_ACCESS_KEY_ID",
        "R2_AGGREGATOR_WRITE_SECRET_ACCESS_KEY",
        "DATASET_BINS_PATH",
    ]

    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        logger.warning(f"Missing required environment variables: {', '.join(missing)}")
        raise ImportError(f"Required environment variables missing: {missing}")

    return secrets


client_config = botocore.config.Config(
    max_pool_connections=256,
    tcp_keepalive=True,
    retries={"max_attempts": 10, "mode": "adaptive"},
)
BUCKET_SECRETS = load_bucket_secrets()
