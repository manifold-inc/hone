from __future__ import annotations
import os
from dataclasses import dataclass


def _get_env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


@dataclass(frozen=True)
class Settings:
    max_concurrency: int = int(os.getenv("ORCH_MAX_CONCURRENCY", "4"))
    child_timeout_min: int = int(os.getenv("ORCH_CHILD_TIMEOUT_MIN", "30"))
    child_memory: str = os.getenv("ORCH_CHILD_MEMORY", "8g") # e.g., 2g, 4096m
    child_cpus: float = float(os.getenv("ORCH_CHILD_CPUS", "2.0"))
    child_pids: int = int(os.getenv("ORCH_CHILD_PIDS", "512"))
    pull_always: bool = os.getenv("ORCH_PULL_ALWAYS", "false").lower() == "true"
    secure_mode: bool = os.getenv("ORCH_SECURE_MODE", "true").lower() == "true"


    data_root: str = os.getenv("ORCH_DATA_ROOT", "/orchestrator/data")
    out_root: str = os.getenv("ORCH_OUT_ROOT", "/orchestrator/out")
    tmp_root: str = os.getenv("ORCH_TMP_ROOT", "/orchestrator/tmp")


    aws_region: str = os.getenv("AWS_REGION", "eu-central-1")
    s3_metrics_bucket: str | None = os.getenv("ORCH_S3_METRICS_BUCKET")


    log_level: str = os.getenv("ORCH_LOG_LEVEL", "INFO")


settings = Settings()