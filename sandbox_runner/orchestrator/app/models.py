from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Dict, Any


class TaskState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class CheckMinerRequest(BaseModel):
    miner_uid: str = Field(..., examples=["miner_0xabc..."])
    s3_dataset_url: HttpUrl
    docker_image: str = Field(..., examples=["username/miner-image:latest"])
    task_id: Optional[str] = None

class CheckMinerResponse(BaseModel):
    task_id: str
    state: TaskState
    message: str
    metrics: Optional[Dict[str, Any]] = None
    outputs_s3_uri: Optional[str] = None
    logs_tail: Optional[str] = None