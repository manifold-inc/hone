from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ARCTask:
    task_id: str
    problem_id: str
    input_grid: List[List[int]]
    difficulty: str
    timestamp: float
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None
