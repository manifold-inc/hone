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
    train_examples: List[Dict[str, List[List[int]]]]  # List of {"input": grid, "output": grid}
    test_input: List[List[int]]
    timestamp: float
    num_train: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None