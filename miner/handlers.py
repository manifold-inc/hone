from __future__ import annotations
import time
import uuid
from typing import Dict, Any, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from miner.arc.models import ARCTask, TaskStatus
from miner.arc.solver import ARCSolver
from miner.task_queue import ARCTaskQueue


_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ARCSolver")
_solver = ARCSolver()
_task_queue = ARCTaskQueue()


def _solve_task_worker():
    """Worker thread for solving tasks"""
    while True:
        task = _task_queue.get_task(timeout=1.0)
        if task:
            try:
                logger.info(f"Processing task {task.task_id} (problem: {task.problem_id}, difficulty: {task.difficulty})")
                
                _task_queue.update_task_status(task.task_id, TaskStatus.PROCESSING)
                
                result = _solver.solve(task.input_grid, task.difficulty)
                
                _task_queue.update_task_status(
                    task.task_id, 
                    TaskStatus.COMPLETED,
                    result={"output": result, "cached": False}
                )
                
                logger.info(f"Completed task {task.task_id}")
                
            except Exception as e:
                logger.error(f"Error solving task {task.task_id}: {e}")
                _task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    error=str(e)
                )


for _ in range(2):
    _executor.submit(_solve_task_worker)


def handle_health() -> Dict[str, Any]:
    """Handle health check requests"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "queue_size": _task_queue.queue.qsize(),
        "solver_status": "operational"
    }


def handle_query(state: Any, query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle ARC problem query requests by creating a task and returning task ID.
    
    Args:
        state: Application state (contains keypair, config, etc.)
        query_data: Query data containing the ARC problem
    
    Returns:
        Response with task_id for polling
    """
    problem_id = query_data.get("problem_id", "unknown")
    input_grid = query_data.get("input", [])
    difficulty = query_data.get("difficulty", "medium")
    
    if not input_grid or not isinstance(input_grid, list):
        logger.error(f"Invalid input grid for problem {problem_id}")
        return {"error": "Invalid input grid"}
    
    task_id = str(uuid.uuid4())
    
    task = ARCTask(
        task_id=task_id,
        problem_id=problem_id,
        input_grid=input_grid,
        difficulty=difficulty,
        timestamp=time.time()
    )
    
    if not _task_queue.add_task(task):
        logger.error(f"Failed to queue task for problem {problem_id}")
        return {"error": "Task queue full"}
    
    state.queries_handled += 1
    state.last_payload = {"problem_id": problem_id, "difficulty": difficulty}
    
    logger.info(f"Created task {task_id} for problem {problem_id}")
    
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "Task queued for processing"
    }


def handle_check_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Check the status of a task.
    
    Args:
        task_id: The task ID to check
    
    Returns:
        Task status and results if complete
    """
    status_info = _task_queue.get_task_status(task_id)
    
    if not status_info:
        logger.warning(f"Task {task_id} not found")
        return None
    
    return status_info