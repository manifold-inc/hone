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
                logger.info(f"Processing task {task.task_id} (problem: {task.problem_id}, "
                           f"train_examples: {task.num_train})")
                
                _task_queue.update_task_status(task.task_id, TaskStatus.PROCESSING)
                
                result = _solver.solve(task.train_examples, task.test_input)
                
                _task_queue.update_task_status(
                    task.task_id, 
                    TaskStatus.COMPLETED,
                    result={"output": result, "cached": False}
                )
                
                logger.info(f"Completed task {task.task_id}")
                
            except Exception as e:
                logger.error(f"Error solving task {task.task_id}: {e}")
                logger.exception(e)
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
    Handle ARC problem query requests with training examples.
    
    Args:
        state: Application state (contains keypair, config, etc.)
        query_data: Query data containing training examples and test input
    
    Returns:
        Response with task_id for polling
    """
    problem_id = query_data.get("problem_id", "unknown")
    train_examples = query_data.get("train_examples", [])
    test_input = query_data.get("test_input", [])
    num_train = query_data.get("num_train", len(train_examples))
    
    logger.info(f"Received problem {problem_id}: {num_train} training examples")
    logger.debug(f"Query data keys: {query_data.keys()}")
    logger.debug(f"Train examples type: {type(train_examples)}, length: {len(train_examples) if isinstance(train_examples, list) else 'N/A'}")
    
    if not isinstance(train_examples, list):
        logger.error(f"Invalid training examples type for problem {problem_id}: {type(train_examples)}")
        return {"error": f"Invalid training examples type: {type(train_examples)}"}
    
    if len(train_examples) == 0:
        logger.error(f"Empty training examples for problem {problem_id}")
        return {"error": "Empty training examples"}
    
    if not isinstance(test_input, list) or len(test_input) == 0:
        logger.error(f"Invalid test input for problem {problem_id}: type={type(test_input)}, len={len(test_input) if isinstance(test_input, list) else 'N/A'}")
        return {"error": "Invalid or empty test input"}
    
    for i, ex in enumerate(train_examples):
        if not isinstance(ex, dict):
            logger.error(f"Invalid training example {i} for problem {problem_id}: not a dict, got {type(ex)}")
            return {"error": f"Invalid training example {i}: not a dict"}
        
        if "input" not in ex or "output" not in ex:
            logger.error(f"Invalid training example {i} for problem {problem_id}: missing 'input' or 'output' keys. Keys present: {ex.keys()}")
            return {"error": f"Invalid training example {i}: missing 'input' or 'output'"}
        
        if not isinstance(ex["input"], list) or not isinstance(ex["output"], list):
            logger.error(f"Invalid training example {i} for problem {problem_id}: input/output not lists")
            return {"error": f"Invalid training example {i}: input/output not lists"}
    
    task_id = str(uuid.uuid4())
    
    task = ARCTask(
        task_id=task_id,
        problem_id=problem_id,
        train_examples=train_examples,
        test_input=test_input,
        num_train=num_train,
        timestamp=time.time()
    )
    
    if not _task_queue.add_task(task):
        logger.error(f"Failed to queue task for problem {problem_id}")
        return {"error": "Task queue full"}
    
    state.queries_handled += 1
    state.last_payload = {
        "problem_id": problem_id, 
        "num_train": num_train
    }
    
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