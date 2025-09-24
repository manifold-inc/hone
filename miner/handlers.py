from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Lock
from loguru import logger

@dataclass
class ARCTask:
    problem_id: str
    input_grid: List[List[int]]
    difficulty: str
    timestamp: float


class ARCTaskQueue:
    """Thread-safe task queue for managing ARC problems"""
    
    def __init__(self, max_size: int = 100):
        self.queue = Queue(maxsize=max_size)
        self.results = {}
        self.results_lock = Lock()
        
    def add_task(self, task: ARCTask) -> bool:
        try:
            self.queue.put_nowait(task)
            return True
        except:
            return False
    
    def get_task(self, timeout: float = 1.0) -> Optional[ARCTask]:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
    
    def store_result(self, problem_id: str, result: List[List[int]]):
        with self.results_lock:
            self.results[problem_id] = result
            if len(self.results) > 1000:
                oldest_keys = list(self.results.keys())[:100]
                for key in oldest_keys:
                    del self.results[key]
    
    def get_result(self, problem_id: str) -> Optional[List[List[int]]]:
        with self.results_lock:
            return self.results.get(problem_id)


# Basic ARC solver - Replace with your custom solution
class ARCSolver:
    """
    Basic ARC solver implementation
    """
    
    def __init__(self):
        self.strategies = [
            self._identity_transform,
            self._color_swap,
            self._pattern_complete,
            self._symmetry_detect,
            self._size_transform
        ]
    
    def solve(self, input_grid: List[List[int]], difficulty: str = "medium") -> List[List[int]]:
        """
        Attempt to solve the ARC problem.
        This is a placeholder - implement your actual solving logic here.
        """
        if difficulty == "easy":
            return self._apply_strategy(input_grid, [self._identity_transform, self._color_swap])
        elif difficulty == "hard":
            return self._apply_strategy(input_grid, self.strategies)
        else:
            return self._apply_strategy(input_grid, self.strategies[:3])
    
    def _apply_strategy(self, grid: List[List[int]], strategies: List) -> List[List[int]]:
        """Try different strategies and return the most promising result"""
        for strategy in strategies:
            try:
                result = strategy(grid)
                if self._is_valid_output(result):
                    return result
            except:
                continue
        
        return grid
    
    def _is_valid_output(self, grid: List[List[int]]) -> bool:
        """Check if output is valid"""
        if not grid or not grid[0]:
            return False
        
        if len(grid) > 30 or len(grid[0]) > 30:
            return False
        
        for row in grid:
            for val in row:
                if not isinstance(val, int) or val < 0 or val > 9:
                    return False
        
        return True
    
    def _identity_transform(self, grid: List[List[int]]) -> List[List[int]]:
        """Return the grid as-is"""
        return [row[:] for row in grid]
    
    def _color_swap(self, grid: List[List[int]]) -> List[List[int]]:
        """Swap the two most common colors"""
        flat = [val for row in grid for val in row]
        if not flat:
            return grid
        
        from collections import Counter
        counts = Counter(flat)
        if len(counts) < 2:
            return grid
        
        most_common = counts.most_common(2)
        c1, c2 = most_common[0][0], most_common[1][0]
        
        result = []
        for row in grid:
            new_row = []
            for val in row:
                if val == c1:
                    new_row.append(c2)
                elif val == c2:
                    new_row.append(c1)
                else:
                    new_row.append(val)
            result.append(new_row)
        
        return result
    
    def _pattern_complete(self, grid: List[List[int]]) -> List[List[int]]:
        """Try to complete patterns in the grid"""
        h, w = len(grid), len(grid[0]) if grid else 0
        
        if h < 3 or w < 3:
            return grid
        
        result = [row[:] for row in grid]
        for i in range(h):
            if result[i][0] == result[i][-1] and result[i][0] != 0:
                for j in range(1, w // 2):
                    if result[i][j] == 0:
                        result[i][j] = result[i][w - 1 - j]
                    elif result[i][w - 1 - j] == 0:
                        result[i][w - 1 - j] = result[i][j]
        
        return result
    
    def _symmetry_detect(self, grid: List[List[int]]) -> List[List[int]]:
        """Apply symmetry transformations"""
        return [row[::-1] for row in grid]
    
    def _size_transform(self, grid: List[List[int]]) -> List[List[int]]:
        """Change grid size based on patterns"""
        h, w = len(grid), len(grid[0]) if grid else 0
        
        if h % 2 == 0 and w % 2 == 0 and h > 2 and w > 2:
            result = []
            for i in range(0, h, 2):
                row = []
                for j in range(0, w, 2):
                    block = [
                        grid[i][j], grid[i][j+1],
                        grid[i+1][j], grid[i+1][j+1]
                    ]
                    from collections import Counter
                    most_common = Counter(block).most_common(1)[0][0]
                    row.append(most_common)
                result.append(row)
            return result
        
        return grid


_solver = ARCSolver()
_task_queue = ARCTaskQueue()
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ARCSolver")


def _solve_task_worker():
    """Worker thread for solving tasks"""
    while True:
        task = _task_queue.get_task(timeout=1.0)
        if task:
            try:
                logger.info(f"Solving task {task.problem_id} (difficulty: {task.difficulty})")
                result = _solver.solve(task.input_grid, task.difficulty)
                _task_queue.store_result(task.problem_id, result)
                logger.info(f"Solved task {task.problem_id}")
            except Exception as e:
                logger.error(f"Error solving task {task.problem_id}: {e}")
                _task_queue.store_result(task.problem_id, task.input_grid)


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
    Handle ARC problem query requests.
    
    Args:
        state: Application state (contains keypair, config, etc.)
        query_data: Query data containing the ARC problem
    
    Returns:
        Response with the predicted output
    """
    problem_id = query_data.get("problem_id", "unknown")
    input_grid = query_data.get("input", [])
    difficulty = query_data.get("difficulty", "medium")
    
    if not input_grid or not isinstance(input_grid, list):
        logger.error(f"Invalid input grid for problem {problem_id}")
        return {"error": "Invalid input grid"}
    
    cached_result = _task_queue.get_result(problem_id)
    if cached_result:
        logger.info(f"Returning cached result for problem {problem_id}")
        return {"output": cached_result, "cached": True}
    
    task = ARCTask(
        problem_id=problem_id,
        input_grid=input_grid,
        difficulty=difficulty,
        timestamp=time.time()
    )
    
    future = _executor.submit(_solver.solve, input_grid, difficulty)
    try:
        result = future.result(timeout=25)
        _task_queue.store_result(problem_id, result)
        
        state.queries_handled += 1
        state.last_payload = {"problem_id": problem_id, "difficulty": difficulty}
        
        logger.info(f"Successfully solved problem {problem_id}")
        return {"output": result, "cached": False}
        
    except FuturesTimeoutError:
        logger.warning(f"Timeout solving problem {problem_id}, returning fallback")
        fallback = _solver._identity_transform(input_grid)
        return {"output": fallback, "timeout": True}
    
    except Exception as e:
        logger.error(f"Error handling query for problem {problem_id}: {e}")
        return {"error": str(e)}