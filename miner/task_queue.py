from queue import Queue, Empty
from threading import Lock
from miner.arc.models import ARCTask, TaskStatus
from typing import Dict, Any, Optional
import time


class ARCTaskQueue:
    """Thread-safe task queue for managing ARC problems"""
    
    def __init__(self, max_size: int = 100, max_stored_results: int = 1000):
        self.queue = Queue(maxsize=max_size)
        self.tasks = {}  # task_id -> ARCTask
        self.tasks_lock = Lock()
        self.max_stored_results = max_stored_results
        
    def add_task(self, task: ARCTask) -> bool:
        try:
            with self.tasks_lock:
                self.tasks[task.task_id] = task
                self._cleanup_old_tasks()
            
            self.queue.put_nowait(task)
            return True
        except:
            return False
    
    def get_task(self, timeout: float = 1.0) -> Optional[ARCTask]:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
    
    def update_task_status(self, task_id: str, status: TaskStatus, result: Dict = None, error: str = None):
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task.completed_at = time.time()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self.tasks_lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            response = {
                "task_id": task.task_id,
                "status": task.status.value,
                "created_at": task.timestamp,
                "problem_id": task.problem_id,
                "difficulty": task.difficulty
            }
            
            if task.completed_at:
                response["completed_at"] = task.completed_at
                
            if task.status == TaskStatus.COMPLETED and task.result:
                response["result"] = task.result
            elif task.status == TaskStatus.FAILED and task.error:
                response["error"] = task.error
                
            return response
    
    def _cleanup_old_tasks(self):
        """Remove old completed tasks"""
        if len(self.tasks) <= self.max_stored_results:
            return
        
        completed_tasks = [
            (task_id, task.completed_at) 
            for task_id, task in self.tasks.items() 
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and task.completed_at
        ]
        
        if not completed_tasks:
            return
        
        completed_tasks.sort(key=lambda x: x[1])
        to_remove = len(self.tasks) - self.max_stored_results
        
        for task_id, _ in completed_tasks[:to_remove]:
            del self.tasks[task_id]
