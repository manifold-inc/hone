"""
Manages dataset persistence and task tracking
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DatasetStorage:
    """Manages task history and dataset persistence"""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Files
        self.task_hashes_file = storage_dir / "task_hashes.json"
        self.unsolved_tasks_file = storage_dir / "unsolved_tasks.json"
        self.current_dataset_file = storage_dir / "current_dataset.json"
        self.history_dir = storage_dir / "history"
        self.history_dir.mkdir(exist_ok=True)
        
        # Load existing data
        self.seen_task_hashes: Set[str] = self._load_task_hashes()
        self.unsolved_tasks: List[Dict] = self._load_unsolved_tasks()
    
    def hash_task(self, task: Dict) -> str:
        """Generate deterministic hash for a task"""
        # Hash based on input grids and transformation chain
        metadata = task.get("metadata", {})
        
        hash_data = {
            "base_task": metadata.get("base_task"),
            "transformation_chain": metadata.get("transformation_chain"),
            "train_count": len(task.get("train_examples", [])),
        }
        
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def is_task_seen(self, task_hash: str) -> bool:
        """Check if task was previously generated"""
        return task_hash in self.seen_task_hashes
    
    def mark_task_seen(self, task_hash: str):
        """Mark task as seen"""
        self.seen_task_hashes.add(task_hash)
        self._save_task_hashes()
    
    def save_current_dataset(self, tasks: List[Dict]):
        """Save current active dataset"""
        dataset = {
            "generated_at": datetime.utcnow().isoformat(),
            "task_count": len(tasks),
            "tasks": tasks
        }
        
        with open(self.current_dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Archive to history
        date_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_file = self.history_dir / f"dataset_{date_str}.json"
        with open(archive_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved dataset with {len(tasks)} tasks")
    
    def load_current_dataset(self) -> List[Dict]:
        """Load current active dataset"""
        if not self.current_dataset_file.exists():
            return []
        
        with open(self.current_dataset_file, 'r') as f:
            dataset = json.load(f)
        
        return dataset.get("tasks", [])
    
    def update_unsolved_tasks(self, task_results: List[Dict]):
        """
        Update unsolved tasks based on miner results
        
        Args:
            task_results: List of dicts with {task_hash, solved, metrics}
        """
        # Keep tasks that weren't solved
        new_unsolved = []
        
        for result in task_results:
            if not result.get("solved", False):
                # Find the full task data
                task_hash = result["task_hash"]
                task_data = self._find_task_by_hash(task_hash)
                
                if task_data:
                    new_unsolved.append({
                        "task_hash": task_hash,
                        "task": task_data,
                        "attempts": result.get("attempts", 0) + 1,
                        "last_attempt": datetime.utcnow().isoformat()
                    })
        
        self.unsolved_tasks = new_unsolved
        self._save_unsolved_tasks()
        
        logger.info(f"Updated unsolved tasks: {len(new_unsolved)} tasks remain unsolved")
    
    def get_unsolved_tasks(self) -> List[Dict]:
        """Get tasks that miners haven't solved"""
        return [item["task"] for item in self.unsolved_tasks]
    
    def _load_task_hashes(self) -> Set[str]:
        """Load set of seen task hashes"""
        if not self.task_hashes_file.exists():
            return set()
        
        with open(self.task_hashes_file, 'r') as f:
            data = json.load(f)
        
        return set(data.get("hashes", []))
    
    def _save_task_hashes(self):
        """Save seen task hashes"""
        data = {
            "count": len(self.seen_task_hashes),
            "last_updated": datetime.utcnow().isoformat(),
            "hashes": list(self.seen_task_hashes)
        }
        
        with open(self.task_hashes_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_unsolved_tasks(self) -> List[Dict]:
        """Load unsolved tasks"""
        if not self.unsolved_tasks_file.exists():
            return []
        
        with open(self.unsolved_tasks_file, 'r') as f:
            data = json.load(f)
        
        return data.get("tasks", [])
    
    def _save_unsolved_tasks(self):
        """Save unsolved tasks"""
        data = {
            "count": len(self.unsolved_tasks),
            "last_updated": datetime.utcnow().isoformat(),
            "tasks": self.unsolved_tasks
        }
        
        with open(self.unsolved_tasks_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _find_task_by_hash(self, task_hash: str) -> Dict:
        """Find task by hash in current dataset"""
        current_dataset = self.load_current_dataset()
        
        for task in current_dataset:
            if self.hash_task(task) == task_hash:
                return task
        
        return None