"""
Manages daily dataset generation with adaptive difficulty
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta

from synthetics.arc_agi2_generator import ARC2Generator
from storage.dataset_storage import DatasetStorage

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages daily dataset generation
    
    Strategy:
    1. Load unsolved tasks from yesterday
    2. Generate new synthetic tasks (never seen before)
    3. Combine into today's dataset
    4. Block submissions during generation
    """
    
    def __init__(
        self,
        storage_dir: Path,
        num_unsolved_to_keep: int = 50,
        num_new_tasks: int = 50,
        generation_time: str = "00:00"  # UTC time
    ):
        self.storage = DatasetStorage(storage_dir)
        self.generator = ARC2Generator()
        
        self.num_unsolved_to_keep = num_unsolved_to_keep
        self.num_new_tasks = num_new_tasks
        self.generation_time = generation_time
        
        self.is_generating = False
        self.last_generation: datetime = None
        
        logger.info("Dataset Manager initialized")
    
    async def should_generate_today(self) -> bool:
        """Check if we need to generate dataset today"""
        if self.is_generating:
            return False
        
        if not self.last_generation:
            return True
        
        now = datetime.utcnow()
        last_gen_date = self.last_generation.date()
        
        if now.date() > last_gen_date:
            hour, minute = map(int, self.generation_time.split(":"))
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if now >= target_time:
                return True
        
        return False
    
    async def generate_daily_dataset(self) -> bool:
        """
        Generate today's dataset
        
        Returns:
            True if generation successful
        """
        if self.is_generating:
            logger.warning("Dataset generation already in progress")
            return False
        
        self.is_generating = True
        
        try:
            logger.info("=" * 60)
            logger.info("STARTING DAILY DATASET GENERATION")
            logger.info("=" * 60)
            
            unsolved = self.storage.get_unsolved_tasks()
            logger.info(f"Found {len(unsolved)} unsolved tasks from yesterday")
            
            unsolved_sorted = sorted(
                self.storage.unsolved_tasks,
                key=lambda x: x.get("attempts", 0),
                reverse=True
            )
            tasks_to_keep = [
                item["task"] for item in unsolved_sorted[:self.num_unsolved_to_keep]
            ]
            
            logger.info(f"Keeping {len(tasks_to_keep)} hardest unsolved tasks")
            
            new_tasks = []
            attempts = 0
            max_attempts = self.num_new_tasks * 3 
            
            logger.info(f"Generating {self.num_new_tasks} new synthetic tasks...")
            
            while len(new_tasks) < self.num_new_tasks and attempts < max_attempts:
                attempts += 1
                
                try:
                    problem = self.generator.generate_problem_set(
                        num_train=3,
                        num_test=1,
                        chain_length=3,
                        preserves_size_only=False
                    )
                    
                    task_hash = self.storage.hash_task(problem)
                    
                    if not self.storage.is_task_seen(task_hash):
                        task_for_miners = self._prepare_task_for_miners(problem)
                        
                        new_tasks.append({
                            "task": task_for_miners,
                            "task_hash": task_hash,
                            "test_output": problem["test_output"],  # Store separately
                            "metadata": problem.get("metadata", {})
                        })
                        
                        self.storage.mark_task_seen(task_hash)
                        
                        if len(new_tasks) % 10 == 0:
                            logger.info(f"Generated {len(new_tasks)}/{self.num_new_tasks} new tasks")
                
                except Exception as e:
                    logger.warning(f"Failed to generate task: {e}")
                    continue
            
            logger.info(f"Successfully generated {len(new_tasks)} new tasks")
            
            todays_dataset = []
            
            for task in tasks_to_keep:
                task_hash = self.storage.hash_task(task)
                todays_dataset.append({
                    "task": self._prepare_task_for_miners(task),
                    "task_hash": task_hash,
                    "test_output": task.get("test_output"),
                    "metadata": task.get("metadata", {}),
                    "source": "unsolved"
                })
            
            for item in new_tasks:
                todays_dataset.append({
                    **item,
                    "source": "synthetic"
                })
            
            self.storage.save_current_dataset(todays_dataset)
            
            self.last_generation = datetime.utcnow()
            
            logger.info("=" * 60)
            logger.info(f"DATASET GENERATION COMPLETE")
            logger.info(f"Total tasks: {len(todays_dataset)}")
            logger.info(f"  - Unsolved: {len(tasks_to_keep)}")
            logger.info(f"  - New: {len(new_tasks)}")
            logger.info("=" * 60)
            
            return True
        
        except Exception as e:
            logger.exception(f"Dataset generation failed: {e}")
            return False
        
        finally:
            self.is_generating = False
    
    def _prepare_task_for_miners(self, problem: Dict) -> Dict:
        """
        Prepare task for miners by removing test output
        
        Miners should only see:
        - train_examples (with inputs and outputs)
        - test_input (without output)
        """
        return {
            "train_examples": problem.get("train_examples", []),
            "test_input": problem.get("test_input"),
        }
    
    def get_current_dataset(self) -> List[Dict]:
        """Get current active dataset for miners"""
        dataset = self.storage.load_current_dataset()
        
        return [
            {
                "task_hash": item["task_hash"],
                "task": item["task"],
                "metadata": item.get("metadata", {})
            }
            for item in dataset
        ]
    
    def validate_miner_predictions(
        self,
        predictions: List[Dict]
    ) -> List[Dict]:
        """
        Validate miner predictions against test outputs
        
        Args:
            predictions: List of {task_hash, predicted_output}
        
        Returns:
            List of {task_hash, solved, metrics}
        """
        dataset = self.storage.load_current_dataset()
        
        # Build lookup
        task_map = {item["task_hash"]: item for item in dataset}
        
        results = []
        
        for pred in predictions:
            task_hash = pred.get("task_hash")
            predicted = pred.get("predicted_output")
            
            if task_hash not in task_map:
                logger.warning(f"Unknown task hash: {task_hash}")
                continue
            
            task_data = task_map[task_hash]
            expected = task_data.get("test_output")
            
            # Check if solved
            solved = (predicted == expected)
            
            # Calculate metrics
            from utils.metrics import calculate_metrics_for_prediction
            metrics = calculate_metrics_for_prediction(predicted, expected)
            
            results.append({
                "task_hash": task_hash,
                "solved": solved,
                "metrics": metrics,
                "attempts": 1
            })
        
        return results