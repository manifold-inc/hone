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


# In synthetics/dataset_manager.py

class DatasetManager:
    """
    Manages daily dataset generation
    
    Strategy:
    1. Load unsolved tasks from yesterday
    2. Generate new synthetic tasks (never seen before)
    3. Ensure minimum of 100 total tasks
    4. Combine into today's dataset
    5. Block submissions during generation
    """
    
    def __init__(
        self,
        storage_dir: Path,
        num_unsolved_to_keep: int = 80,
        num_new_tasks: int = 20,
        min_total_tasks: int = 100,  # NEW: Minimum total tasks required
        generation_time: str = "00:00"  # UTC time
    ):
        self.storage = DatasetStorage(storage_dir)
        self.generator = ARC2Generator()
        
        self.num_unsolved_to_keep = num_unsolved_to_keep
        self.num_new_tasks = num_new_tasks
        self.min_total_tasks = min_total_tasks  # NEW
        self.generation_time = generation_time
        
        self.is_generating = False
        self.last_generation: datetime = None
        
        logger.info(f"Dataset Manager initialized (min tasks: {self.min_total_tasks})")
    
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
            logger.info(f"Target: Minimum {self.min_total_tasks} total tasks")
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
            
            num_unsolved_kept = len(tasks_to_keep)
            num_needed = max(
                self.min_total_tasks - num_unsolved_kept,
                self.num_new_tasks  
            )
            
            logger.info(f"Need to generate {num_needed} new tasks to reach {self.min_total_tasks} total")
            
            new_tasks = []
            attempts = 0
            max_attempts = num_needed * 3  
            
            logger.info(f"Generating {num_needed} new synthetic tasks...")
            
            generation_batch_size = 10  
            
            while len(new_tasks) < num_needed and attempts < max_attempts:
                attempts += 1
                
                try:
                    import random
                    difficulty_roll = random.random()
                    if difficulty_roll < 0.3:
                        chain_length = random.randint(3, 5)
                    else:
                        chain_length = random.randint(5, 7)
                    
                    problem = None
                    while not problem:
                        try:
                            problem = self.generator.generate_problem_set(
                                num_train=3,
                                num_test=1,
                                chain_length=chain_length,
                                preserves_size_only=False
                            )
                        except:
                            pass
                    
                    task_hash = self.storage.hash_task(problem)
                    
                    if not self.storage.is_task_seen(task_hash):
                        task_for_miners = self._prepare_task_for_miners(problem)
                        
                        new_tasks.append({
                            "task": task_for_miners,
                            "task_hash": task_hash,
                            "test_output": problem["test_output"],  
                            "metadata": {
                                **problem.get("metadata", {}),
                                "difficulty": self._classify_difficulty(chain_length)
                            }
                        })
                        
                        self.storage.mark_task_seen(task_hash)
                        
                        if len(new_tasks) % generation_batch_size == 0:
                            logger.info(f"Generated {len(new_tasks)}/{num_needed} new tasks")
                
                except Exception as e:
                    logger.warning(f"Failed to generate task (attempt {attempts}): {e}")
                    continue
            
            if len(new_tasks) < num_needed:
                logger.warning(
                    f"Only generated {len(new_tasks)}/{num_needed} unique tasks. "
                    "Generating additional tasks with relaxed constraints..."
                )
                
                while len(new_tasks) < num_needed:
                    try:
                        problem = None
                        while not problem:
                            try:
                                problem = self.generator.generate_problem_set(
                                    num_train=3,
                                    num_test=1,
                                    chain_length=random.randint(4, 7),
                                    preserves_size_only=False
                                )
                            except:
                                pass
                        
                        task_hash = self.storage.hash_task(problem)                        
                        task_for_miners = self._prepare_task_for_miners(problem)
                        
                        new_tasks.append({
                            "task": task_for_miners,
                            "task_hash": task_hash,
                            "test_output": problem["test_output"],
                            "metadata": {
                                **problem.get("metadata", {}),
                                "duplicate": self.storage.is_task_seen(task_hash),
                                "difficulty": "easy"
                            }
                        })
                        
                        if not self.storage.is_task_seen(task_hash):
                            self.storage.mark_task_seen(task_hash)
                        
                    except Exception as e:
                        logger.error(f"Critical: Failed to generate fallback task: {e}")
                        # As last resort, duplicate an existing task with modification
                        if new_tasks:
                            base_task = new_tasks[0].copy()
                            base_task["task_hash"] = f"{base_task['task_hash']}_dup_{len(new_tasks)}"
                            base_task["metadata"]["duplicate"] = True
                            new_tasks.append(base_task)
            
            logger.info(f"Successfully generated {len(new_tasks)} new tasks")
            
            # Combine all tasks for today's dataset
            todays_dataset = []
            
            # Add unsolved tasks
            for task in tasks_to_keep:
                task_hash = self.storage.hash_task(task)
                todays_dataset.append({
                    "task": self._prepare_task_for_miners(task),
                    "task_hash": task_hash,
                    "test_output": task.get("test_output"),
                    "metadata": task.get("metadata", {}),
                    "source": "unsolved"
                })
            
            # Add new tasks
            for item in new_tasks:
                todays_dataset.append({
                    **item,
                    "source": "synthetic"
                })
            
            # Final validation
            total_tasks = len(todays_dataset)
            if total_tasks < self.min_total_tasks:
                logger.error(
                    f"Failed to generate minimum required tasks! "
                    f"Have {total_tasks}, need {self.min_total_tasks}"
                )
                # Don't save incomplete dataset
                return False
            
            # Save the dataset
            self.storage.save_current_dataset(todays_dataset)
            
            self.last_generation = datetime.utcnow()
            
            # Log statistics
            difficulty_stats = self._calculate_difficulty_distribution(todays_dataset)
            
            logger.info("=" * 60)
            logger.info(f"DATASET GENERATION COMPLETE")
            logger.info(f"Total tasks: {total_tasks}")
            logger.info(f"  - Unsolved (carried over): {len(tasks_to_keep)}")
            logger.info(f"  - New synthetic: {len(new_tasks)}")
            logger.info(f"Difficulty distribution:")
            for difficulty, count in difficulty_stats.items():
                logger.info(f"  - {difficulty}: {count} tasks ({count*100/total_tasks:.1f}%)")
            logger.info("=" * 60)
            
            return True
        
        except Exception as e:
            logger.exception(f"Dataset generation failed: {e}")
            return False
        
        finally:
            self.is_generating = False
    
    def _classify_difficulty(self, chain_length: int) -> str:
        """Classify task difficulty based on chain length."""
        if chain_length <= 2:
            return "easy"
        elif chain_length <= 4:
            return "medium"
        else:
            return "hard"
    
    def _calculate_difficulty_distribution(self, dataset: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of task difficulties."""
        distribution = {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}
        
        for task in dataset:
            difficulty = task.get("metadata", {}).get("difficulty", "unknown")
            if difficulty in distribution:
                distribution[difficulty] += 1
            else:
                distribution["unknown"] += 1
        
        return distribution
    
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
