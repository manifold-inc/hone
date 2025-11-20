"""
Job Queue Module

Implements a priority-based job queue with:
- Priority levels (0-10, higher = more important)
- FIFO ordering within same priority
- Weight class grouping
- Thread-safe async operations
- Queue depth metrics
- Estimated wait time calculation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    CLONING = "cloning"
    BUILDING = "building"
    PREP = "prep"
    INFERENCE = "inference"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class WeightClass(str, Enum):
    """GPU weight class"""
    ONE_GPU = "1xH200"
    TWO_GPU = "2xH200"
    FOUR_GPU = "4xH200"
    EIGHT_GPU = "8xH200"
    
    def gpu_count(self) -> int:
        """Return number of GPUs required"""
        return {
            "1xH200": 1,
            "2xH200": 2,
            "4xH200": 4,
            "8xH200": 8,
        }[self.value]


@dataclass
class Job:
    """
    Job data model representing a miner submission
    
    Contains all information needed to execute a job:
    - Repository information
    - Resource requirements
    - Input/output paths
    - Priority and scheduling info
    - Runtime state
    """
    job_id: str
    
    repo_url: str
    repo_branch: str = "main"
    repo_commit: Optional[str] = None
    repo_path: str = ""
    
    weight_class: WeightClass = WeightClass.ONE_GPU
        
    priority: int = 0  # 0-10, higher = more important
    
    validator_hotkey: Optional[str] = None
    miner_hotkey: str = ""
    
    custom_env_vars: Dict[str, str] = field(default_factory=dict)
    
    status: JobStatus = JobStatus.PENDING
    assigned_gpus: Optional[List[int]] = None
    
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    error_message: Optional[str] = None
    retry_count: int = 0
    
    current_phase: Optional[str] = None
    progress_percentage: float = 0.0

    metrics: Optional[Dict] = None

    use_vllm: bool = False
    vllm_config: Optional[Dict[str, Any]] = None

class JobQueue:
    """
    Priority-based job queue with weight class awareness
    
    Features:
    - Jobs are ordered by priority (higher first), then FIFO within priority
    - Track queue depth per weight class for scheduling decisions
    - Thread-safe async operations with proper locking
    - Support for job cancellation
    - Estimated wait time calculation
    """
    
    def __init__(self):
        """Initialize empty job queue"""
        self._lock = asyncio.Lock()        
        self._queues: Dict[int, deque] = defaultdict(deque)
        self._jobs: Dict[str, Job] = {}
        self._positions: Dict[str, int] = {}
        self._weight_class_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("Job Queue initialized")
    
    async def enqueue(self, job: Job) -> int:
        """
        Add a job to the queue
        
        Jobs are queued based on priority, with FIFO ordering
        within the same priority level
        
        Args:
            job: Job to enqueue
            
        Returns:
            Queue position (0-indexed)
        """
        async with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"Job {job.job_id} already in queue")
            
            self._queues[job.priority].append(job)            
            self._jobs[job.job_id] = job
            self._weight_class_counts[job.weight_class.value] += 1            
            position = self._calculate_position(job)
            self._positions[job.job_id] = position
            
            logger.info(
                f"Enqueued job {job.job_id} "
                f"(priority={job.priority}, "
                f"weight_class={job.weight_class.value}, "
                f"position={position})"
            )
            
            return position
    
    async def dequeue(self) -> Optional[Job]:
        """
        Remove and return the highest priority job from the queue
        
        Returns:
            Next job to execute, or None if queue is empty
        """
        async with self._lock:
            for priority in sorted(self._queues.keys(), reverse=True):
                if self._queues[priority]:
                    job = self._queues[priority].popleft()
                    
                    if not self._queues[priority]:
                        del self._queues[priority]
                    
                    del self._jobs[job.job_id]
                    del self._positions[job.job_id]
                    self._weight_class_counts[job.weight_class.value] -= 1
                    
                    self._recalculate_positions()
                    
                    logger.info(
                        f"Dequeued job {job.job_id} "
                        f"(priority={job.priority}, "
                        f"weight_class={job.weight_class.value})"
                    )
                    
                    return job
            
            return None
    
    async def peek(self) -> Optional[Job]:
        """
        Get the next job without removing it from queue
        
        Returns:
            Next job that would be dequeued, or None if empty
        """
        async with self._lock:
            for priority in sorted(self._queues.keys(), reverse=True):
                if self._queues[priority]:
                    return self._queues[priority][0]
            return None
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID without removing it from queue
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job object, or None if not found
        """
        async with self._lock:
            return self._jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job and remove it from the queue
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False if not found
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.warning(f"Cannot cancel job {job_id}: not found")
                return False
            
            priority_queue = self._queues[job.priority]
            try:
                priority_queue.remove(job)
            except ValueError:
                logger.warning(f"Job {job_id} not in priority queue")
                return False
            
            if not self._queues[job.priority]:
                del self._queues[job.priority]
            
            del self._jobs[job_id]
            del self._positions[job_id]
            self._weight_class_counts[job.weight_class.value] -= 1
            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            
            self._recalculate_positions()
            
            logger.info(f"Cancelled job {job_id}")
            
            return True
    
    async def get_queue_depth(self) -> Dict[str, int]:
        """
        Get queue depth by weight class
        
        Returns:
            Dictionary mapping weight class to count of pending jobs
        """
        async with self._lock:
            return dict(self._weight_class_counts)
    
    async def get_total_depth(self) -> int:
        """
        Get total number of jobs in queue
        
        Returns:
            Total job count
        """
        async with self._lock:
            return len(self._jobs)
    
    async def get_queue_position(self, job_id: str) -> Optional[int]:
        """
        Get position of a job in the queue
        
        Args:
            job_id: Job identifier
            
        Returns:
            Queue position (0-indexed), or None if not found
        """
        async with self._lock:
            return self._positions.get(job_id)
    
    async def estimate_wait_time(
        self,
        job_id: str,
        avg_job_duration_seconds: float = 1800
    ) -> Optional[timedelta]:
        """
        Estimate how long a job will wait before starting
        
        This is a rough estimate based on:
        - Jobs ahead in queue
        - Average job duration
        - Does NOT account for GPU availability or weight classes
        
        Args:
            job_id: Job identifier
            avg_job_duration_seconds: Average job duration (default: 30 min)
            
        Returns:
            Estimated wait time, or None if job not found
        """
        position = await self.get_queue_position(job_id)
        if position is None:
            return None
        
        # this depends on GPU availability and job packing
        wait_seconds = position * avg_job_duration_seconds
        
        return timedelta(seconds=wait_seconds)
    
    async def get_jobs_by_priority(self, priority: int) -> List[Job]:
        """
        Get all jobs at a specific priority level
        
        Args:
            priority: Priority level (0-10)
            
        Returns:
            List of jobs at that priority
        """
        async with self._lock:
            if priority not in self._queues:
                return []
            return list(self._queues[priority])
    
    async def get_all_jobs(self) -> List[Job]:
        """
        Get all jobs in queue, ordered by priority and then FIFO
        
        Returns:
            List of all jobs in queue order
        """
        async with self._lock:
            jobs = []
            for priority in sorted(self._queues.keys(), reverse=True):
                jobs.extend(list(self._queues[priority]))
            return jobs
    
    async def get_stats(self) -> Dict:
        """
        Get queue statistics for monitoring
        
        Returns:
            Dictionary with queue statistics
        """
        async with self._lock:
            return {
                "total_jobs": len(self._jobs),
                "weight_class_counts": dict(self._weight_class_counts),
                "priority_distribution": {
                    priority: len(queue)
                    for priority, queue in self._queues.items()
                },
                "oldest_job_age_seconds": self._get_oldest_job_age()
            }
    
    def _calculate_position(self, job: Job) -> int:
        """
        Calculate queue position for a job
        
        Position is based on:
        1. All jobs with higher priority come first
        2. Within same priority, earlier submissions come first
        
        Args:
            job: Job to calculate position for
            
        Returns:
            Queue position (0-indexed)
        """
        position = 0
        
        for priority in sorted(self._queues.keys(), reverse=True):
            if priority > job.priority:
                position += len(self._queues[priority])
            elif priority == job.priority:
                for queued_job in self._queues[priority]:
                    if queued_job.job_id == job.job_id:
                        break
                    position += 1
                break
        
        return position
    
    def _recalculate_positions(self):
        """
        Recalculate queue positions for all jobs
        Called after removing jobs from queue to keep positions accurate
        """
        position = 0
        for priority in sorted(self._queues.keys(), reverse=True):
            for job in self._queues[priority]:
                self._positions[job.job_id] = position
                position += 1
    
    def _get_oldest_job_age(self) -> float:
        """
        Get age of oldest job in queue
        
        Returns:
            Age in seconds, or 0 if queue is empty
        """
        if not self._jobs:
            return 0.0
        
        oldest_job = min(
            self._jobs.values(),
            key=lambda j: j.submitted_at
        )
        
        age = (datetime.utcnow() - oldest_job.submitted_at).total_seconds()
        return age
    
    async def clear(self):
        """
        Clear all jobs from the queue
        
        WARNING: This removes all pending jobs without executing them!
        """
        async with self._lock:
            self._queues.clear()
            self._jobs.clear()
            self._positions.clear()
            self._weight_class_counts.clear()
            logger.warning("Queue cleared - all jobs removed")