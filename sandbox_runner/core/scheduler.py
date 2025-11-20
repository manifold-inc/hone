"""
Intelligent Scheduler Module

Optimizes parallel GPU execution by:
- Matching jobs to available GPU capacity
- Maximizing GPU utilization through efficient packing
- Fair scheduling across validators
- Starvation prevention for large jobs
- Weight class compatibility checking
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from collections import defaultdict

from core.job_queue import Job, JobQueue, WeightClass
from core.gpu_pool import GPUPoolManager

logger = logging.getLogger(__name__)


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision"""
    job: Optional[Job]
    reason: str
    alternative_jobs: List[Job] = None


class IntelligentScheduler:
    """
    Intelligent job scheduler that optimizes GPU utilization
    
    Scheduling Strategy:
    1. Maximize GPU utilization - pack jobs efficiently
    2. Prefer multiple small jobs over waiting for large job
    3. Prevent starvation of large jobs (8x weight class)
    4. Fair scheduling across validators
    5. Respect priority levels
    
    Example Scenarios:
    - 8 GPUs free, queue: [2x 1xH200, 1x 4xH200, 1x 2xH200]
      → Run both 1x jobs + one 4x job simultaneously (6 GPUs used)
    
    - 8 GPUs free, queue: [1x 8xH200, 4x 1xH200]
      → If 8x job has been waiting long, run it despite lower utilization
      → Otherwise run all 4x 1xH200 jobs for better throughput
    """
    
    def __init__(
        self,
        gpu_pool: GPUPoolManager,
        job_queue: JobQueue,
        starvation_threshold_seconds: int = 3600  # 1 hour
    ):
        """
        Initialize intelligent scheduler
        
        Args:
            gpu_pool: GPU pool manager
            job_queue: Job queue
            starvation_threshold_seconds: Time before large jobs get priority
        """
        self.gpu_pool = gpu_pool
        self.job_queue = job_queue
        self.starvation_threshold = timedelta(seconds=starvation_threshold_seconds)
        
        self._validator_last_scheduled: Dict[str, datetime] = {}
        self._validator_job_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("Intelligent Scheduler initialized")
    
    async def schedule_next(self) -> Optional[Job]:
        """
        Select the best job to run next based on scheduling policy
        
        Algorithm:
        1. Get available GPU count
        2. Check for starving large jobs (waiting > threshold)
        3. Try to find optimal job packing
        4. Apply fairness constraints
        5. Return best job or None
        
        Returns:
            Job to execute next, or None if no suitable job found
        """
        available_gpus = await self.gpu_pool.get_available_gpu_count()
        
        if available_gpus == 0:
            logger.debug("No GPUs available for scheduling")
            return None
                
        jobs = await self.job_queue.get_all_jobs()
        
        if not jobs:
            logger.debug("No jobs in queue")
            return None
        
        starving_job = await self._check_for_starvation(jobs, available_gpus)
        if starving_job:
            logger.info(
                f"Scheduling starving job {starving_job.job_id} "
                f"({starving_job.weight_class.value})"
            )
            return starving_job
        
        best_job = await self._find_optimal_job(jobs, available_gpus)
        
        if best_job:
            logger.info(
                f"Scheduling job {best_job.job_id} "
                f"({best_job.weight_class.value}, priority={best_job.priority})"
            )
            return best_job
        
        logger.debug("No suitable job found for current GPU availability")
        return None
    
    async def can_schedule(self, job: Job) -> bool:
        """
        Check if a job can be scheduled with current GPU availability
        
        Args:
            job: Job to check
            
        Returns:
            True if job can be scheduled, False otherwise
        """
        return await self.gpu_pool.can_allocate(job.weight_class)
    
    async def optimize_allocation(self) -> List[Job]:
        """
        Analyze current allocation and suggest improvements
        
        This can identify opportunities to:
        - Repack running jobs for better utilization
        - Preempt low-priority jobs for high-priority ones
                
        Returns:
            List of jobs that could be scheduled for better utilization
        """
        available_gpus = await self.gpu_pool.get_available_gpu_count()
        jobs = await self.job_queue.get_all_jobs()
        
        schedulable_jobs = []
        for job in jobs:
            if job.weight_class.gpu_count() <= available_gpus:
                schedulable_jobs.append(job)
        
        return schedulable_jobs
    
    async def get_scheduling_stats(self) -> Dict:
        """
        Get scheduling statistics for monitoring
        
        Returns:
            Dictionary with scheduling metrics
        """
        return {
            "validator_job_counts": dict(self._validator_job_counts),
            "validators_scheduled": len(self._validator_last_scheduled),
            "starvation_threshold_seconds": self.starvation_threshold.total_seconds()
        }
    
    async def _check_for_starvation(
        self,
        jobs: List[Job],
        available_gpus: int
    ) -> Optional[Job]:
        """
        Check if any large jobs are starving and should be prioritized
        
        A job is considered starving if:
        1. It requires many GPUs (4x or 8x)
        2. It has been waiting longer than starvation_threshold
        3. Enough GPUs are available
        
        Args:
            jobs: List of pending jobs
            available_gpus: Number of available GPUs
            
        Returns:
            Starving job to prioritize, or None
        """
        now = datetime.utcnow()
        
        for job in jobs:
            # check large jobs (4x, 8x)
            if job.weight_class not in [WeightClass.FOUR_GPU, WeightClass.EIGHT_GPU]:
                continue
            
            wait_time = now - job.submitted_at
            if wait_time < self.starvation_threshold:
                continue
            
            # if we have enough GPUs
            required_gpus = job.weight_class.gpu_count()
            if available_gpus >= required_gpus:
                logger.warning(
                    f"Job {job.job_id} is starving "
                    f"(waited {wait_time.total_seconds():.0f}s, "
                    f"threshold={self.starvation_threshold.total_seconds()}s)"
                )
                return job
        
        return None
    
    async def _find_optimal_job(
        self,
        jobs: List[Job],
        available_gpus: int
    ) -> Optional[Job]:
        """
        Find the job that provides optimal GPU utilization
        
        Strategy:
        1. Prefer jobs that maximize GPU usage without waste
        2. Among equal utilization, prefer higher priority
        3. Among equal priority, prefer FIFO (already sorted)
        4. Consider validator fairness
        
        Args:
            jobs: List of pending jobs (sorted by priority/FIFO)
            available_gpus: Number of available GPUs
            
        Returns:
            Best job to schedule, or None
        """
        best_job = None
        best_score = -1
        
        for job in jobs:
            required_gpus = job.weight_class.gpu_count()
            
            if required_gpus > available_gpus:
                continue
            
            score = self._calculate_job_score(
                job,
                required_gpus,
                available_gpus
            )
            
            if score > best_score:
                best_score = score
                best_job = job
        
        return best_job
    
    def _calculate_job_score(
        self,
        job: Job,
        required_gpus: int,
        available_gpus: int
    ) -> float:
        """
        Calculate a scheduling score for a job
                
        Factors:
        - GPU utilization: Higher is better
        - Priority: Higher priority gets bonus
        - Wait time: Longer wait gets bonus
        - Validator fairness: Less recent validators get bonus
        
        Args:
            job: Job to score
            required_gpus: GPUs required by job
            available_gpus: GPUs currently available
            
        Returns:
            Scheduling score (higher is better)
        """
        utilization_score = (required_gpus / available_gpus) * 100
        priority_bonus = job.priority
        wait_hours = (datetime.utcnow() - job.submitted_at).total_seconds() / 3600
        wait_bonus = min(wait_hours, 5)        
        fairness_bonus = self._calculate_fairness_bonus(job.validator_hotkey)
        
        score = utilization_score + priority_bonus + wait_bonus + fairness_bonus
        
        return score
    
    def _calculate_fairness_bonus(self, validator_hotkey: Optional[str]) -> float:
        """
        Calculate fairness bonus for a validator
        
        Validators who haven't had jobs scheduled recently get a bonus
        
        Args:
            validator_hotkey: Validator identifier
            
        Returns:
            Fairness bonus (0-10 points)
        """
        if not validator_hotkey:
            return 0
        
        if validator_hotkey not in self._validator_last_scheduled:
            return 10
        
        last_scheduled = self._validator_last_scheduled[validator_hotkey]
        time_since = (datetime.utcnow() - last_scheduled).total_seconds()
        
        bonus = min(time_since / 600, 10)
        
        return bonus
    
    async def record_scheduled_job(self, job: Job):
        """
        Record that a job was scheduled for fairness tracking
        
        Args:
            job: Job that was scheduled
        """
        if job.validator_hotkey:
            self._validator_last_scheduled[job.validator_hotkey] = datetime.utcnow()
            self._validator_job_counts[job.validator_hotkey] += 1
    
    async def get_scheduling_recommendation(self) -> Dict:
        """
        Get a recommendation for current scheduling state
        
        Returns:
            Dictionary with scheduling recommendations
        """
        available_gpus = await self.gpu_pool.get_available_gpu_count()
        jobs = await self.job_queue.get_all_jobs()
        
        if not jobs:
            return {
                "recommendation": "no_jobs",
                "message": "No jobs in queue"
            }
        
        if available_gpus == 0:
            return {
                "recommendation": "no_gpus",
                "message": "No GPUs available"
            }
        
        next_job = await self.schedule_next()
        
        if next_job:
            return {
                "recommendation": "schedule",
                "job_id": next_job.job_id,
                "weight_class": next_job.weight_class.value,
                "message": f"Can schedule job {next_job.job_id}"
            }
        
        min_gpu_requirement = min(job.weight_class.gpu_count() for job in jobs)
        
        if available_gpus < min_gpu_requirement:
            return {
                "recommendation": "wait_for_gpus",
                "message": f"Need {min_gpu_requirement} GPUs, have {available_gpus}",
                "gpus_needed": min_gpu_requirement,
                "gpus_available": available_gpus
            }
        
        return {
            "recommendation": "unclear",
            "message": "Cannot determine scheduling action"
        }