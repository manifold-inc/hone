"""
Meta-Manager Module

Central orchestrator that coordinates all components:
- GPU Pool Manager
- Job Queue
- Intelligent Scheduler
- Executor

Handles:
- Job submission and lifecycle
- Background processing loop
- State machine transitions
- Error handling and retry logic
- Metrics collection
"""

import asyncio
import logging
import uuid
import subprocess
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import asdict

from job_queue import Job, JobQueue, JobStatus, WeightClass
from gpu_pool import GPUPoolManager
from scheduler import IntelligentScheduler
from executor import Executor
from config import Config

logger = logging.getLogger(__name__)


class MetaManager:
    """
    Central orchestrator for the sandbox runner.
    
    Responsibilities:
    1. Initialize and manage all subsystems
    2. Handle job submissions
    3. Run background processing loop
    4. Coordinate job state transitions
    5. Collect and report metrics
    
    Job Lifecycle:
    PENDING → CLONING → BUILDING → PREP → INFERENCE → COMPLETED/FAILED
    """
    
    def __init__(self, config: Config):
        """
        Initialize meta-manager with all subsystems.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Initialize subsystems
        self.gpu_pool = GPUPoolManager(config.hardware.gpu_count)
        self.job_queue = JobQueue()
        self.scheduler = IntelligentScheduler(
            self.gpu_pool,
            self.job_queue,
            starvation_threshold_seconds=3600  # 1 hour
        )
        
        # Initialize executor
        self.executor = Executor(config, self.gpu_pool)
        
        # Track running jobs
        self._running_jobs: Dict[str, Job] = {}
        
        # Background tasks
        self._processing_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self._total_jobs_submitted = 0
        self._total_jobs_completed = 0
        self._total_jobs_failed = 0
        
        logger.info("Meta-Manager initialized with executor")
    
    async def start(self):
        """
        Start the meta-manager and background processing loop.
        
        This should be called during application startup.
        """
        if self._running:
            logger.warning("Meta-Manager already running")
            return
        
        self._running = True
        
        # Start background processing loop
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        # Start GPU monitoring loop
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Meta-Manager started")
    
    async def stop(self):
        """
        Stop the meta-manager and cleanup resources.
        
        This should be called during application shutdown.
        """
        if not self._running:
            return
        
        logger.info("Stopping Meta-Manager...")
        
        self._running = False
        
        # Cancel background tasks
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running jobs
        for job_id in list(self._running_jobs.keys()):
            await self.cancel_job(job_id)
        
        logger.info("Meta-Manager stopped")
    
    async def submit_job(self, request: Dict) -> Dict:
        """
        Submit a new job for execution.
        
        Args:
            request: Job submission request with fields:
                - repo_url: Repository URL
                - repo_branch: Git branch (default: main)
                - repo_commit: Specific commit (optional)
                - weight_class: GPU weight class
                - input_data_s3_path: S3 input path
                - output_data_s3_path: S3 output path
                - priority: Job priority 0-10
                - validator_hotkey: Validator identifier
                - miner_hotkey: Miner identifier
                - custom_env_vars: Custom environment variables
        
        Returns:
            Job submission response with:
                - job_id: Unique job identifier
                - status: Initial status
                - queue_position: Position in queue
                - estimated_start_time: Estimated start time
        """
        # Generate unique job ID
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        # Create job object
        job = Job(
            job_id=job_id,
            repo_url=request.get("repo_url"),
            repo_branch=request.get("repo_branch", "main"),
            repo_commit=request.get("repo_commit"),
            weight_class=WeightClass(request.get("weight_class", "1xH200")),
            input_s3_path=request.get("input_data_s3_path", ""),
            output_s3_path=request.get("output_data_s3_path", ""),
            priority=request.get("priority", 0),
            validator_hotkey=request.get("validator_hotkey"),
            miner_hotkey=request.get("miner_hotkey", ""),
            custom_env_vars=request.get("custom_env_vars", {}),
            status=JobStatus.PENDING,
            submitted_at=datetime.utcnow()
        )
        
        # Enqueue job
        queue_position = await self.job_queue.enqueue(job)
        
        # Update statistics
        self._total_jobs_submitted += 1
        
        # Estimate start time (rough estimate)
        estimated_wait = await self.job_queue.estimate_wait_time(job_id)
        estimated_start = None
        if estimated_wait:
            estimated_start = datetime.utcnow() + estimated_wait
        
        logger.info(
            f"Job submitted: {job_id} "
            f"(weight_class={job.weight_class.value}, "
            f"priority={job.priority}, "
            f"position={queue_position})"
        )
        
        return {
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "queue_position": queue_position,
            "estimated_start_time": estimated_start.isoformat() if estimated_start else None
        }
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Get current status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary, or None if not found
        """
        # Check running jobs first
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
            return self._job_to_response(job)
        
        # Check executor's active jobs
        job = await self.executor.get_job(job_id)
        if job:
            return self._job_to_response(job)
        
        # Check queue
        job = await self.job_queue.get_job(job_id)
        if job:
            return self._job_to_response(job)
        
        # Job not found
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False if not found
        """
        # Try to cancel from queue first
        cancelled = await self.job_queue.cancel_job(job_id)
        if cancelled:
            logger.info(f"Cancelled queued job: {job_id}")
            return True
        
        # Try to cancel running job from executor
        cancelled = await self.executor.cancel_job(job_id)
        if cancelled:
            # Release GPUs
            await self.gpu_pool.release_gpus(job_id)
            
            # Remove from running jobs if tracked here
            if job_id in self._running_jobs:
                del self._running_jobs[job_id]
            
            logger.info(f"Cancelled running job: {job_id}")
            return True
        
        # Try to cancel from our tracking
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
            
            # Release GPUs
            await self.gpu_pool.release_gpus(job_id)
            
            # Update status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            
            # Remove from running jobs
            del self._running_jobs[job_id]
            
            logger.info(f"Cancelled tracked job: {job_id}")
            return True
        
        return False
    
    async def get_runner_status(self) -> Dict:
        """
        Get overall status of the sandbox runner.
        
        Returns:
            Dictionary with:
                - GPU status and utilization
                - Queue depth
                - Active jobs
                - Statistics
        """
        gpu_stats = await self.gpu_pool.get_allocation_stats()
        queue_stats = await self.job_queue.get_stats()
        
        # Get active jobs from executor
        active_jobs = self.executor.get_active_jobs()
        
        return {
            "runner_id": self.config.runner.id,
            "status": "operational" if self._running else "stopped",
            "gpu_stats": gpu_stats,
            "queue_stats": queue_stats,
            "active_jobs": len(active_jobs),
            "total_submitted": self._total_jobs_submitted,
            "total_completed": self._total_jobs_completed,
            "total_failed": self._total_jobs_failed,
            "execution_mode": self.config.execution.mode
        }
    
    async def _processing_loop(self):
        """
        Background processing loop that continuously schedules and executes jobs.
        
        Loop steps:
        1. Check for completed jobs
        2. Release GPUs from completed jobs
        3. Update metrics
        4. Check if we can schedule more jobs
        5. Schedule next job if GPUs available
        6. Execute job
        7. Sleep briefly
        """
        logger.info("Processing loop started")
        
        while self._running:
            try:
                # 1. Check for completed jobs
                await self._check_completed_jobs()
                
                # 2. Try to schedule more jobs
                scheduled = await self._schedule_next_job()
                
                if not scheduled:
                    # No job was scheduled, sleep longer
                    await asyncio.sleep(2.0)
                else:
                    # Job was scheduled, check again quickly
                    await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                logger.info("Processing loop cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in processing loop: {e}")
                await asyncio.sleep(5.0)  # Back off on error
        
        logger.info("Processing loop stopped")
    
    async def _monitoring_loop(self):
        """
        Background monitoring loop for GPU metrics.
        
        Updates GPU utilization metrics every 10 seconds.
        """
        logger.info("Monitoring loop started")
        
        while self._running:
            try:
                # Update GPU metrics
                await self._update_gpu_metrics()
                
                # Sleep for 10 seconds
                await asyncio.sleep(10.0)
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10.0)
        
        logger.info("Monitoring loop stopped")
    
    async def _check_completed_jobs(self):
        """
        Check for completed jobs and clean them up.
        """
        # Get active jobs from executor
        active_jobs = self.executor.get_active_jobs()
        
        completed_jobs = []
        
        for job_id, job in active_jobs.items():
            # Check if job is in terminal state
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, 
                            JobStatus.TIMEOUT, JobStatus.CANCELLED]:
                completed_jobs.append(job_id)
        
        # Clean up completed jobs
        for job_id in completed_jobs:
            job = active_jobs[job_id]
            
            # Release GPUs
            await self.gpu_pool.release_gpus(job_id)
            
            # Update statistics
            if job.status == JobStatus.COMPLETED:
                self._total_jobs_completed += 1
                logger.info(f"Job completed: {job_id}")
            else:
                self._total_jobs_failed += 1
                logger.info(f"Job failed: {job_id} (status={job.status.value})")
            
            # Remove from our tracking if present
            if job_id in self._running_jobs:
                del self._running_jobs[job_id]
    
    async def _schedule_next_job(self) -> bool:
        """
        Try to schedule the next job from the queue.
        
        Returns:
            True if a job was scheduled, False otherwise
        """
        # Get next job to schedule
        job = await self.scheduler.schedule_next()
        
        if not job:
            return False
        
        # Allocate GPUs
        gpus = await self.gpu_pool.allocate_gpus(job.weight_class, job.job_id)
        
        if not gpus:
            logger.warning(
                f"Failed to allocate GPUs for job {job.job_id} "
                f"despite scheduler selection"
            )
            return False
        
        # Remove job from queue
        dequeued_job = await self.job_queue.dequeue()
        if not dequeued_job or dequeued_job.job_id != job.job_id:
            logger.error(
                f"Queue mismatch: expected {job.job_id}, "
                f"got {dequeued_job.job_id if dequeued_job else None}"
            )
            # Release GPUs
            await self.gpu_pool.release_gpus(job.job_id)
            return False
        
        # Update job state
        job.assigned_gpus = gpus
        job.started_at = datetime.utcnow()
        
        # Track as running
        self._running_jobs[job.job_id] = job
        
        # Record for fairness tracking
        await self.scheduler.record_scheduled_job(job)
        
        # Start actual job execution in background and track the task
        task = asyncio.create_task(self.executor.execute_job(job))
        # Store task in executor's tracking for potential cancellation
        self.executor._job_tasks[job.job_id] = task
        
        logger.info(
            f"Scheduled job {job.job_id} on GPUs {gpus} "
            f"({job.weight_class.value})"
        )
        
        return True
    
    async def _update_gpu_metrics(self):
        """
        Update GPU utilization metrics using nvidia-smi.
        
        Queries GPU stats and updates the GPU pool manager.
        """
        try:
            # Run nvidia-smi to get GPU stats
            result = await asyncio.create_subprocess_exec(
                'nvidia-smi',
                '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.warning(f"nvidia-smi failed: {stderr.decode()}")
                return
            
            # Parse output
            lines = stdout.decode().strip().split('\n')
            
            for line in lines:
                if not line.strip():
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 5:
                    continue
                
                try:
                    gpu_id = int(parts[0])
                    utilization = float(parts[1])
                    memory_used = int(parts[2])
                    memory_total = int(parts[3])
                    temperature = float(parts[4])
                    
                    # Update GPU pool with metrics
                    await self.gpu_pool.update_gpu_utilization(
                        gpu_id=gpu_id,
                        utilization_percent=utilization,
                        memory_used_mb=memory_used,
                        temperature_celsius=temperature
                    )
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse nvidia-smi output: {line}: {e}")
                    continue
            
            logger.debug(f"Updated GPU metrics for {len(lines)} GPUs")
            
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, skipping GPU monitoring")
        except Exception as e:
            logger.warning(f"Error updating GPU metrics: {e}")
    
    def _job_to_response(self, job: Job) -> Dict:
        """
        Convert Job object to API response format.
        
        Args:
            job: Job object
            
        Returns:
            Dictionary with job information
        """
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "weight_class": job.weight_class.value,
            "submitted_at": job.submitted_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "current_phase": job.current_phase,
            "progress_percentage": job.progress_percentage,
            "assigned_gpus": job.assigned_gpus,
            "error_message": job.error_message,
            "input_s3_path": job.input_s3_path,
            "output_s3_path": job.output_s3_path,
            "validator_hotkey": job.validator_hotkey,
            "miner_hotkey": job.miner_hotkey,
            "priority": job.priority
        }