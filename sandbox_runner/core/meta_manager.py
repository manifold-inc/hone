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
import json
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import asdict
from pathlib import Path

from core.job_queue import Job, JobQueue, JobStatus, WeightClass
from core.gpu_pool import GPUPoolManager
from core.scheduler import IntelligentScheduler
from core.executor import Executor
from synthetics.dataset_manager import DatasetManager
from config import Config


logger = logging.getLogger(__name__)


class MetaManager:
    """
    Central orchestrator for the sandbox runner
    
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
        Initialize meta-manager with all subsystems
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        self.gpu_pool = GPUPoolManager(config.hardware.gpu_count)
        self.job_queue = JobQueue()
        self.scheduler = IntelligentScheduler(
            self.gpu_pool,
            self.job_queue,
            starvation_threshold_seconds=3600  # 1 hour
        )
        
        self.executor = Executor(config, self.gpu_pool)        
        self._running_jobs: Dict[str, Job] = {}        
        self._processing_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        self._total_jobs_submitted = 0
        self._total_jobs_completed = 0
        self._total_jobs_failed = 0

        self._job_history: Dict[str, Job] = {}
        self._max_history_size = 1000

        self._lock = asyncio.Lock()
        self.active_jobs = self._running_jobs

        dataset_storage_dir = Path("/app/data/datasets")
        self.dataset_manager = DatasetManager(
            storage_dir=dataset_storage_dir,
            num_unsolved_to_keep=80,
            num_new_tasks=20,
            min_total_tasks=100,
            generation_time="00:00"  # UTC
        )
        
        self._dataset_generation_task: Optional[asyncio.Task] = None
        self._dataset_ready_event = asyncio.Event()
        self._dataset_ready_event.set()  

        logger.info("Meta-Manager initialized with executor")
    
    async def start(self):
        """
        Start the meta-manager and background processing loop        
        """
        if self._running:
            logger.warning("Meta-Manager already running")
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._processing_loop())        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._dataset_generation_task = asyncio.create_task(self._dataset_generation_loop())
        await self.executor.start()
        
        logger.info("Meta-Manager started")
    
    async def stop(self):
        """
        Stop the meta-manager and cleanup resources        
        """
        if not self._running:
            return
        
        logger.info("Stopping Meta-Manager...")
        
        self._running = False
        
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
        
        if self._dataset_generation_task:
            self._dataset_generation_task.cancel()
            try:
                await self._dataset_generation_task
            except asyncio.CancelledError:
                pass
        
        for job_id in list(self._running_jobs.keys()):
            await self.cancel_job(job_id)
        
        logger.info("Meta-Manager stopped")
    
    async def _dataset_generation_loop(self):
        """Background loop to check if dataset needs regeneration"""
        await asyncio.sleep(5)
        while self._running:
            try:
                if await self.dataset_manager.should_generate_today():
                    logger.info("=" * 60)
                    logger.info("STARTING DAILY DATASET GENERATION")
                    logger.info("All job submissions will be queued until generation completes")
                    logger.info("=" * 60)
                    
                    self._dataset_ready_event.clear()
                    
                    try:
                        success = await self.dataset_manager.generate_daily_dataset()
                        
                        if success:
                            logger.info("=" * 60)
                            logger.info("DAILY DATASET GENERATION COMPLETED")
                            logger.info("Processing queued job submissions...")
                            logger.info("=" * 60)
                        else:
                            logger.error("Daily dataset generation failed")
                    
                    finally:
                        self._dataset_ready_event.set()
                        logger.info("Job submission queue released")
                
                await asyncio.sleep(3600)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in dataset generation loop: {e}")
                self._dataset_ready_event.set()
                await asyncio.sleep(3600)

    async def get_job_with_results(self, job_id: str) -> Optional[Dict]:
        """Get job with full results including predictions"""
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
            return self._job_to_response(job)
        
        job = await self.executor.get_job(job_id)
        if job:
            return self._job_to_response(job)
        
        results_file = Path(f"/app/data/job_results/{job_id}.json")
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
    
        return None
    
    async def submit_job(self, request: Dict) -> Dict:
        """
        Submit a new job for execution
        
        Args:
            request: Job submission request with fields:
                - repo_url: Repository URL
                - repo_branch: Git branch (default: main)
                - repo_commit: Specific commit (optional)
                - weight_class: GPU weight class
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
        
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        job = Job(
            job_id=job_id,
            repo_url=request.get("repo_url"),
            repo_branch=request.get("repo_branch", "main"),
            repo_commit=request.get("repo_commit"),
            repo_path=request.get("repo_path", ""),
            weight_class=WeightClass(request.get("weight_class", "1xH200")),
            priority=request.get("priority", 0),
            validator_hotkey=request.get("validator_hotkey"),
            miner_hotkey=request.get("miner_hotkey", ""),
            custom_env_vars=request.get("custom_env_vars", {}),
            use_vllm=request.get("use_vllm", False),
            vllm_config=request.get("vllm_config"),
            status=JobStatus.PENDING,
            submitted_at=datetime.utcnow()
        )
        
        queue_position = await self.job_queue.enqueue(job)        
        self._total_jobs_submitted += 1
        
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
        Get current status of a job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary, or None if not found
        """
        job = None
        
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
            return self._job_to_response(job)
        
        job = await self.executor.get_job(job_id)
        if job:
            return self._job_to_response(job)
        
        job = await self.job_queue.get_job(job_id)
        if job:
            return self._job_to_response(job)
        
        if job_id in self._job_history:
            job = self._job_history[job_id]
            return self._job_to_response(job)
        
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False if not found
        """
        cancelled = await self.job_queue.cancel_job(job_id)
        if cancelled:
            logger.info(f"Cancelled queued job: {job_id}")
            return True
        
        cancelled = await self.executor.cancel_job(job_id)
        if cancelled:
            await self.gpu_pool.release_gpus(job_id)
            
            if job_id in self._running_jobs:
                del self._running_jobs[job_id]
            
            logger.info(f"Cancelled running job: {job_id}")
            return True
        
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
            
            await self.gpu_pool.release_gpus(job_id)            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            
            del self._running_jobs[job_id]
            
            logger.info(f"Cancelled tracked job: {job_id}")
            return True
        
        return False
    
    async def get_runner_status(self) -> Dict:
        """
        Get overall status of the sandbox runner
        
        Returns:
            Dictionary with:
                - GPU status and utilization
                - Queue depth
                - Active jobs
                - Statistics
        """
        gpu_stats = await self.gpu_pool.get_allocation_stats()
        queue_stats = await self.job_queue.get_stats()
        
        active_jobs = self.executor.get_active_jobs()

        queue_breakdown = await self.get_queue_breakdown()
        queue_by_weight = {
            weight_class: data["count"]
            for weight_class, data in queue_breakdown.items()
        }

        return {
            "runner_id": self.config.runner.id,
            "status": "operational" if self._running else "stopped",
            "gpu_stats": gpu_stats,
            "queue_stats": queue_stats,
            "active_jobs": len(active_jobs),
            "total_submitted": self._total_jobs_submitted,
            "total_completed": self._total_jobs_completed,
            "total_failed": self._total_jobs_failed,
            "execution_mode": self.config.execution.mode,
            "active_job_ids": list(active_jobs.keys()),
            "queue_stats": {
                "total_jobs": await self.job_queue.get_total_depth(),
                "by_weight_class": queue_by_weight
            }
        }
    
    async def _processing_loop(self):
        """
        Background processing loop that continuously schedules and executes jobs
        
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
                await self._check_completed_jobs()


                while self.dataset_manager.is_generating:
                    await asyncio.sleep(5.0)
                    continue
                
                scheduled = await self._schedule_next_job()
                
                if not scheduled:
                    await asyncio.sleep(2.0)
                else:
                    await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                logger.info("Processing loop cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in processing loop: {e}")
                await asyncio.sleep(5.0)
        
        logger.info("Processing loop stopped")
    
    async def _monitoring_loop(self):
        """
        Background monitoring loop for GPU metrics
        
        Updates GPU utilization metrics every 10 seconds
        """
        logger.info("Monitoring loop started")
        
        while self._running:
            try:
                await self._update_gpu_metrics()                
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
        Check for completed jobs and clean them up
        """
        active_jobs = self.executor.get_active_jobs()
        
        completed_jobs = []
        
        for job_id, job in active_jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, 
                            JobStatus.TIMEOUT, JobStatus.CANCELLED]:
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            job = active_jobs[job_id]
            
            await self.gpu_pool.release_gpus(job_id)
            
            self._store_job_in_history(job)
            
            if job.status == JobStatus.COMPLETED:
                self._total_jobs_completed += 1
                logger.info(f"Job completed: {job_id}")
            else:
                self._total_jobs_failed += 1
                logger.info(f"Job failed: {job_id} (status={job.status.value})")
            
            if job_id in self._running_jobs:
                del self._running_jobs[job_id]

    def _store_job_in_history(self, job: Job):
        """
        Store completed job in history for later retrieval
        
        Args:
            job: Completed job to store
        """
        self._job_history[job.job_id] = job
        
        if len(self._job_history) > self._max_history_size:
            sorted_jobs = sorted(
                self._job_history.items(),
                key=lambda x: x[1].completed_at or x[1].submitted_at
            )
            
            jobs_to_remove = sorted_jobs[:-self._max_history_size]
            for job_id, _ in jobs_to_remove:
                del self._job_history[job_id]
            
            logger.debug(f"Trimmed job history: removed {len(jobs_to_remove)} old jobs")
    
    async def _schedule_next_job(self) -> bool:
        """
        Try to schedule the next job from the queue
        
        Returns:
            True if a job was scheduled, False otherwise
        """
        job = await self.scheduler.schedule_next()
        
        if not job:
            return False
        
        gpus = await self.gpu_pool.allocate_gpus(job.weight_class, job.job_id)
        
        if not gpus:
            logger.warning(
                f"Failed to allocate GPUs for job {job.job_id} "
                f"despite scheduler selection"
            )
            return False
        
        dequeued_job = await self.job_queue.dequeue()
        if not dequeued_job or dequeued_job.job_id != job.job_id:
            logger.error(
                f"Queue mismatch: expected {job.job_id}, "
                f"got {dequeued_job.job_id if dequeued_job else None}"
            )
            await self.gpu_pool.release_gpus(job.job_id)
            return False
        
        job.assigned_gpus = gpus
        job.started_at = datetime.utcnow()
        
        self._running_jobs[job.job_id] = job
        
        await self.scheduler.record_scheduled_job(job)
        
        task = asyncio.create_task(self.executor.execute_job(job))
        self.executor._job_tasks[job.job_id] = task
        
        logger.info(
            f"Scheduled job {job.job_id} on GPUs {gpus} "
            f"({job.weight_class.value})"
        )
        
        return True
    
    async def _update_gpu_metrics(self):
        """
        Update GPU utilization metrics using nvidia-smi
        Queries GPU stats and updates the GPU pool manager
        """
        try:
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
        Convert Job object to API response format
        
        Args:
            job: Job object
            
        Returns:
            Dictionary with job information
        """
        response = {
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
            "validator_hotkey": job.validator_hotkey,
            "miner_hotkey": job.miner_hotkey,
            "priority": job.priority
        }
        
        if hasattr(job, 'metrics') and job.metrics:
            response["has_metrics"] = True
            response["metrics_summary"] = job.metrics.get("aggregate", {})
        else:
            response["has_metrics"] = False
        
        return response
    
    async def get_gpu_status(self) -> Dict[int, Dict]:
        """Get detailed status of all GPUs."""
        gpu_status_dict = await self.gpu_pool.get_gpu_status()
        
        result = {}
        for gpu_id, gpu_info in gpu_status_dict.items():
            result[gpu_id] = {
                "gpu_id": gpu_info.gpu_id,
                "status": gpu_info.status.value,
                "allocated_to_job": gpu_info.allocated_to_job,
                "utilization_percent": gpu_info.utilization_percent,
                "memory_used_mb": gpu_info.memory_used_mb,
                "memory_total_mb": gpu_info.memory_total_mb,
                "temperature_celsius": gpu_info.temperature_celsius,
                "last_updated": gpu_info.last_updated.isoformat()
            }
        
        return result
    
    async def get_queue_breakdown(self) -> Dict[str, Dict]:
        """Get queue breakdown by weight class"""
        from core.job_queue import WeightClass
        
        result = {}
        all_jobs = await self.job_queue.get_all_jobs()
        
        for weight_class in WeightClass:
            class_jobs = [
                job for job in all_jobs 
                if job.weight_class == weight_class
            ]
            
            result[weight_class.value] = {
                "count": len(class_jobs),
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "priority": job.priority,
                        "submitted_at": job.submitted_at.isoformat(),
                        "miner_hotkey": job.miner_hotkey
                    }
                    for job in class_jobs
                ]
            }
        
        return result
    
    async def get_active_jobs(self) -> List[Dict]:
        """Get list of active jobs"""
        active_jobs = []
        
        for job_id, job in self._running_jobs.items():
            progress = self._calculate_job_progress(job)
            
            active_jobs.append({
                "job_id": job.job_id,
                "status": job.status.value,
                "weight_class": job.weight_class.value,
                "miner_hotkey": job.miner_hotkey,
                "validator_hotkey": job.validator_hotkey,
                "priority": job.priority,
                "progress_percentage": progress,
                "current_phase": job.current_phase,
                "assigned_gpus": job.assigned_gpus or [],
                "started_at": job.started_at.isoformat() if job.started_at else None
            })

        return active_jobs
    
    async def get_job_logs(self, job_id: str, lines: int = 100, offset: int = 0) -> Optional[Dict]:
        """Get job logs"""
        job = None
        
        # Check active jobs first
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
        elif job_id in self._job_history:
            job = self._job_history[job_id]
        
        if not job:
            return None
        
        # Try to get logs from executor (for running jobs)
        executor_job = await self.executor.get_job(job_id)
        if executor_job:
            # For running jobs, try to get container logs
            try:
                import docker
                client = docker.from_env()
                
                # Try to find container by name pattern
                container_name = f"sandbox-{job_id}"
                try:
                    containers = client.containers.list(
                        filters={"name": container_name}
                    )
                    
                    if containers:
                        container = containers[0]
                        log_output = container.logs(
                            stdout=True,
                            stderr=True,
                            tail=lines
                        ).decode('utf-8', errors='replace')
                        
                        # Parse into structured format
                        log_lines = log_output.split('\n')
                        logs = []
                        for line in log_lines:
                            if line.strip():
                                logs.append({
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "level": "INFO",
                                    "message": line.strip()
                                })
                        
                        return {
                            "logs": logs[offset:offset+lines],
                            "total_lines": len(logs),
                            "has_more": (offset + lines) < len(logs)
                        }
                except Exception as e:
                    logger.debug(f"Could not fetch container logs: {e}")
            except Exception as e:
                logger.debug(f"Docker not available for log fetching: {e}")
        
        logs = []
        
        return {
            "logs": logs[offset:offset+lines],
            "total_lines": len(logs),
            "has_more": (offset + lines) < len(logs)
        }
    
    def _calculate_job_progress(self, job) -> float:
        """Calculate job progress percentage"""
        from core.job_queue import JobStatus
        
        phase_progress = {
            JobStatus.PENDING: 0,
            JobStatus.CLONING: 5,
            JobStatus.BUILDING: 20,
            JobStatus.PREP: 45,
            JobStatus.INFERENCE: 80,
            JobStatus.COMPLETED: 100
        }
        
        return phase_progress.get(job.status, 0)
    
    def _get_mock_logs(self, job) -> List[Dict]:
        """Generate mock logs"""
        from datetime import timedelta
        
        logs = [
            {
                "timestamp": job.submitted_at.isoformat(),
                "level": "INFO",
                "message": f"Job {job.job_id} submitted"
            }
        ]
        
        if job.started_at:
            logs.extend([
                {
                    "timestamp": job.started_at.isoformat(),
                    "level": "INFO",
                    "message": f"Job started on GPUs {job.assigned_gpus}"
                },
                {
                    "timestamp": (job.started_at + timedelta(seconds=5)).isoformat(),
                    "level": "INFO",
                    "message": f"Cloning repository: {job.repo_url}"
                },
                {
                    "timestamp": (job.started_at + timedelta(seconds=30)).isoformat(),
                    "level": "INFO",
                    "message": "Building Docker image..."
                }
            ])
        
        return logs
    
    async def get_job_metrics(self, job_id: str) -> Optional[Dict]:
        """
        Get calculated metrics for a completed job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Metrics dictionary or None if not available
        """        
        job = None
        
        job = await self.executor.get_job(job_id)
        
        if not job and job_id in self._running_jobs:
            job = self._running_jobs[job_id]
        
        if not job and job_id in self._job_history:
            job = self._job_history[job_id]
        
        if not job:
            logger.warning(f"Job not found: {job_id}")
            return None
        
        if job.status.value not in ["completed", "failed", "timeout"]:
            logger.warning(
                f"Attempted to get metrics for non-completed job: {job_id} "
                f"(status: {job.status.value})"
            )
            return None
        
        if hasattr(job, 'metrics') and job.metrics:
            logger.info(f"Retrieved metrics for job: {job_id}")
            return job.metrics
        
        if job.status.value == "completed":
            logger.warning(
                f"Job {job_id} is completed but has no metrics. "
                "This may be a prep-only job or metrics calculation failed."
            )
            return {
                "status": "no_metrics",
                "message": "Job completed but no metrics available",
                "job_status": job.status.value
            }
        else:
            return {
                "status": "job_failed",
                "message": f"Job did not complete successfully: {job.status.value}",
                "job_status": job.status.value,
                "error_message": job.error_message
            }