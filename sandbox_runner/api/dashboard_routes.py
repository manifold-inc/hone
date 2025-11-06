"""
Dashboard API Routes Extension

Additional endpoints to support the real-time monitoring dashboard:
- /v1/dashboard/gpus - Detailed GPU status
- /v1/dashboard/queue - Queue breakdown by weight class
- /v1/dashboard/jobs/active - List of active jobs with details
- /v1/dashboard/jobs/{job_id}/logs - Job logs
"""

from typing import List, Dict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
import logging

from config import Config
from api.auth import AuthenticationManager, RateLimiter

logger = logging.getLogger("api.dashboard")


# ============================================================================
# Response Models
# ============================================================================

class GPUStatusDetail(BaseModel):
    """Detailed GPU status for dashboard."""
    gpu_id: int
    status: str  # free, allocated, error, offline
    allocated_to_job: str = None
    utilization_percent: float = 0.0
    memory_used_mb: int = 0
    memory_total_mb: int = 81920  # H200 = 80GB
    temperature_celsius: float = 0.0
    last_updated: datetime


class QueueBreakdown(BaseModel):
    """Queue breakdown by weight class."""
    weight_class: str
    count: int
    jobs: List[Dict] = []


class ActiveJobDetail(BaseModel):
    """Detailed active job information."""
    job_id: str
    status: str
    weight_class: str
    miner_hotkey: str
    validator_hotkey: str = None
    priority: int
    progress_percentage: float = 0.0
    current_phase: str = None
    assigned_gpus: List[int] = []
    started_at: datetime
    estimated_completion: datetime = None


class JobLogs(BaseModel):
    """Job execution logs."""
    job_id: str
    logs: List[Dict]  # List of log entries with timestamp and message
    total_lines: int
    has_more: bool = False


class DashboardSummary(BaseModel):
    """Complete dashboard summary."""
    timestamp: datetime
    runner_id: str
    execution_mode: str
    
    # GPU stats
    total_gpus: int
    free_gpus: int
    allocated_gpus: int
    avg_gpu_utilization: float
    avg_gpu_temperature: float
    
    # Job stats
    active_jobs: int
    queued_jobs: int
    total_submitted: int
    total_completed: int
    total_failed: int
    success_rate: float
    
    # Queue breakdown
    queue_by_weight: Dict[str, int]
    
    # Estimated wait times
    estimated_queue_time_seconds: float = 0


# ============================================================================
# Dashboard Router
# ============================================================================

def create_dashboard_router(config: Config) -> APIRouter:
    """Create dashboard API router."""
    router = APIRouter(tags=["dashboard"])
    
    # Import dependencies
    from api.routes import (
        authenticate_request,
        get_meta_manager,
        get_config
    )
    
    @router.get(
        "/dashboard/summary",
        response_model=DashboardSummary,
        summary="Get complete dashboard summary"
    )
    async def get_dashboard_summary(
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager),
        config: Config = Depends(get_config)
    ):
        """
        Get comprehensive dashboard summary with all key metrics.
        This is the main endpoint for dashboard refresh.
        """
        logger.info("Dashboard summary requested")
        
        # Get runner status
        runner_status = await meta_manager.get_runner_status()
        
        # Get GPU stats
        gpu_stats = runner_status.get("gpu_stats", {})
        
        # Get queue stats
        queue_stats = runner_status.get("queue_stats", {})
        
        # Calculate derived metrics
        total_finished = (
            runner_status.get("total_completed", 0) + 
            runner_status.get("total_failed", 0)
        )
        success_rate = 0.0
        if total_finished > 0:
            success_rate = (
                runner_status.get("total_completed", 0) / total_finished * 100
            )
        
        # Estimate queue time (simple calculation)
        active_jobs = runner_status.get("active_jobs", 0)
        queued_jobs = queue_stats.get("total_jobs", 0)
        avg_job_duration = 1800  # 30 minutes default
        
        estimated_queue_time = 0
        if active_jobs > 0 and queued_jobs > 0:
            estimated_queue_time = (queued_jobs / active_jobs) * avg_job_duration
        
        return DashboardSummary(
            timestamp=datetime.utcnow(),
            runner_id=runner_status.get("runner_id", "unknown"),
            execution_mode=runner_status.get("execution_mode", "unknown"),
            total_gpus=gpu_stats.get("total_gpus", 0),
            free_gpus=gpu_stats.get("free_gpus", 0),
            allocated_gpus=gpu_stats.get("allocated_gpus", 0),
            avg_gpu_utilization=0.0,  # TODO: Calculate from GPU pool
            avg_gpu_temperature=0.0,  # TODO: Calculate from GPU pool
            active_jobs=runner_status.get("active_jobs", 0),
            queued_jobs=queued_jobs,
            total_submitted=runner_status.get("total_submitted", 0),
            total_completed=runner_status.get("total_completed", 0),
            total_failed=runner_status.get("total_failed", 0),
            success_rate=success_rate,
            queue_by_weight=queue_stats.get("by_weight_class", {}),
            estimated_queue_time_seconds=estimated_queue_time
        )
    
    @router.get(
        "/dashboard/gpus",
        response_model=List[GPUStatusDetail],
        summary="Get detailed GPU status"
    )
    async def get_gpu_details(
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager)
    ):
        """Get detailed status of all GPUs."""
        logger.info("GPU details requested")
        
        # Get GPU status from meta-manager
        gpu_status = await meta_manager.get_gpu_status()
        
        result = []
        for gpu_id, gpu_info in gpu_status.items():
            result.append(GPUStatusDetail(
                gpu_id=gpu_id,
                status=gpu_info.get("status", "unknown"),
                allocated_to_job=gpu_info.get("allocated_to_job"),
                utilization_percent=gpu_info.get("utilization_percent", 0.0),
                memory_used_mb=gpu_info.get("memory_used_mb", 0),
                memory_total_mb=gpu_info.get("memory_total_mb", 81920),
                temperature_celsius=gpu_info.get("temperature_celsius", 0.0),
                last_updated=datetime.fromisoformat(
                    gpu_info.get("last_updated", datetime.utcnow().isoformat())
                )
            ))
        
        return result
    
    @router.get(
        "/dashboard/queue",
        response_model=List[QueueBreakdown],
        summary="Get queue breakdown by weight class"
    )
    async def get_queue_breakdown(
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager)
    ):
        """Get detailed queue breakdown by weight class."""
        logger.info("Queue breakdown requested")
        
        # Get queue breakdown from meta-manager
        queue_data = await meta_manager.get_queue_breakdown()
        
        result = []
        for weight_class, data in queue_data.items():
            result.append(QueueBreakdown(
                weight_class=weight_class,
                count=data.get("count", 0),
                jobs=data.get("jobs", [])
            ))
        
        return result
    
    @router.get(
        "/dashboard/jobs/active",
        response_model=List[ActiveJobDetail],
        summary="Get active jobs with details"
    )
    async def get_active_jobs(
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager)
    ):
        """Get list of currently active jobs with detailed information."""
        logger.info("Active jobs requested")
        
        # Get active jobs from meta-manager
        active_jobs = await meta_manager.get_active_jobs()
        
        result = []
        for job in active_jobs:
            result.append(ActiveJobDetail(
                job_id=job.get("job_id"),
                status=job.get("status"),
                weight_class=job.get("weight_class"),
                miner_hotkey=job.get("miner_hotkey"),
                validator_hotkey=job.get("validator_hotkey"),
                priority=job.get("priority", 0),
                progress_percentage=job.get("progress_percentage", 0.0),
                current_phase=job.get("current_phase"),
                assigned_gpus=job.get("assigned_gpus", []),
                started_at=datetime.fromisoformat(job.get("started_at")),
                estimated_completion=datetime.fromisoformat(
                    job.get("estimated_completion")
                ) if job.get("estimated_completion") else None
            ))
        
        return result
    
    @router.get(
        "/dashboard/jobs/{job_id}/logs",
        response_model=JobLogs,
        summary="Get job logs"
    )
    async def get_job_logs(
        job_id: str,
        lines: int = 100,
        offset: int = 0,
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager)
    ):
        """
        Get execution logs for a specific job.
        
        Args:
            job_id: Job identifier
            lines: Number of log lines to return (default: 100)
            offset: Starting line offset (default: 0)
        """
        logger.info(f"Job logs requested: {job_id}")
        
        # Get logs from meta-manager
        log_data = await meta_manager.get_job_logs(job_id, lines, offset)
        
        if not log_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Logs not found for job: {job_id}"
            )
        
        return JobLogs(
            job_id=job_id,
            logs=log_data.get("logs", []),
            total_lines=log_data.get("total_lines", 0),
            has_more=log_data.get("has_more", False)
        )
    
    return router