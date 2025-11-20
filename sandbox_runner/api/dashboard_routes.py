from typing import List, Dict
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
import logging

from config import Config
from api.auth import AuthenticationManager, RateLimiter
from api.routes import (
        authenticate_request,
        get_meta_manager,
        get_config
    )

logger = logging.getLogger("api.dashboard")


# ============================================================================
# Response Models
# ============================================================================

class GPUStatusDetail(BaseModel):
    """Detailed GPU status for dashboard."""
    gpu_id: int
    status: str  # free, allocated, error, offline
    allocated_to_job: str | None = None
    utilization_percent: float = 0.0
    memory_used_mb: int = 0
    memory_total_mb: int = 81920  # H200 = 80GB
    temperature_celsius: float = 0.0
    last_updated: datetime | None = None
    

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
    validator_hotkey: str | None = None
    priority: int
    progress_percentage: float = 0.0
    current_phase: str | None = None
    assigned_gpus: List[int] = []
    started_at: datetime
    estimated_completion: datetime | None = None


class JobLogs(BaseModel):
    """Job execution logs."""
    job_id: str
    logs: List[Dict]
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
        
        runner_status = await meta_manager.get_runner_status()        
        gpu_stats = runner_status.get("gpu_stats", {})        
        queue_stats = runner_status.get("queue_stats", {})
        
        gpu_status = await meta_manager.get_gpu_status()
        avg_utilization = 0.0
        avg_temperature = 0.0
        
        if gpu_status:
            total_util = sum(gpu.get("utilization_percent", 0.0) for gpu in gpu_status.values())
            total_temp = sum(gpu.get("temperature_celsius", 0.0) for gpu in gpu_status.values())
            avg_utilization = total_util / len(gpu_status)
            avg_temperature = total_temp / len(gpu_status)
        
        total_finished = (
            runner_status.get("total_completed", 0) + 
            runner_status.get("total_failed", 0)
        )
        success_rate = 0.0
        if total_finished > 0:
            success_rate = (
                runner_status.get("total_completed", 0) / total_finished * 100
            )
        
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
            avg_gpu_utilization=round(avg_utilization, 2),
            avg_gpu_temperature=round(avg_temperature, 2),
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
        """Get detailed status of all GPUs with real metrics"""
        logger.info("GPU details requested")
        
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
    
    @router.get(
        "/dashboard/jobs/completed",
        summary="Get completed jobs with metrics"
    )
    async def get_completed_jobs(
        limit: int = 50,
        offset: int = 0,
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager)
    ):
        """
        Get list of completed jobs with their metrics.
        
        Args:
            limit: Maximum number of jobs to return
            offset: Offset for pagination
        """
        logger.info("Completed jobs requested")
        
        completed_jobs = []
        
        # get jobs from history
        history_items = list(meta_manager._job_history.items())
        for job_id, job in history_items[offset:offset+limit]:
            if job.status.value in ["completed", "failed", "timeout"]:
                job_data = {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "miner_hotkey": job.miner_hotkey,
                    "validator_hotkey": job.validator_hotkey,
                    "weight_class": job.weight_class.value,
                    "submitted_at": job.submitted_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error_message": job.error_message,
                }
                
                if job.status.value == "completed" and hasattr(job, 'metrics') and job.metrics:
                    job_data["metrics"] = job.metrics.get("aggregate", {})
                    job_data["has_metrics"] = True
                else:
                    job_data["has_metrics"] = False
                
                if job.started_at and job.completed_at:
                    job_data["execution_time"] = (job.completed_at - job.started_at).total_seconds()
                
                completed_jobs.append(job_data)
        
        # also check persisted results
        results_dir = Path("/app/data/job_results")
        if results_dir.exists():
            result_files = sorted(
                results_dir.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            for result_file in result_files:
                job_id = result_file.stem
                
                # skip if already in completed_jobs
                if any(j["job_id"] == job_id for j in completed_jobs):
                    continue
                
                try:
                    import json
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                    
                    job_data = {
                        "job_id": result_data.get("job_id"),
                        "status": result_data.get("status"),
                        "miner_hotkey": result_data.get("miner_hotkey"),
                        "completed_at": result_data.get("completed_at"),
                        "error_message": result_data.get("error_message"),
                        "execution_time": result_data.get("execution_time"),
                        "has_metrics": bool(result_data.get("metrics")),
                    }
                    
                    if result_data.get("metrics"):
                        job_data["metrics"] = result_data["metrics"].get("aggregate", {})
                    
                    completed_jobs.append(job_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to load result file {result_file}: {e}")
                
                # respect limit
                if len(completed_jobs) >= limit:
                    break
        
        return {
            "jobs": completed_jobs,
            "total": len(completed_jobs),
            "limit": limit,
            "offset": offset
        }
    
    return router