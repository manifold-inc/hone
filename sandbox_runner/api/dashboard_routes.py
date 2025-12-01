from typing import List, Dict
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import logging

from config import Config
from api.routes import (
        authenticate_request,
        get_meta_manager,
        get_config
    )

logger = logging.getLogger("api.dashboard")
    

def create_dashboard_router(config: Config) -> APIRouter:
    """Create dashboard API router."""
    router = APIRouter(tags=["dashboard"])
    
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