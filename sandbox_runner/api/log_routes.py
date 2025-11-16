"""
Log Streaming API Routes

Provides endpoints for retrieving and streaming job execution logs.
"""

from typing import Optional, Dict, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
import logging

from api.auth import AuthenticationManager
from api.routes import authenticate_request, get_meta_manager
from core.log_manager import get_log_manager

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class LogEntry(BaseModel):
    """Single log entry"""
    timestamp: str
    phase: str
    level: str
    message: str
    job_id: str


class LogStreamResponse(BaseModel):
    """Response for log stream request"""
    job_id: str
    entries: List[LogEntry]
    cursor_id: Optional[str] = None
    has_more: bool = False
    total_entries: int = 0
    current_position: int = 0
    phase_filter: Optional[str] = None


class LogStatsResponse(BaseModel):
    """Response for log statistics"""
    active_streams: int
    total_log_entries: int
    retention_hours: int
    active_jobs: List[str]


# ============================================================================
# Router Creation
# ============================================================================

def create_logs_router() -> APIRouter:
    """Create logs API router."""
    router = APIRouter(tags=["logs"])
    
    @router.get(
        "/logs/{job_id}",
        response_model=LogStreamResponse,
        summary="Get logs for a job"
    )
    async def get_job_logs(
        job_id: str,
        cursor_id: Optional[str] = Query(None, description="Cursor for incremental fetching"),
        limit: int = Query(1000, ge=1, le=10000, description="Maximum number of entries to return"),
        phase: Optional[str] = Query(None, description="Filter by phase (build, prep, inference, vllm)"),
        auth: tuple = Depends(authenticate_request)
    ):
        """
        Get logs for a specific job with cursor support.
        
        This endpoint supports incremental fetching using cursors:
        - First call: Don't provide cursor_id, returns initial logs and a cursor
        - Subsequent calls: Use the cursor_id to get only new logs
        
        Args:
            job_id: Job identifier
            cursor_id: Optional cursor for incremental fetching
            limit: Maximum number of log entries to return (1-10000)
            phase: Optional phase filter
        """
        try:
            log_manager = get_log_manager()
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Log service not available"
            )
        
        result = log_manager.get_logs(
            job_id=job_id,
            cursor_id=cursor_id,
            limit=limit,
            phase=phase
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        return LogStreamResponse(**result)
    
    @router.get(
        "/logs/{job_id}/all",
        response_model=LogStreamResponse,
        summary="Get all logs for a job"
    )
    async def get_all_job_logs(
        job_id: str,
        phase: Optional[str] = Query(None, description="Filter by phase"),
        auth: tuple = Depends(authenticate_request)
    ):
        """
        Get all logs for a job (no cursor/pagination).
        
        Warning: This can return a large amount of data for long-running jobs.
        Consider using the cursor-based endpoint for large logs.
        
        Args:
            job_id: Job identifier
            phase: Optional phase filter
        """
        try:
            log_manager = get_log_manager()
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Log service not available"
            )
        
        result = log_manager.get_all_logs(job_id=job_id, phase=phase)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        return LogStreamResponse(**result)
    
    @router.get(
        "/logs/{job_id}/tail",
        response_model=LogStreamResponse,
        summary="Get latest logs for a job"
    )
    async def tail_job_logs(
        job_id: str,
        lines: int = Query(100, ge=1, le=1000, description="Number of latest lines to return"),
        phase: Optional[str] = Query(None, description="Filter by phase"),
        auth: tuple = Depends(authenticate_request)
    ):
        """
        Get the latest N log entries for a job (like tail command).
        
        Args:
            job_id: Job identifier
            lines: Number of latest lines to return (1-1000)
            phase: Optional phase filter
        """
        try:
            log_manager = get_log_manager()
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Log service not available"
            )
        
        # Get all logs and return only the last N entries
        result = log_manager.get_all_logs(job_id=job_id, phase=phase)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        # Get only the last N entries
        entries = result.get("entries", [])
        if len(entries) > lines:
            result["entries"] = entries[-lines:]
        
        return LogStreamResponse(**result)
    
    @router.delete(
        "/logs/{job_id}",
        summary="Clear logs for a job"
    )
    async def clear_job_logs(
        job_id: str,
        auth: tuple = Depends(authenticate_request)
    ):
        """
        Clear all logs for a specific job.
        
        Args:
            job_id: Job identifier
        """
        try:
            log_manager = get_log_manager()
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Log service not available"
            )
        
        log_manager.clear_job_logs(job_id)
        
        return {
            "message": f"Logs cleared for job {job_id}",
            "job_id": job_id
        }
    
    @router.get(
        "/logs/stats",
        response_model=LogStatsResponse,
        summary="Get log service statistics"
    )
    async def get_log_stats(
        auth: tuple = Depends(authenticate_request)
    ):
        """Get statistics about the log service."""
        try:
            log_manager = get_log_manager()
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Log service not available"
            )
        
        stats = log_manager.get_stats()
        
        return LogStatsResponse(**stats)
    
    @router.get(
        "/logs/active",
        summary="Get list of jobs with active logs"
    )
    async def get_active_log_jobs(
        auth: tuple = Depends(authenticate_request)
    ):
        """Get list of job IDs that have active log streams."""
        try:
            log_manager = get_log_manager()
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Log service not available"
            )
        
        active_jobs = log_manager.get_active_jobs()
        
        return {
            "active_jobs": active_jobs,
            "count": len(active_jobs)
        }
    
    @router.get(
        "/logs/{job_id}/phases",
        summary="Get available phases for a job"
    )
    async def get_job_phases(
        job_id: str,
        auth: tuple = Depends(authenticate_request)
    ):
        """
        Get list of phases that have logs for a specific job.
        
        Args:
            job_id: Job identifier
        """
        try:
            log_manager = get_log_manager()
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Log service not available"
            )
        
        # Get all logs to determine available phases
        result = log_manager.get_all_logs(job_id=job_id)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        # Extract unique phases
        phases = set()
        for entry in result.get("entries", []):
            phases.add(entry.get("phase"))
        
        return {
            "job_id": job_id,
            "phases": sorted(list(phases)),
            "count": len(phases)
        }
    
    return router