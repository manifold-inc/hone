"""
Log Streaming API Routes

Provides endpoints for retrieving and streaming job execution logs
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
import logging

from api.routes import authenticate_request
from core.log_manager import get_log_manager, initialize_log_manager


logger = logging.getLogger(__name__)


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


def ensure_log_manager():
    """Ensure log manager is initialized and available"""
    try:
        return get_log_manager()
    except RuntimeError:
        try:
            return initialize_log_manager(retention_hours=1)
        except:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Log service not available"
            )


def create_logs_router() -> APIRouter:
    """Create logs API router."""
    router = APIRouter(tags=["logs"])
    
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
        log_manager = ensure_log_manager()
        
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
    
    return router