"""
Log Manager Module

Handles persistence and streaming of job execution logs.
Provides:
- Log capture and storage during execution
- API access to historical logs with cursor support
- Automatic cleanup of old logs
- Per-job, per-phase log tracking
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import threading
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Single log entry with metadata"""
    timestamp: datetime
    phase: str  # build, prep, inference, vllm
    level: str  # info, warning, error, debug
    message: str
    job_id: str
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase,
            "level": self.level,
            "message": self.message,
            "job_id": self.job_id
        }


@dataclass
class LogStream:
    """Stream of logs for a specific job"""
    job_id: str
    entries: deque = field(default_factory=lambda: deque(maxlen=100000))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    cursors: Dict[str, int] = field(default_factory=dict)  # cursor_id -> position
    
    def append(self, phase: str, level: str, message: str):
        """Add a log entry to the stream"""
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            phase=phase,
            level=level,
            message=message,
            job_id=self.job_id
        )
        self.entries.append(entry)
        self.last_accessed = datetime.utcnow()
    
    def get_entries(
        self, 
        cursor_id: Optional[str] = None, 
        limit: int = 1000,
        phase: Optional[str] = None
    ) -> Tuple[List[LogEntry], str]:
        """
        Get log entries from stream
        
        Args:
            cursor_id: Cursor ID for incremental fetching
            limit: Maximum number of entries to return
            phase: Filter by phase (optional)
            
        Returns:
            Tuple of (entries, new_cursor_id)
        """
        # Create new cursor if not provided
        if cursor_id is None:
            cursor_id = str(uuid.uuid4())
            self.cursors[cursor_id] = 0
        
        # Get current position
        position = self.cursors.get(cursor_id, 0)
        
        # Filter entries
        all_entries = list(self.entries)
        if phase:
            filtered_entries = [e for e in all_entries if e.phase == phase]
        else:
            filtered_entries = all_entries
        
        # Get entries from position
        new_entries = filtered_entries[position:position + limit]
        
        # Update cursor position
        new_position = position + len(new_entries)
        self.cursors[cursor_id] = new_position
        
        # Clean up old cursors (not accessed in last hour)
        self._cleanup_old_cursors()
        
        return new_entries, cursor_id
    
    def _cleanup_old_cursors(self):
        """Remove cursors that haven't been used recently"""
        # Keep cursor cleanup simple for now
        if len(self.cursors) > 100:
            # Keep only the 50 most recent cursors
            keep_count = 50
            if len(self.cursors) > keep_count:
                sorted_cursors = sorted(self.cursors.items(), key=lambda x: x[1], reverse=True)
                self.cursors = dict(sorted_cursors[:keep_count])
    
    def get_all_entries(self, phase: Optional[str] = None) -> List[LogEntry]:
        """Get all entries, optionally filtered by phase"""
        if phase:
            return [e for e in self.entries if e.phase == phase]
        return list(self.entries)
    
    def clear(self):
        """Clear all log entries"""
        self.entries.clear()
        self.cursors.clear()


class LogManager:
    """
    Centralized log management for all jobs
    
    Features:
    - In-memory log storage with configurable retention
    - Per-job log streams
    - Cursor-based incremental fetching
    - Automatic cleanup of old logs
    - Thread-safe operations
    """
    
    def __init__(self, retention_hours: int = 1, persist_to_disk: bool = False):
        """
        Initialize log manager
        
        Args:
            retention_hours: How long to keep logs in memory
            persist_to_disk: Whether to also persist logs to disk (future feature)
        """
        self.retention_hours = retention_hours
        self.persist_to_disk = persist_to_disk
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._streams: Dict[str, LogStream] = {}
        
        # Start cleanup task
        self._cleanup_task = None
        self._running = False
        
        logger.info(f"LogManager initialized (retention={retention_hours}h)")
    
    async def start(self):
        """Start background cleanup task"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("LogManager cleanup task started")
    
    async def stop(self):
        """Stop background cleanup task"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("LogManager stopped")
    
    async def _cleanup_loop(self):
        """Background task to clean up old logs"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                self._cleanup_old_streams()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_old_streams(self):
        """Remove log streams older than retention period"""
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
            
            to_remove = []
            for job_id, stream in self._streams.items():
                if stream.last_accessed < cutoff_time:
                    to_remove.append(job_id)
            
            for job_id in to_remove:
                del self._streams[job_id]
                logger.debug(f"Removed old log stream for job {job_id}")
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old log streams")
    
    def create_stream(self, job_id: str) -> LogStream:
        """Create a new log stream for a job"""
        with self._lock:
            if job_id in self._streams:
                logger.warning(f"Log stream already exists for job {job_id}")
                return self._streams[job_id]
            
            stream = LogStream(job_id=job_id)
            self._streams[job_id] = stream
            logger.debug(f"Created log stream for job {job_id}")
            return stream
    
    def get_stream(self, job_id: str) -> Optional[LogStream]:
        """Get log stream for a job"""
        with self._lock:
            stream = self._streams.get(job_id)
            if stream:
                stream.last_accessed = datetime.utcnow()
            return stream
    
    def append_log(
        self, 
        job_id: str, 
        phase: str, 
        message: str, 
        level: str = "info"
    ):
        """
        Append a log entry to a job's stream
        
        Args:
            job_id: Job identifier
            phase: Execution phase (build, prep, inference, vllm)
            message: Log message
            level: Log level (info, warning, error, debug)
        """
        with self._lock:
            stream = self._streams.get(job_id)
            if not stream:
                stream = self.create_stream(job_id)
            
            stream.append(phase, level, message)
    
    def append_logs_batch(
        self,
        job_id: str,
        phase: str,
        messages: List[str],
        level: str = "info"
    ):
        """Append multiple log entries at once"""
        with self._lock:
            stream = self._streams.get(job_id)
            if not stream:
                stream = self.create_stream(job_id)
            
            for message in messages:
                stream.append(phase, level, message)
    
    def get_logs(
        self,
        job_id: str,
        cursor_id: Optional[str] = None,
        limit: int = 1000,
        phase: Optional[str] = None
    ) -> Dict:
        """
        Get logs for a job with cursor support
        
        Args:
            job_id: Job identifier
            cursor_id: Cursor for incremental fetching
            limit: Maximum number of entries
            phase: Filter by phase
            
        Returns:
            Dictionary with logs and metadata
        """
        with self._lock:
            stream = self._streams.get(job_id)
            if not stream:
                return {
                    "job_id": job_id,
                    "error": "No logs found for job",
                    "entries": [],
                    "cursor_id": None,
                    "has_more": False
                }
            
            entries, new_cursor_id = stream.get_entries(cursor_id, limit, phase)
            
            # Check if there are more entries
            total_entries = len(stream.entries)
            current_position = stream.cursors.get(new_cursor_id, 0)
            has_more = current_position < total_entries
            
            return {
                "job_id": job_id,
                "entries": [e.to_dict() for e in entries],
                "cursor_id": new_cursor_id,
                "has_more": has_more,
                "total_entries": total_entries,
                "current_position": current_position,
                "phase_filter": phase
            }
    
    def get_all_logs(self, job_id: str, phase: Optional[str] = None) -> Dict:
        """Get all logs for a job"""
        with self._lock:
            stream = self._streams.get(job_id)
            if not stream:
                return {
                    "job_id": job_id,
                    "error": "No logs found for job",
                    "entries": []
                }
            
            entries = stream.get_all_entries(phase)
            
            return {
                "job_id": job_id,
                "entries": [e.to_dict() for e in entries],
                "total_entries": len(entries),
                "phase_filter": phase
            }
    
    def clear_job_logs(self, job_id: str):
        """Clear all logs for a specific job"""
        with self._lock:
            if job_id in self._streams:
                self._streams[job_id].clear()
                logger.debug(f"Cleared logs for job {job_id}")
    
    def get_active_jobs(self) -> List[str]:
        """Get list of jobs with active log streams"""
        with self._lock:
            return list(self._streams.keys())
    
    def get_stats(self) -> Dict:
        """Get statistics about log storage"""
        with self._lock:
            total_entries = sum(len(s.entries) for s in self._streams.values())
            
            return {
                "active_streams": len(self._streams),
                "total_log_entries": total_entries,
                "retention_hours": self.retention_hours,
                "active_jobs": self.get_active_jobs()
            }


# Global log manager instance
log_manager: Optional[LogManager] = None


def get_log_manager() -> LogManager:
    """Get the global log manager instance"""
    global log_manager
    if log_manager is None:
        raise RuntimeError("LogManager not initialized")
    return log_manager


def initialize_log_manager(retention_hours: int = 1) -> LogManager:
    """Initialize the global log manager"""
    global log_manager
    if log_manager is None:
        log_manager = LogManager(retention_hours=retention_hours)
    return log_manager