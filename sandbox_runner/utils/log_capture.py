"""
Log Capture Utilities

Provides utilities for capturing logs from various sources (Docker, subprocess, etc.)
and sending them to the LogManager for persistence.
"""

import asyncio
import logging
from typing import Optional, Callable, List
from pathlib import Path

from core.log_manager import get_log_manager

logger = logging.getLogger(__name__)


class LogCapture:
    """
    Captures logs from various sources and sends to LogManager
    
    Can be used as a context manager or standalone
    """
    
    def __init__(
        self, 
        job_id: str, 
        phase: str,
        show_terminal: bool = True
    ):
        """
        Initialize log capture
        
        Args:
            job_id: Job identifier
            phase: Execution phase (build, prep, inference, vllm)
            show_terminal: Whether to show logs in terminal
        """
        self.job_id = job_id
        self.phase = phase
        self.show_terminal = show_terminal
        self.log_manager = None
        
        try:
            self.log_manager = get_log_manager()
        except RuntimeError:
            logger.warning("LogManager not initialized, logs will not be persisted")
    
    def capture_line(self, line: str, level: str = "info"):
        """Capture a single log line"""
        if self.log_manager:
            self.log_manager.append_log(
                job_id=self.job_id,
                phase=self.phase,
                message=line.rstrip("\n\r"),
                level=level
            )
    
    def capture_lines(self, lines: List[str], level: str = "info"):
        """Capture multiple log lines"""
        if self.log_manager:
            clean_lines = [line.rstrip("\n\r") for line in lines]
            self.log_manager.append_logs_batch(
                job_id=self.job_id,
                phase=self.phase,
                messages=clean_lines,
                level=level
            )
    
    async def capture_stream(
        self, 
        stream, 
        level: str = "info",
        line_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Capture from an async stream
        
        Args:
            stream: Async stream to read from
            level: Log level
            line_callback: Optional callback for each line (e.g., for display)
        """
        buffer = ""
        
        async for chunk in stream:
            if chunk:
                # Handle bytes
                if isinstance(chunk, bytes):
                    text = chunk.decode('utf-8', errors='replace')
                else:
                    text = chunk
                
                buffer += text
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line or buffer:  # Don't process empty lines unless there's more content
                        self.capture_line(line, level)
                        if line_callback:
                            line_callback(line)
        
        # Capture remaining buffer
        if buffer:
            self.capture_line(buffer, level)
            if line_callback:
                line_callback(buffer)
    
    def capture_docker_build_logs(self, build_generator, display=None):
        """
        Capture Docker build logs
        
        Args:
            build_generator: Docker build log generator
            display: Optional BuildLogDisplay for terminal output
        """
        for entry in build_generator:
            if 'error' in entry:
                message = f"ERROR: {entry['error']}"
                self.capture_line(message, "error")
                if display and self.show_terminal:
                    display.update(message)
                    
            elif 'stream' in entry:
                for line in entry['stream'].splitlines():
                    if line.strip():
                        self.capture_line(line, "info")
                        if display and self.show_terminal:
                            display.update(line)
                            
            elif 'status' in entry:
                message = f"{entry.get('id', '')} {entry['status']}".strip()
                if message:
                    self.capture_line(message, "info")
                    if display and self.show_terminal:
                        display.update(message)
    
    def capture_docker_container_logs(self, container, display=None):
        """
        Capture Docker container logs in real-time
        
        Args:
            container: Docker container object
            display: Optional BuildLogDisplay for terminal output
        """
        try:
            for chunk in container.logs(stdout=True, stderr=True, stream=True, follow=True):
                try:
                    text = chunk.decode("utf-8", errors="replace")
                except Exception:
                    text = str(chunk)
                
                for line in text.splitlines():
                    if line.strip():
                        self.capture_line(line, "info")
                        if display and self.show_terminal:
                            display.update(line)
                            
        except Exception as e:
            error_msg = f"[log-stream] ERROR: {e}"
            self.capture_line(error_msg, "error")
            if display and self.show_terminal:
                display.update(error_msg)
    
    async def capture_docker_container_logs_async(self, container, display=None):
        """
        Async version of Docker container log capture
        
        Args:
            container: Docker container object
            display: Optional BuildLogDisplay for terminal output
        """
        loop = asyncio.get_event_loop()
        
        def _capture_blocking():
            self.capture_docker_container_logs(container, display)
        
        await loop.run_in_executor(None, _capture_blocking)


class BuildLogDisplayAdapter:
    """
    Adapter for BuildLogDisplay that also captures to LogManager
    """
    
    def __init__(
        self, 
        original_display, 
        job_id: str, 
        phase: str,
        show_terminal: bool = True
    ):
        """
        Initialize display adapter
        
        Args:
            original_display: Original BuildLogDisplay instance
            job_id: Job identifier
            phase: Execution phase
            show_terminal: Whether to show in terminal
        """
        self.original = original_display
        self.capture = LogCapture(job_id, phase, show_terminal)
        self.show_terminal = show_terminal
    
    def start(self):
        """Start the display"""
        if self.show_terminal and self.original:
            self.original.start()
    
    def update(self, line: str):
        """Update display and capture log"""
        # Always capture to log manager
        self.capture.capture_line(line)
        
        # Only show in terminal if enabled
        if self.show_terminal and self.original:
            self.original.update(line)
    
    def write_below_box(self, text: str):
        """Write below the box"""
        if self.show_terminal and self.original:
            self.original.write_below_box(text)
    
    def end(self, status: str = None):
        """End the display"""
        if status:
            self.capture.capture_line(f"[STATUS] {status}", "info")
        
        if self.show_terminal and self.original:
            if status:
                self.original.end(status)
            else:
                self.original.end()
    
    @property
    def box_active(self):
        """Check if box is active"""
        if self.original:
            return self.original.box_active
        return False


def create_log_display(
    job_id: str,
    phase: str,
    show_terminal: bool = True,
    box_lines: int = 50,
    title: str = None
):
    """
    Factory function to create appropriate log display
    
    Args:
        job_id: Job identifier
        phase: Execution phase
        show_terminal: Whether to show in terminal
        box_lines: Number of lines in terminal box
        title: Box title
        
    Returns:
        BuildLogDisplayAdapter that captures logs
    """
    # Import here to avoid circular dependency
    from execution.docker_only import BuildLogDisplay
    
    if title is None:
        title = f"🔨 {phase.upper()} LOGS"
    
    # Create original display only if terminal output is enabled
    if show_terminal:
        original = BuildLogDisplay(box_lines=box_lines, title=title)
    else:
        original = None
    
    # Return adapter that always captures logs
    return BuildLogDisplayAdapter(original, job_id, phase, show_terminal)