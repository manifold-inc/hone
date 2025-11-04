"""
Prometheus Metrics Module

Provides Prometheus metrics for monitoring the sandbox runner:
- Job submission rates
- Job success/failure rates
- GPU utilization
- Queue depth
- Execution times
- Resource usage
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class MetricsManager:
    """
    Central manager for Prometheus metrics.
    
    Provides a single interface for recording metrics across the application.
    """
    
    def __init__(self):
        """Initialize metrics manager with Prometheus registry."""
        self.registry = CollectorRegistry()
        self._initialized = False
        
        # Initialize all metrics
        self._init_job_metrics()
        self._init_gpu_metrics()
        self._init_queue_metrics()
        self._init_performance_metrics()
        self._init_system_metrics()
    
    def _init_job_metrics(self):
        """Initialize job-related metrics."""
        # Job submission counter
        self.jobs_submitted = Counter(
            "sandbox_jobs_submitted_total",
            "Total number of jobs submitted",
            labelnames=["validator", "weight_class"],
            registry=self.registry
        )
        
        # Job completion counter with status
        self.jobs_completed = Counter(
            "sandbox_jobs_completed_total",
            "Total number of jobs completed",
            labelnames=["status", "weight_class", "validator"],
            registry=self.registry
        )
        
        # Currently active jobs
        self.jobs_active = Gauge(
            "sandbox_jobs_active",
            "Number of currently active jobs",
            labelnames=["phase", "weight_class"],
            registry=self.registry
        )
        
        # Job execution duration
        self.job_duration = Histogram(
            "sandbox_job_duration_seconds",
            "Job execution duration in seconds",
            labelnames=["phase", "weight_class"],
            buckets=[10, 30, 60, 300, 600, 1800, 3600, 7200],
            registry=self.registry
        )
    
    def _init_gpu_metrics(self):
        """Initialize GPU-related metrics."""
        # GPU utilization per device
        self.gpu_utilization = Gauge(
            "sandbox_gpu_utilization_percent",
            "GPU utilization percentage",
            labelnames=["gpu_id"],
            registry=self.registry
        )
        
        # GPU memory usage
        self.gpu_memory_used = Gauge(
            "sandbox_gpu_memory_used_bytes",
            "GPU memory used in bytes",
            labelnames=["gpu_id"],
            registry=self.registry
        )
        
        # GPU memory total
        self.gpu_memory_total = Gauge(
            "sandbox_gpu_memory_total_bytes",
            "Total GPU memory in bytes",
            labelnames=["gpu_id"],
            registry=self.registry
        )
        
        # GPU allocation state
        self.gpu_allocated = Gauge(
            "sandbox_gpu_allocated",
            "GPU allocation state (1=allocated, 0=free)",
            labelnames=["gpu_id"],
            registry=self.registry
        )
        
        # GPU temperature
        self.gpu_temperature = Gauge(
            "sandbox_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            labelnames=["gpu_id"],
            registry=self.registry
        )
    
    def _init_queue_metrics(self):
        """Initialize queue-related metrics."""
        # Queue depth by priority
        self.queue_depth = Gauge(
            "sandbox_queue_depth",
            "Number of jobs in queue",
            labelnames=["priority", "weight_class"],
            registry=self.registry
        )
        
        # Queue wait time
        self.queue_wait_time = Histogram(
            "sandbox_queue_wait_seconds",
            "Time spent waiting in queue",
            labelnames=["weight_class"],
            buckets=[10, 30, 60, 300, 600, 1800, 3600],
            registry=self.registry
        )
    
    def _init_performance_metrics(self):
        """Initialize performance-related metrics."""
        # API request duration
        self.api_request_duration = Histogram(
            "sandbox_api_request_duration_seconds",
            "API request duration",
            labelnames=["method", "endpoint", "status"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5],
            registry=self.registry
        )
        
        # API request counter
        self.api_requests = Counter(
            "sandbox_api_requests_total",
            "Total API requests",
            labelnames=["method", "endpoint", "status"],
            registry=self.registry
        )
        
        # Rate limit hits
        self.rate_limit_hits = Counter(
            "sandbox_rate_limit_hits_total",
            "Number of rate limit violations",
            labelnames=["validator"],
            registry=self.registry
        )
        
        # Authentication failures
        self.auth_failures = Counter(
            "sandbox_auth_failures_total",
            "Authentication failures",
            labelnames=["auth_type", "reason"],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system-level metrics."""
        # System info
        self.system_info = Info(
            "sandbox_system",
            "System information",
            registry=self.registry
        )
        
        # CPU usage
        self.cpu_usage = Gauge(
            "sandbox_cpu_usage_percent",
            "CPU usage percentage",
            registry=self.registry
        )
        
        # Memory usage
        self.memory_used = Gauge(
            "sandbox_memory_used_bytes",
            "Memory used in bytes",
            registry=self.registry
        )
        
        self.memory_total = Gauge(
            "sandbox_memory_total_bytes",
            "Total memory in bytes",
            registry=self.registry
        )
        
        # Disk usage
        self.disk_used = Gauge(
            "sandbox_disk_used_bytes",
            "Disk space used in bytes",
            labelnames=["mount"],
            registry=self.registry
        )
        
        self.disk_total = Gauge(
            "sandbox_disk_total_bytes",
            "Total disk space in bytes",
            labelnames=["mount"],
            registry=self.registry
        )
    
    def initialize(self):
        """Initialize metrics with system information."""
        if self._initialized:
            return
        
        # Set system info
        self.system_info.info({
            "version": "1.0.0",
            "service": "hone-subnet-sandbox-runner"
        })
        
        self._initialized = True
        logger.info("Metrics initialized")
    
    def export_metrics(self) -> tuple[bytes, str]:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Tuple of (metrics_bytes, content_type)
        """
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
    
    # ========================================================================
    # Convenience methods for recording metrics
    # ========================================================================
    
    def record_job_submitted(self, validator: str, weight_class: str):
        """Record a job submission."""
        self.jobs_submitted.labels(
            validator=validator,
            weight_class=weight_class
        ).inc()
    
    def record_job_completed(
        self,
        status: str,
        weight_class: str,
        validator: str,
        duration: float
    ):
        """Record a job completion with duration."""
        self.jobs_completed.labels(
            status=status,
            weight_class=weight_class,
            validator=validator
        ).inc()
        
        # Record duration (use 'total' for overall duration)
        self.job_duration.labels(
            phase="total",
            weight_class=weight_class
        ).observe(duration)
    
    def record_job_phase_duration(
        self,
        phase: str,
        weight_class: str,
        duration: float
    ):
        """Record duration of a specific job phase."""
        self.job_duration.labels(
            phase=phase,
            weight_class=weight_class
        ).observe(duration)
    
    def set_gpu_utilization(self, gpu_id: int, utilization: float):
        """Set GPU utilization percentage."""
        self.gpu_utilization.labels(gpu_id=str(gpu_id)).set(utilization)
    
    def set_gpu_memory(self, gpu_id: int, used: int, total: int):
        """Set GPU memory usage."""
        self.gpu_memory_used.labels(gpu_id=str(gpu_id)).set(used)
        self.gpu_memory_total.labels(gpu_id=str(gpu_id)).set(total)
    
    def set_gpu_allocated(self, gpu_id: int, allocated: bool):
        """Set GPU allocation state."""
        self.gpu_allocated.labels(gpu_id=str(gpu_id)).set(1 if allocated else 0)
    
    def set_gpu_temperature(self, gpu_id: int, temperature: float):
        """Set GPU temperature."""
        self.gpu_temperature.labels(gpu_id=str(gpu_id)).set(temperature)
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Record an API request."""
        self.api_requests.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).observe(duration)
    
    def record_rate_limit_hit(self, validator: str):
        """Record a rate limit violation."""
        self.rate_limit_hits.labels(validator=validator).inc()
    
    def record_auth_failure(self, auth_type: str, reason: str):
        """Record an authentication failure."""
        self.auth_failures.labels(
            auth_type=auth_type,
            reason=reason
        ).inc()


# Global metrics manager instance
metrics_manager = MetricsManager()